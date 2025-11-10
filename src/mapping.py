from __future__ import annotations
"""
Mapping utilities to convert internal catalog rows into API responses.

Centralises the mapping logic from the catalog snapshot into the Pydantic
schemas (AssessmentItem / RecommendResponse) and applies global result-size
policy consistently.
"""

import re
from typing import Iterable, List, Sequence, Dict, Any

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    class _FallbackLogger:
        def __init__(self, logger): self._logger = logger
        def info(self, msg: str, *args, **kwargs) -> None: self._logger.info(msg.format(*args))
        def warning(self, msg: str, *args, **kwargs) -> None: self._logger.warning(msg.format(*args))
        def error(self, msg: str, *args, **kwargs) -> None: self._logger.error(msg.format(*args))
        def exception(self, msg: str, *args, **kwargs) -> None: self._logger.exception(msg.format(*args))
    logger = _FallbackLogger(logging.getLogger(__name__))

import pandas as pd  # type: ignore
import numpy as np

from .config import AssessmentItem, RecommendResponse, RESULT_MIN, RESULT_MAX

CATALOG_URL_BASE = "https://www.shl.com/solutions/products/product-catalog/view/"


def _extract_catalog_slug(url: str) -> str:
    """Return the slug portion of a catalog URL, best-effort."""
    if not isinstance(url, str) or not url:
        return ""
    match = re.search(r"/view/([^/?#]+)", url)
    slug = match.group(1) if match else url.rstrip("/").split("/")[-1]
    return slug.strip("/")


def _normalise_catalog_url(url: str) -> str:
    """Ensure catalog URLs share the /solutions/products/... base path."""
    if not isinstance(url, str) or not url:
        return ""
    slug = _extract_catalog_slug(url)
    if not slug:
        return CATALOG_URL_BASE
    return f"{CATALOG_URL_BASE}{slug.strip('/')}/"


def _ensure_iterable_ids(item_ids: Iterable[int]) -> List[int]:
    """Normalize IDs to a list[int]."""
    if isinstance(item_ids, pd.Series):
        return [int(x) for x in item_ids.tolist()]
    return [int(x) for x in item_ids]


def _clip_result_count(n: int) -> int:
    """Clip number of requested results into [RESULT_MIN, RESULT_MAX]."""
    if n < RESULT_MIN:
        return RESULT_MIN
    if n > RESULT_MAX:
        return RESULT_MAX
    return n


def _coerce_int(val, default: int = 0) -> int:
    try:
        if val is None:
            return default
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, float) and not np.isnan(val):
            return int(val)
        s = str(val).strip()
        return int(float(s)) if s else default
    except Exception:
        return default


def _build_assessment_item(row: pd.Series) -> AssessmentItem:
    """
    Convert a catalog row into an AssessmentItem.

    Robust handling of test_type shapes:
      - NaN / None -> []
      - "a, b, c" -> ["a","b","c"]
      - list/tuple/np.ndarray -> list[str]
    """
    url = _normalise_catalog_url(row.get("url", ""))
    name = str(row.get("name", "") or "").strip()
    desc = str(row.get("description", "") or "").strip()

    duration = _coerce_int(row.get("duration", 0), default=0)

    adaptive = str(row.get("adaptive_support", "") or "").strip() or "No"
    remote  = str(row.get("remote_support", "") or "").strip() or "No"

    raw_tt = row.get("test_type", None)
    test_types: List[str] = []
    if isinstance(raw_tt, (list, tuple, np.ndarray)):
        test_types = [str(v).strip() for v in raw_tt if str(v).strip()]
    elif raw_tt is None:
        test_types = []
    else:
        try:
            if not pd.isna(raw_tt):
                s = str(raw_tt).strip()
                if s:
                    test_types = [t.strip() for t in s.split(",")] if "," in s else [s]
        except Exception:
            # treat as scalar string
            s = str(raw_tt).strip()
            test_types = [s] if s else []

    item = AssessmentItem(
        url=url,
        name=name,
        description=desc,
        duration=duration,
        adaptive_support=adaptive,
        remote_support=remote,
        test_type=test_types or [],
    )
    item.ensure_flags_are_literal()
    return item


def map_items_to_response(
    item_ids: Sequence[int],
    catalog_df: pd.DataFrame,
) -> RecommendResponse:
    """
    Convert a sequence of item IDs into a RecommendResponse.
    Deduplicates while preserving first-seen order, then clips to policy.
    """
    if catalog_df is None:
        raise ValueError("catalog_df must be provided to map_items_to_response")

    ids = _ensure_iterable_ids(item_ids)
    seen = set()
    deduped: List[int] = []
    for iid in ids:
        if iid not in seen:
            seen.add(iid)
            deduped.append(iid)

    if not deduped:
        logger.warning("map_items_to_response called with empty ID list")
        return RecommendResponse(recommended_assessments=[])

    n_final = _clip_result_count(len(deduped))
    final_ids = deduped[:n_final]

    # determine id column
    if "item_id" in catalog_df.columns:
        id_col = "item_id"
    elif "id" in catalog_df.columns:
        id_col = "id"
    else:
        raise KeyError("Catalog DataFrame must contain 'item_id' or 'id'.")

    # build index
    required_cols = ["url", "name", "description", "duration",
                     "adaptive_support", "remote_support", "test_type"]
    missing = [c for c in required_cols if c not in catalog_df.columns]
    for c in missing:
        catalog_df[c] = None  # fill if missing

    id_to_row: Dict[int, pd.Series] = {
        int(row[id_col]): row
        for _, row in catalog_df[[id_col] + required_cols].iterrows()
    }

    items: List[AssessmentItem] = []
    for iid in final_ids:
        row = id_to_row.get(iid)
        if row is None:
            logger.warning("Item id {} not found in catalog DataFrame; skipping", iid)
            continue
        try:
            items.append(_build_assessment_item(row))
        except Exception as e:
            logger.exception("Failed to build AssessmentItem for id {}: {}", iid, e)

    logger.info("Mapped {} items into API schema", len(items))
    return RecommendResponse(recommended_assessments=items)
