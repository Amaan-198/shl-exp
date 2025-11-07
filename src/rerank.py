from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import CrossEncoder

from .config import RERANK_CUTOFF
from .catalog_build import load_catalog_snapshot as load_catalog_df


@dataclass
class Candidate:
    item_id: int
    fused_score: float
    rerank_score: float


def build_candidate_text(row: pd.Series) -> str:
    """
    Richer text for cross-encoder:
      name. description. test_type words. flags (adaptive|remote).
    All short and human-readable (row is already normalized).
    """
    name = str(row.get("name", "")).strip()
    desc = str(row.get("description", "")).strip()
    types = row.get("test_type", [])
    if isinstance(types, str):
        # Fallback if something slipped through
        types = [t.strip() for t in types.replace("[", "").replace("]", "").replace("'", "").split(",") if t.strip()]
    flags = []
    if str(row.get("adaptive_support", "")).strip().lower() == "yes":
        flags.append("adaptive")
    if str(row.get("remote_support", "")).strip().lower() == "yes":
        flags.append("remote")

    parts: List[str] = []
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    if types:
        parts.append(" ".join(types))
    if flags:
        parts.append(" ".join(flags))

    return ". ".join(p for p in parts if p)


def get_candidate_texts(item_ids: Sequence[int], catalog_df: Optional[pd.DataFrame] = None) -> List[str]:
    if catalog_df is None:
        catalog_df = load_catalog_df()
    texts: List[str] = []
    for iid in item_ids:
        if iid not in catalog_df.index:
            logger.warning("Item id {} not found in catalog; using empty text.", iid)
            texts.append("")
            continue
        row = catalog_df.loc[iid]
        texts.append(build_candidate_text(row))
    return texts


def load_reranker() -> CrossEncoder:
    from .config import BGE_RERANKER_MODEL
    return CrossEncoder(BGE_RERANKER_MODEL)


def score_with_model(model: CrossEncoder, query_text: str, candidate_texts: Sequence[str]) -> np.ndarray:
    pairs = [(query_text, c) for c in candidate_texts]
    scores = model.predict(pairs)
    return np.asarray(scores, dtype="float32")


def rerank_candidates(
    query_text: str,
    fused_candidates: Sequence[Tuple[int, float]],
    cutoff: Optional[int] = None,
    *,
    catalog_df: Optional[pd.DataFrame] = None,
    model: Optional[CrossEncoder] = None,
) -> List[Candidate]:
    if cutoff is None:
        cutoff = RERANK_CUTOFF
    if not fused_candidates:
        return []

    top = list(fused_candidates)[:cutoff]
    item_ids = [iid for iid, _ in top]
    fused_scores = [fs for _, fs in top]

    if catalog_df is None:
        catalog_df = load_catalog_df()
    if model is None:
        model = load_reranker()

    candidate_texts = get_candidate_texts(item_ids, catalog_df=catalog_df)
    logger.info("Running reranker on {} candidates (cutoff={})", len(candidate_texts), cutoff)
    scores = score_with_model(model, query_text, candidate_texts)

    candidates = [
        Candidate(item_id=iid, fused_score=float(fused), rerank_score=float(rscore))
        for iid, fused, rscore in zip(item_ids, fused_scores, scores)
    ]
    candidates.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return candidates
