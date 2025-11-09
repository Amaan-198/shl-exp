# src/rerank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from . import config
from .catalog_build import load_catalog_snapshot

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception as e:
    CrossEncoder = None
    _import_err = e

# ---------------------------------------------------------------------------
# HF model handling
# ---------------------------------------------------------------------------

RERANKER_CANDIDATES: List[str] = [
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

_RERANKER: Optional["CrossEncoder"] = None
_RERANKER_IS_DUMMY: bool = False


def load_reranker() -> Optional["CrossEncoder"]:
    """
    Load and cache a CrossEncoder. Respects HF offline cache via env.
    """
    global _RERANKER, _RERANKER_IS_DUMMY

    if _RERANKER is not None or _RERANKER_IS_DUMMY:
        return _RERANKER

    if CrossEncoder is None:
        logger.warning("sentence_transformers not available: {}", _import_err)
        _RERANKER_IS_DUMMY = True
        return None

    for rid in RERANKER_CANDIDATES:
        try:
            logger.info("Loading cross-encoder reranker: {}", rid)
            model = CrossEncoder(rid, device="cpu")
            _RERANKER = model
            logger.info("Loaded cross-encoder reranker: {}", rid)
            return _RERANKER
        except Exception as e:
            logger.warning("Failed to load CrossEncoder '{}': {}", rid, e)

    logger.warning("No cross-encoder available; falling back to dummy reranker.")
    _RERANKER_IS_DUMMY = True
    _RERANKER = None
    return None


# ---------------------------------------------------------------------------
# Public structures and helpers
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    item_id: int
    fused_score: float
    rerank_score: float


def build_candidate_text(row: pd.Series) -> str:
    """
    Build a text string for cross-encoder reranking.
    Include name + description + test_type + flags for extra lexical anchors.
    """
    name = str(row.get("name", "") or "").strip()
    desc = str(row.get("description", "") or "").strip()
    ttypes = row.get("test_type", [])
    if isinstance(ttypes, str):
        ttypes = [x.strip() for x in ttypes.split(",") if x.strip()]
    elif not isinstance(ttypes, (list, tuple)):
        ttypes = []
    ttypes_str = " ".join(str(x) for x in ttypes)

    adaptive = str(row.get("adaptive_support", "") or "").strip()
    remote = str(row.get("remote_support", "") or "").strip()

    bits = [name, desc]
    if ttypes_str:
        bits.append(f"Types: {ttypes_str}.")
    if adaptive == "Yes":
        bits.append("adaptive.")
    if remote == "Yes":
        bits.append("remote.")
    text = " ".join(b for b in bits if b).strip()
    return text


def get_candidate_texts(
    item_ids: Sequence[int],
    catalog_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    if catalog_df is None:
        catalog_df = load_catalog_snapshot()

    texts: List[str] = []
    for iid in item_ids:
        try:
            row = catalog_df.loc[iid]
        except Exception:
            texts.append("")
            continue
        texts.append(build_candidate_text(row))
    return texts


def score_with_model(
    model,
    query: str,
    candidate_texts: Sequence[str],
) -> np.ndarray:
    if not candidate_texts:
        return np.zeros((0,), dtype="float32")

    if model is None:
        return np.zeros((len(candidate_texts),), dtype="float32")

    pairs = [(query, t) for t in candidate_texts]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.warning("Reranker model.predict failed; falling back to zeros: {}", e)
        return np.zeros((len(candidate_texts),), dtype="float32")

    return np.asarray(scores, dtype="float32")


def rerank_pairs(
    model,
    query: str,
    docs: List[str],
) -> List[Tuple[int, float]]:
    if not docs:
        return []

    if model is None:
        # dummy: keep original order with strictly descending scores
        base = 1.0
        step = 1.0 / max(len(docs), 1)
        return [(i, base - i * step) for i in range(len(docs))]

    scores = score_with_model(model, query, docs)
    ranked = sorted(
        [(i, float(s)) for i, s in enumerate(scores)],
        key=lambda x: -x[1],
    )
    return ranked


# ---------------------------------------------------------------------------
# Deterministic final ordering with URL tie-break
# ---------------------------------------------------------------------------

def _canon_url(u: str) -> str:
    # minimal, dependency-free canonicalization for deterministic sort
    return (u or "").strip().lower().rstrip("/")


def _build_url_map(item_ids: Sequence[int], catalog_df: Optional[pd.DataFrame]) -> dict[int, str]:
    if catalog_df is None:
        try:
            catalog_df = load_catalog_snapshot()
        except Exception:
            catalog_df = None

    url_map: dict[int, str] = {}
    if catalog_df is not None:
        col = "url" if "url" in catalog_df.columns else None
        if col:
            for iid in item_ids:
                try:
                    url_map[int(iid)] = str(catalog_df.loc[int(iid)][col])
                except Exception:
                    url_map[int(iid)] = ""
    return url_map


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def rerank_candidates(
    query_text: str,
    fused_candidates: List[Tuple[int, float]],
    cutoff: Optional[int] = None,
    catalog_df: Optional[pd.DataFrame] = None,
    model=None,
) -> List[Candidate]:
    """
    High-level rerank:
      1) take top-N fused (N=cutoff or RERANK_CUTOFF)
      2) score with cross-encoder (or dummy)
      3) sort by (-rerank_score, url) deterministically
      4) backfill to RESULT_MIN..RESULT_MAX if the model returned too few
    """
    if not fused_candidates:
        return []

    # Cutoff
    if cutoff is None:
        cutoff = min(config.RERANK_CUTOFF, len(fused_candidates))
    else:
        cutoff = min(cutoff, len(fused_candidates))

    fused_top = fused_candidates[:cutoff]
    item_ids = [int(iid) for iid, _ in fused_top]
    fused_scores = [float(score) for _, score in fused_top]

    # Load texts (and make sure catalog_df is available for URL map & safety)
    if catalog_df is None:
        try:
            catalog_df = load_catalog_snapshot()
        except Exception as e:
            logger.warning("Failed to load catalog snapshot in rerank_candidates; proceeding without texts: {}", e)
            catalog_df = None

    if catalog_df is not None:
        texts = get_candidate_texts(item_ids, catalog_df=catalog_df)
    else:
        texts = [str(iid) for iid in item_ids]

    # Model
    if model is None:
        model = load_reranker()


    # --- rerank ---
    idx_scores = rerank_pairs(model, query_text, texts)

    # Build sortable triples so ties are deterministic:
    triples = []
    for local_idx, score in idx_scores:
        iid = item_ids[local_idx]
        fused = fused_scores[local_idx]
        # sort by: rerank_score desc, fused_score desc, item_id asc
        triples.append((iid, float(score), float(fused)))

    triples.sort(key=lambda t: (-t[1], -t[2], t[0]))

    ranked: List[Candidate] = [
        Candidate(item_id=int(iid), fused_score=float(fused), rerank_score=float(rscore))
        for (iid, rscore, fused) in triples
    ]


    need = max(config.RESULT_MIN, min(config.RESULT_MAX, cutoff))
    seen = {c.item_id for c in ranked}
    if len(ranked) < need:
        for iid, fused in fused_top:
            if iid in seen:
                continue
            ranked.append(Candidate(item_id=int(iid), fused_score=float(fused), rerank_score=float(-1e6)))
            seen.add(iid)
            if len(ranked) >= need:
                break


    return ranked
