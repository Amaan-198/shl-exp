from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger

from .config import (
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    IDS_MAPPING_PATH,
    FUSION_TOP_K,
    FUSION_WINSORIZE_MIN,
    FUSION_WINSORIZE_MAX,
    FUSION_EPS,
    # Align weights with spec (dense a bit stronger)
    # Doc says ~0.45/0.55 — we keep them here to avoid hidden drift.
)
from .embed_index import load_bm25 as load_bm25_cache, load_dense_components, embed_query_with_chunking

# Explicit weights here to avoid confusion with older configs
BM25_WEIGHT = 0.45
DENSE_WEIGHT = 0.55


def _winsorize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def _to_z(arr: np.ndarray, eps: float = FUSION_EPS) -> np.ndarray:
    m = arr.mean() if arr.size else 0.0
    s = arr.std() if arr.size else 1.0
    if s < eps:
        s = eps
    return (arr - m) / s


def _normalize_slice(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    ids, scores = zip(*pairs)
    scores = np.asarray(scores, dtype="float32")
    scores = _to_z(_winsorize(scores, FUSION_WINSORIZE_MIN, FUSION_WINSORIZE_MAX))
    return {i: float(s) for i, s in zip(ids, scores)}


def fuse_scores(
    bm25: List[Tuple[int, float]],
    dense: List[Tuple[int, float]],
    top_k: int = FUSION_TOP_K,
) -> List[Tuple[int, float]]:
    nz_bm25 = _normalize_slice(bm25)
    nz_dense = _normalize_slice(dense)

    all_ids = set(nz_bm25) | set(nz_dense)
    fused: List[Tuple[int, float]] = []
    for i in all_ids:
        s = BM25_WEIGHT * nz_bm25.get(i, 0.0) + DENSE_WEIGHT * nz_dense.get(i, 0.0)
        fused.append((i, s))

    fused.sort(key=lambda x: (-x[1], x[0]))
    return fused[:top_k]


def _load_retrieval_components():
    bm25 = load_bm25_cache(BM25_INDEX_PATH)
    model, faiss_index, id_map, dense_stale = load_dense_components(
        FAISS_INDEX_PATH, IDS_MAPPING_PATH
    )
    return bm25, model, faiss_index, id_map, dense_stale


def retrieve_bm25(cleaned_query: str) -> List[Tuple[int, float]]:
    bm25 = load_bm25_cache(BM25_INDEX_PATH)
    return bm25.query(cleaned_query)


def retrieve_dense(model, faiss_index, id_map, query_vec) -> List[Tuple[int, float]]:
    # cosine similarity top-N; index returns ids and scores
    D, I = faiss_index.search(query_vec[np.newaxis, :], 200)
    out: List[Tuple[int, float]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        item_id = id_map[str(idx)]
        out.append((int(item_id), float(dist)))
    return out


def retrieve_candidates(raw_query: str) -> Tuple[str, str, List[Tuple[int, float]]]:
    from .normalize import normalize_for_lexical_index as clean_text

    cleaned_query = clean_text(raw_query)
    bm25, model, faiss_index, id_map, dense_stale = _load_retrieval_components()

    bm25_results = retrieve_bm25(cleaned_query)

    if dense_stale:
        dense_results: List[Tuple[int, float]] = []
        logger.warning("Dense index stale or missing — falling back to BM25 only.")
    else:
        q_vec = embed_query_with_chunking(model, cleaned_query)
        dense_results = retrieve_dense(model, faiss_index, id_map, q_vec)

    fused = fuse_scores(bm25_results, dense_results)
    logger.info("Retrieved {} fused candidates for query", len(fused))
    return raw_query, cleaned_query, fused


if __name__ == "__main__":
    q = input("Enter query or JD URL: ")
    _, cleaned, results = retrieve_candidates(q)
    print("CLEANED:", cleaned)
    for rid, score in results[:10]:
        print(f"{rid}: {score:.4f}")
