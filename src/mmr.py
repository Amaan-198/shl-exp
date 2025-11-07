from __future__ import annotations

import numpy as np
from loguru import logger
from typing import List, Sequence, Tuple

from .config import MMR_LAMBDA, RESULT_DEFAULT_TARGET, EMBEDDINGS_PATH, IDS_MAPPING_PATH


# ---------------------------
# Embedding utilities
# ---------------------------

def load_item_embeddings() -> tuple[np.ndarray, list[int]]:
    """
    Load the saved item embedding matrix (float32, L2-normalized)
    and the corresponding item_id mapping list.
    """
    import json
    import os

    logger.info("Loading item embeddings and ID mapping")

    emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    with open(IDS_MAPPING_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)

    if emb.shape[0] != len(ids):
        raise ValueError(f"Embeddings ({emb.shape[0]}) and id map ({len(ids)}) length mismatch")

    logger.info("Loaded embeddings: shape={}, items={}", emb.shape, len(ids))
    return emb, ids


def get_embedding_for_id(item_id: int, emb: np.ndarray, ids: list[int]) -> np.ndarray:
    """
    Lookup the vector for a specific item_id.
    """
    try:
        idx = ids.index(item_id)
    except ValueError:
        raise KeyError(f"Item id {item_id} not found in id map")

    return emb[idx]


# ---------------------------
# MMR core algorithm
# ---------------------------

def mmr_select(
    candidates: Sequence[Tuple[int, float]],
    embeddings: np.ndarray,
    ids: list[int],
    k: int = RESULT_DEFAULT_TARGET,
    lambda_: float = MMR_LAMBDA,
) -> List[int]:
    """
    Apply Maximal Marginal Relevance (MMR) to diversify reranked results.

    candidates: [(item_id, relevance_score)] sorted by relevance desc.
    embeddings: item embedding matrix (normalized)
    ids: list mapping FAISS row -> item_id
    k: target number of items to select
    lambda_: relevance vs novelty tradeoff (0.7 = 70% relevance / 30% novelty)
    """
    if not candidates:
        return []

    # Map item_id -> embedding row index
    id_to_index = {id_: idx for idx, id_ in enumerate(ids)}

    # Precompute embedding vectors for these candidates
    chosen_ids = [cid for cid, _ in candidates]
    vecs = np.stack([embeddings[id_to_index[cid]] for cid in chosen_ids])
    relevance_scores = np.array([score for _, score in candidates], dtype="float32")

    # Cosine similarities between candidates (since embeddings are normalized)
    sim_matrix = np.dot(vecs, vecs.T)
    n = len(candidates)

    selected: list[int] = []
    unselected = list(range(n))

    # Select first item (highest relevance)
    selected.append(unselected.pop(0))

    while len(selected) < min(k, n) and unselected:
        mmr_scores = []
        for idx in unselected:
            rel = relevance_scores[idx]
            if not selected:
                div = 0
            else:
                div = np.max(sim_matrix[idx, selected])
            mmr = lambda_ * rel - (1 - lambda_) * div
            mmr_scores.append(mmr)

        next_index = unselected[int(np.argmax(mmr_scores))]
        selected.append(next_index)
        unselected.remove(next_index)

    final_ids = [chosen_ids[i] for i in selected]
    logger.info("MMR selected {} items (λ={})", len(final_ids), lambda_)
    return final_ids
