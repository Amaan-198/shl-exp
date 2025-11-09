from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple
import json
import numpy as np  # type: ignore

try:
    from loguru import logger  # type: ignore
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from .config import EMBEDDINGS_PATH, IDS_MAPPING_PATH, MMR_LAMBDA

# ---- process-wide caches ----
_EMB = None
_IDS = None

def load_item_embeddings(
    embeddings_path: Path = EMBEDDINGS_PATH,
    ids_path: Path = IDS_MAPPING_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cached load with validation + L2 normalisation."""
    global _EMB, _IDS
    if _EMB is not None and _IDS is not None:
        return _EMB, _IDS

    if not embeddings_path.exists() or not ids_path.exists():
        logger.warning(
            "Embedding or ID mapping files missing ({} / {}); using dummy zero vectors.",
            embeddings_path,
            ids_path,
        )
        _EMB, _IDS = np.zeros((0, 1), dtype="float32"), np.array([], dtype=int)
        return _EMB, _IDS

    logger.info("Loading item embeddings from {}", embeddings_path)
    try:
        emb = np.load(embeddings_path, allow_pickle=False)
    except Exception as e:
        logger.warning("Failed to load embeddings from {}: {}", embeddings_path, e)
        _EMB, _IDS = np.zeros((0, 1), dtype="float32"), np.array([], dtype=int)
        return _EMB, _IDS

    try:
        with ids_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning("Failed to load ID mapping from {}: {}", ids_path, e)
        _EMB, _IDS = np.zeros((0, 1), dtype="float32"), np.array([], dtype=int)
        return _EMB, _IDS

    if isinstance(raw, dict):
        try:
            ids_list = [int(v) for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
        except Exception:
            try:
                max_idx = max(int(k) for k in raw.keys())
                ids_list = [int(raw[str(i)]) for i in range(max_idx + 1) if str(i) in raw]
            except Exception:
                ids_list = [int(v) for v in raw.values()]
    elif isinstance(raw, list):
        ids_list = [int(v) for v in raw]
    else:
        ids_list = [int(raw)]

    if emb.ndim != 2:
        raise ValueError(f"Item embeddings must be 2D (N,D). Got {emb.shape}. Rebuild the index.")
    if emb.shape[1] <= 1:
        raise ValueError(f"Embedding dimension {emb.shape[1]} looks wrong. Rebuild the index.")
    if len(ids_list) != emb.shape[0]:
        logger.warning(
            "Embedding rows ({}) != ID rows ({}); truncating to min length.",
            emb.shape[0], len(ids_list),
        )
        m = min(emb.shape[0], len(ids_list))
        emb = emb[:m]
        ids_list = ids_list[:m]
    else:
        logger.info("Loaded embeddings: shape={} items={}", emb.shape, len(ids_list))

    # L2 normalise
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    _EMB = emb.astype("float32", copy=False)
    _IDS = np.array(ids_list, dtype=int)
    return _EMB, _IDS


def mmr_select(
    candidates: Sequence[Tuple[int, float]],
    embeddings: np.ndarray,
    ids: np.ndarray,
    k: int,
    lambda_: float = MMR_LAMBDA,
) -> List[int]:
    """
    Select up to ``k`` items using Maximal Marginal Relevance (MMR).

    Parameters
    ----------
    candidates :
        Sequence of ``(item_id, score)`` pairs, typically rerank scores.
    embeddings :
        Array of shape ``(n_items, dim)`` with L2‑normalised vectors.
    ids :
        Array of shape ``(n_items,)`` mapping row index -> item_id.
    k :
        Maximum number of items to return.
    lambda_ :
        Tradeoff between relevance and diversity.  ``1.0`` = relevance only.

    Returns
    -------
    List[int]
        Selected item IDs in order.
    """
    # If embeddings are unavailable, fall back to top-k by score
    if embeddings.size == 0 or ids.size == 0 or not candidates:
        return [item_id for item_id, _ in candidates[:k]]

    # Build mapping from item_id -> row index
    id_to_row = {int(item_id): idx for idx, item_id in enumerate(ids)}

    # Filter candidates to those that exist in the embedding index
    filtered: List[Tuple[int, float]] = [
        (int(iid), float(score)) for iid, score in candidates if int(iid) in id_to_row
    ]
    if not filtered:
        return [item_id for item_id, _ in candidates[:k]]

    # Normalise scores into [0, 1]
    scores = np.array([s for _, s in filtered], dtype="float32")
    s_min, s_max = float(scores.min()), float(scores.max())
    s_range = s_max - s_min if s_max > s_min else 1.0
    norm_scores = (scores - s_min) / s_range

    selected: List[int] = []
    selected_vecs: List[np.ndarray] = []

    while len(selected) < min(k, len(filtered)):
        best_idx = -1
        best_mmr = -1e9

        for i, (iid, base_score) in enumerate(filtered):
            if iid in selected:
                continue

            row = id_to_row[int(iid)]
            vec = embeddings[row]

            # Relevance component (already normalised)
            rel = norm_scores[i]

            # Diversity component: max similarity to already selected items
            if not selected_vecs:
                div = 0.0
            else:
                sims = [float(np.dot(vec, v)) for v in selected_vecs]
                div = max(sims) if sims else 0.0

            mmr_score = lambda_ * rel - (1.0 - lambda_) * div
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx < 0:
            break

        chosen_id = int(filtered[best_idx][0])
        selected.append(chosen_id)
        row = id_to_row[chosen_id]
        selected_vecs.append(embeddings[row])

    return selected

