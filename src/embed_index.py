from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .catalog_build import load_catalog_snapshot
from .config import (
    BGE_ENCODER_MODEL,
    BM25_INDEX_PATH,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    HF_ENV_VARS,
    IDS_MAPPING_PATH,
)
from .normalize import lexical_tokens_for_bm25


# ---------------------------
# HF env helpers
# ---------------------------

def _ensure_hf_env() -> None:
    """
    Ensure key HF env vars are set (cache dir, transfer optimization).

    We do NOT force HF_HUB_OFFLINE here; that should be controlled by the runtime
    / shell scripts after the first online pull.
    """
    for key, val in HF_ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = val


# ---------------------------
# BM25 index building
# ---------------------------

def prepare_corpus_for_bm25(search_texts: Sequence[str]) -> List[List[str]]:
    """
    Turn a list of search_text strings into a tokenized corpus suitable for BM25.

    - Uses the same lexical normalization + synonym expansion as queries.
    - Each document becomes a list of tokens.
    """
    corpus_tokens: List[List[str]] = []
    for text in search_texts:
        tokens = lexical_tokens_for_bm25(text or "")
        corpus_tokens.append(tokens)
    return corpus_tokens


def build_bm25_index(
    search_texts: Sequence[str],
    item_ids: Sequence[int],
    output_path: Path = BM25_INDEX_PATH,
) -> Path:
    """
    Build a BM25Okapi index over the given search_text corpus.

    Persisted object is a pickle containing:
    {
        "bm25": BM25Okapi instance,
        "item_ids": [int, ...]  # same order as corpus/docs
    }
    """
    logger.info("Building BM25 index over {} documents", len(search_texts))

    corpus_tokens = prepare_corpus_for_bm25(search_texts)
    bm25 = BM25Okapi(corpus_tokens)

    index_obj = {
        "bm25": bm25,
        "item_ids": list(item_ids),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(index_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("BM25 index written to {}", output_path)
    return output_path


# ---------------------------
# Dense embeddings + FAISS
# ---------------------------

def _load_bge_encoder() -> SentenceTransformer:
    """
    Load the BGE encoder (base, en v1.5) with our HF cache hints.
    """
    _ensure_hf_env()
    logger.info("Loading dense encoder model: {}", BGE_ENCODER_MODEL)
    model = SentenceTransformer(BGE_ENCODER_MODEL)
    return model


def build_dense_index(
    search_texts: Sequence[str],
    item_ids: Sequence[int],
    embeddings_path: Path = EMBEDDINGS_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
    ids_mapping_path: Path = IDS_MAPPING_PATH,
) -> None:
    """
    Build dense embeddings with BGE and a FAISS IndexFlatIP over L2-normalized vectors.

    Saves:
    - embeddings_path: np.ndarray float32 (num_items, dim), allow_pickle=False
    - faiss_index_path: FAISS index
    - ids_mapping_path: JSON mapping row index -> item_id (list of ints)
    """
    logger.info("Building dense embeddings for {} documents", len(search_texts))

    model = _load_bge_encoder()

    # SentenceTransformers can normalize embeddings for us
    embeddings = model.encode(
        list(search_texts),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize
    )

    # Ensure float32 dtype
    embeddings = embeddings.astype("float32", copy=False)

    # Save embeddings
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings, allow_pickle=False)
    logger.info("Saved item embeddings to {}", embeddings_path)

    # Build FAISS index (cosine via inner product over normalized vectors)
    dim = embeddings.shape[1]
    logger.info("Creating FAISS IndexFlatIP with dim={}", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path))
    logger.info("FAISS index written to {}", faiss_index_path)

    # Save ids mapping (row index -> item_id)
    mapping = list(item_ids)
    ids_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with ids_mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f)
    logger.info("IDs mapping written to {}", ids_mapping_path)


# ---------------------------
# Orchestrator
# ---------------------------

def build_all_indices(catalog_path: Path | None = None) -> None:
    """
    High-level entrypoint:

    1) Load catalog snapshot
    2) Build BM25 index
    3) Build dense embeddings + FAISS + ids mapping
    """
    if catalog_path is None:
        # load_catalog_snapshot already knows the default path from config
        df = load_catalog_snapshot()
    else:
        df = load_catalog_snapshot(catalog_path)

    if "search_text" not in df.columns or "item_id" not in df.columns:
        raise ValueError(
            "Catalog snapshot is missing required columns 'search_text' and/or 'item_id'. "
            "Run the catalog build pipeline first."
        )

    search_texts = df["search_text"].astype(str).tolist()
    item_ids = df["item_id"].tolist()

    logger.info("Starting index build for {} catalog items", len(df))

    build_bm25_index(search_texts=search_texts, item_ids=item_ids)
    build_dense_index(
        search_texts=search_texts,
        item_ids=item_ids,
    )

    logger.info("All indices built successfully.")

# ---------------------------
# BM25 + FAISS loading helpers
# ---------------------------

def load_bm25() -> BM25Okapi:
    """
    Load the BM25 index and item_ids from the saved pickle file.
    """
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Run the index build first.")
    logger.info("Loading BM25 index from {}", BM25_INDEX_PATH)
    with BM25_INDEX_PATH.open("rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    item_ids = data["item_ids"]
    logger.info("Loaded BM25 index with {} items", len(item_ids))
    return bm25, item_ids


def load_dense_components() -> tuple[faiss.IndexFlatIP, np.ndarray, list[int]]:
    """
    Load FAISS index, dense embeddings, and item ID mapping.
    """
    if not (FAISS_INDEX_PATH.exists() and EMBEDDINGS_PATH.exists() and IDS_MAPPING_PATH.exists()):
        raise FileNotFoundError("Missing dense index components; run the index build first.")

    logger.info("Loading FAISS index from {}", FAISS_INDEX_PATH)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    logger.info("Loading embeddings from {}", EMBEDDINGS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    logger.info("Loading ID mapping from {}", IDS_MAPPING_PATH)
    with IDS_MAPPING_PATH.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return index, embeddings, mapping


# ---------------------------
# CLI entrypoint
# ---------------------------

if __name__ == "__main__":
    # Manual usage:
    # python -m src.embed_index
    build_all_indices()
