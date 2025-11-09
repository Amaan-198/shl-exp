# src/_singletons.py
from functools import lru_cache
from .embed_index import load_bm25_cache, load_dense_components
from .catalog_build import load_catalog_snapshot

@lru_cache(maxsize=1)
def get_catalog_df():
    return load_catalog_snapshot()

@lru_cache(maxsize=1)
def get_bm25():
    return load_bm25_cache()

@lru_cache(maxsize=1)
def get_dense():
    # returns (model, faiss_index, id_map, dense_stale)
    return load_dense_components()
