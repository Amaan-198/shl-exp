from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from .config import RESULT_MAX
from .catalog_build import load_catalog_snapshot
from .retrieval import retrieve_candidates
from .rerank import rerank_candidates
from .mmr import load_item_embeddings, mmr_select
from .balance import allocate
from .mapping import map_items_to_response


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: str


app = FastAPI()

# CORS open for dev; restrict in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_catalog_df = None  # cache snapshot in-process


@app.on_event("startup")
def startup_event():
    global _catalog_df
    logger.info("Starting app warmup...")
    _catalog_df = load_catalog_snapshot()
    logger.info("Loaded catalog snapshot with {} rows", len(_catalog_df))

    # Warm-load indices/models so first request is fast
    try:
        from .retrieval import _load_retrieval_components
        _load_retrieval_components()
        from .mmr import load_item_embeddings as _load_emb
        _load_emb()
    except Exception as e:
        logger.warning("Warmup partial failure: {}", e)
    logger.info("Warmup complete.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy")


@app.post("/recommend")
def recommend(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must be non-empty")

    raw_query, cleaned_query, fused = retrieve_candidates(query)
    ranked = rerank_candidates(cleaned_query, fused)

    # Diversify with MMR
    embeddings, ids = load_item_embeddings()
    mmr_ids = mmr_select(
        candidates=[(c.item_id, c.rerank_score) for c in ranked],
        embeddings=embeddings,
        k=RESULT_MAX,
        lambda_=0.7,
    )

    # Detect intent (K vs P) and allocate with small domain filter using snapshot
    from .balance import DENY_SUBSTR  # just to log what’s filtered
    logger.info("Domain deny substrings: {}", DENY_SUBSTR)

    # Build class map for allocator
    classes = {}
    for iid in mmr_ids:
        row = _catalog_df.loc[iid]
        classes[iid] = list(row.get("test_type") or [])

    final_ids = allocate(mmr_ids, classes, RESULT_MAX, pt=1.0, pb=0.0, catalog_df=_catalog_df)  # NOTE: fill with your real pt/pb

    items = map_items_to_response(final_ids, _catalog_df)
    # 5–10 policy (mapping already preserves order)
    if len(items) > RESULT_MAX:
        items = items[:RESULT_MAX]

    return {"recommended_assessments": items}
