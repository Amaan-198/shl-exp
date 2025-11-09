# SHL Gen AI Assessment Recommender

A production-focused retrieval and recommendation service for SHL's Individual Test
Solutions catalog. The system ingests free-form job descriptions or URLs, maps them to
well-balanced assessment bundles, and exposes both an automated FastAPI endpoint and a
CLI for batch generation.

## Repository layout

| Path | Description |
| ---- | ----------- |
| `src/` | All application code. Key modules include the FastAPI entrypoint (`api.py`), hybrid retrieval (`retrieval.py`), cross-encoder reranking (`rerank.py`), diversification (`mmr.py`), category balancing (`balance.py`), normalization utilities, and catalog/index builders. |
| `src/utils/` | Helper utilities for text cleanup, URL canonicalisation, intent analysis, and logging shared across the pipeline. |
| `data/` | Versioned catalog snapshot (`catalog_snapshot.parquet`) plus the SHL-provided train/test workbooks consumed by the CLI and evaluation harness. |
| `indices/` | Persisted BM25 cache, FAISS index, and embedding matrices used at runtime. |
| `artifacts/` | Generated outputs such as train/test prediction CSVs and metric JSON snapshots. |
| `scripts/` | One-command wrappers for catalog building, index construction, server startup, and batch prediction runs. |
| `tests/` | Pytest suite covering contract compliance, retrieval quality, normalization, indexing, reranking, balancing, and mapping. |
| `models/` | Local Hugging Face cache directory for the dense encoder, reranker, and zero-shot classifier. |

## Pipeline walkthrough

The full request flow is orchestrated by `src/api.py` and executed via `run_full_pipeline`:

1. **Input normalisation and retrieval** – The request text (or fetched JD content) is normalised and expanded with intent-specific seeds before hybrid retrieval fuses BM25 and FAISS scores with duration-aware post filtering.【F:src/api.py†L704-L735】【F:src/retrieval.py†L430-L471】
2. **Cross-encoder reranking** – The top fused candidates are rescored by BAAI/bge-reranker-base for semantic relevance and converted into `ScoredCandidate` objects.【F:src/api.py†L736-L759】
3. **Post-ranking heuristics** – Domain filtering, must-include pinning, duration guards, and category-aware boosts inject SHL-specific business rules while preserving determinism.【F:src/api.py†L759-L824】
4. **Diversification & balancing** – Maximal Marginal Relevance selects a diverse slate which is then split across Knowledge/Skills vs. Personality/Behavior buckets using the learned allocation policy.【F:src/api.py†L826-L895】
5. **Dynamic policy enforcement** – A soft target cutoff, category diversity guard, family top-up, and must-include checks keep the list within result policy bounds (5–10 items).【F:src/api.py†L897-L953】
6. **API mapping** – Final item IDs are converted to schema-validated response objects with canonicalised URLs for evaluation friendliness.【F:src/api.py†L955-L973】

Supporting components include:

- **Catalog snapshot & normalization** – `src/catalog_build.py` and `src/normalize.py` clean raw catalog data, preserve technical tokens, and apply synonym mapping to align retrieval and rerank behaviour.【F:src/catalog_build.py†L525-L533】【F:src/normalize.py†L18-L63】
- **Index loaders** – `src/embed_index.py` handles BM25 cache hydration, dense embedding loading, and chunked query embedding to keep latency low even on CPU-only machines.【F:src/embed_index.py†L199-L226】【F:src/embed_index.py†L374-L414】
- **Intent utilities** – `src/intent.py`, `src/intent_utils.py`, and `src/constants.py` translate domain triggers into curated seed sets that bias retrieval toward proven assessment families.【F:src/intent.py†L16-L82】【F:src/intent_utils.py†L21-L82】【F:src/constants.py†L3-L73】
- **Post-processing helpers** – `src/post_processing.py` encapsulates all deterministic business heuristics (generic penalties, domain vetoes, duration windows, family pinning, and diversity guards) to keep the main pipeline readable.【F:src/post_processing.py†L9-L110】

## Approach & performance optimisation

We treated evaluation on the SHL training set (Recall@K by canonical family slug) as the
going quality bar. The key optimisation milestones were:

| Variant | Description | Recall@1 | Recall@5 | Recall@10 |
| ------- | ----------- | -------- | -------- | --------- |
| **Baseline hybrid retrieval** | Directly map the fused BM25 results (intent expansion + duration-aware ordering) into API responses without reranking or heuristics. | 0.30 | 0.50 | 0.60 |
| **Full pipeline** | Enable reranking, SHL-specific post-processing, MMR diversification, and K/P allocation before policy clamping. | 0.60 | 0.60 | 0.60 |

How we got there:

1. **Baseline measurement** – Using a short Python snippet (see below) we collected top-10
predictions from `retrieve_candidates` and evaluated them with `src.debug_eval`. This
surfaced gaps on leadership and communication-heavy briefs despite decent Recall@10.【43d4a4†L1-L5】
2. **Hybrid retrieval tuning** – Intent-driven BM25 expansion (`_expand_query_with_intents`) and duration-aware reordering reduced lexical misses while keeping FAISS lookups optional offline, stabilising Recall@5.【F:src/retrieval.py†L60-L129】【F:src/retrieval.py†L330-L360】
3. **Reranking + heuristics** – Plugging the cross-encoder reranker and domain-specific post ranking rules doubled Recall@1, fixed over-long bundles, and enforced SHL-mandated families.【F:src/api.py†L736-L824】
4. **Diversification & balancing** – MMR plus the allocation policy from `src/balance.py` maintained coverage of both technical and behavioural intents without sacrificing hit rate.【F:src/api.py†L826-L953】【F:src/balance.py†L20-L120】
5. **Evaluation loop** – The CLI warms caches, deduplicates queries, and writes the exact two-column CSV expected by graders, making repeated offline experiments fast. Final metrics were confirmed via `python -m src.debug_eval --train_csv data/gen_ai_train.xlsx --preds_csv artifacts/train_predictions.csv`.【201fa3†L1-L5】【F:src/cli.py†L62-L147】

Recreate the baseline check locally:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
from src.catalog_build import load_catalog_snapshot
from src.retrieval import retrieve_candidates
from src.mapping import map_items_to_response
from src.config import TRAIN_PATH, RESULT_MAX

catalog = load_catalog_snapshot().set_index("item_id", drop=False)
queries = pd.read_excel(TRAIN_PATH)["Query"].astype(str)
rows = []
for query in queries:
    _, _, fused = retrieve_candidates(query)
    top_ids = [iid for iid, _ in fused[:RESULT_MAX]]
    response = map_items_to_response(top_ids, catalog)
    for item in response.recommended_assessments:
        rows.append({"Query": query, "Assessment_url": item.url})
Path("artifacts/baseline_predictions.csv").write_text(pd.DataFrame(rows).to_csv(index=False))
PY
python -m src.debug_eval --train_csv data/gen_ai_train.xlsx --preds_csv artifacts/baseline_predictions.csv --show_hits 0
```

Then run the full pipeline for comparison:

```bash
python -m src.cli --mode train
python -m src.debug_eval --train_csv data/gen_ai_train.xlsx --preds_csv artifacts/train_predictions.csv --show_hits 0
```

## Running the service

1. **Environment setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Build data artifacts**
   ```bash
   bash scripts/build_catalog.sh
   bash scripts/build_indices.sh
   ```

3. **Start the API**
   ```bash
   bash scripts/run_server.sh
   # POST http://localhost:8000/recommend with {"query": "..."}
   ```

4. **Batch predictions & metrics**
   ```bash
   python -m src.cli --mode test
   python -m src.debug_eval --train_csv data/gen_ai_train.xlsx --preds_csv artifacts/train_predictions.csv
   ```

All stages respect the result policy from `src/config.py` and are CPU-friendly/offline
by default thanks to cached indices and Hugging Face download guards. Jenkins-style
smoke tests include `python -m compileall src` and `pytest` against the suites in
`tests/` for regression protection.

## Support & governance

- Raise issues with reproduction details (query payloads, expectation) via the tracker.
- Escalate urgent incidents through SHL's Gen AI on-call rotation.
- Treat catalog and embedding artifacts as SHL-confidential and handle them under the
organisation's data governance policies.
