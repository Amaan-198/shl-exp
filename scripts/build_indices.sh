#!/usr/bin/env bash
set -euo pipefail

echo "[build_indices] Setting HF cache env vars..."
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE=./models

echo "[build_indices] Building BM25 + FAISS indices..."
python -m src.embed_index
echo "[build_indices] Done. Indices in ./indices"
