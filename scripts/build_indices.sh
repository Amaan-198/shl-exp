# D:\projects\SHL\shl-rag-recommender\scripts\build_indices.sh
#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Remove old indices"
rm -f indices/bm25.pkl indices/faiss.index indices/item_embeddings.npy indices/ids.json || true

echo "[2/3] Env for HF cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE=./models

echo "[3/3] Build all indices (BM25 + Dense + FAISS)"
python - <<'PY'
from src.embed_index import build_all_indices
build_all_indices()
PY

echo "Done."


