#!/usr/bin/env bash
set -euo pipefail

echo "[run_server] Setting HF cache env vars..."
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE=./models

# After your first successful model download, you can uncomment this for fully offline runs:
export HF_HUB_OFFLINE=1

echo "[run_server] Starting API server on http://0.0.0.0:8000 ..."
uvicorn src.api:app --host 0.0.0.0 --port 8000


