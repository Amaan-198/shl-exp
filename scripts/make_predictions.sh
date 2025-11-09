#!/usr/bin/env bash
set -euo pipefail

echo "[make_predictions] Generating test predictions CSV (top-10)..."
python -m src.eval --mode test --k 10
echo "[make_predictions] Done. See artifacts/test_predictions.csv"


