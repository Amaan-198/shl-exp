#!/usr/bin/env bash
set -euo pipefail

echo "[make_predictions] Generating test predictions CSV (dynamic 5–10)..."

# We must call the CLI runner (not the evaluator). Let the pipeline decide final count (5..10).
python -m src.cli --mode test --topk 10

echo "[make_predictions] Done. See artifacts/test_predictions.csv"
