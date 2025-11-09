#!/usr/bin/env bash
set -euo pipefail

echo "[build_catalog] Building catalog snapshot (xlsx if present, else crawl fallback)..."
python -m src.catalog_build
echo "[build_catalog] Done. Snapshot at data/catalog_snapshot.parquet (or .csv fallback)"
