#!/usr/bin/env bash
set -euo pipefail

echo "[build_catalog] Crawling SHL Individual Test Solutions catalog..."
python -m src.catalog_crawl
echo "[build_catalog] Done. Snapshot at data/catalog_snapshot.parquet"


