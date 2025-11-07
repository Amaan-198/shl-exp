from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .config import TRAIN_PATH, TEST_PATH, RESULT_MAX, PROJECT_ROOT
from .api import run_full_pipeline

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ---------------------------
# Data loading
# ---------------------------

def load_train_df(path: Path = TRAIN_PATH) -> pd.DataFrame:
    logger.info("Loading train data from {}", path)
    df = pd.read_excel(path)
    return df


def load_test_df(path: Path = TEST_PATH) -> pd.DataFrame:
    logger.info("Loading test data from {}", path)
    df = pd.read_excel(path)
    return df


# ---------------------------
# Gold / predictions structures
# ---------------------------

def build_gold_from_train(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build mapping: Query -> set of relevant Assessment_url from train sheet.
    """
    gold: Dict[str, Set[str]] = {}
    if "Query" not in df.columns or "Assessment_url" not in df.columns:
        raise ValueError("Train sheet must contain 'Query' and 'Assessment_url' columns.")

    for _, row in df.iterrows():
        q = str(row["Query"]).strip()
        url = str(row["Assessment_url"]).strip()
        if not q or not url:
            continue
        if q not in gold:
            gold[q] = set()
        gold[q].add(url)

    logger.info("Built gold labels for {} unique queries", len(gold))
    return gold


def predict_for_queries(
    queries: Sequence[str],
    top_k: int = RESULT_MAX,
) -> Dict[str, List[str]]:
    """
    Run the full recommend pipeline for each query and collect top_k URLs.
    """
    preds: Dict[str, List[str]] = {}
    for q in tqdm(queries, desc="Predicting", unit="query"):
        try:
            resp = run_full_pipeline(q)
            urls = [item.url for item in resp.recommended_assessments][:top_k]
        except Exception as e:
            logger.exception("Prediction failed for query '{}': {}", q, e)
            urls = []
        preds[q] = urls
    return preds


# ---------------------------
# Metrics
# ---------------------------

def recall_at_k(gold: Set[str], preds: List[str], k: int) -> float:
    """
    Recall@k for a single query.
    """
    if not gold:
        return 0.0
    top = preds[:k]
    hits = len(gold.intersection(set(top)))
    return hits / len(gold)


def mean_recall_at_k(
    gold: Dict[str, Set[str]],
    preds: Dict[str, List[str]],
    k: int,
) -> float:
    """
    Mean Recall@k across all queries.
    """
    scores: List[float] = []
    for q, gold_urls in gold.items():
        p = preds.get(q, [])
        r = recall_at_k(gold_urls, p, k)
        scores.append(r)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------
# Train evaluation
# ---------------------------

def evaluate_train(k: int = 10) -> float:
    df_train = load_train_df()
    gold = build_gold_from_train(df_train)
    queries = list(gold.keys())
    logger.info("Running predictions on {} train queries", len(queries))

    preds = predict_for_queries(queries, top_k=k)
    mr = mean_recall_at_k(gold, preds, k=k)

    metrics = {
        "k": k,
        "mean_recall_at_k": mr,
        "num_queries": len(queries),
    }

    metrics_path = ARTIFACTS_DIR / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Train metrics written to {}: {}", metrics_path, metrics)

    return mr


# ---------------------------
# Test predictions CSV
# ---------------------------

def make_test_predictions(
    k: int = RESULT_MAX,
    output_path: Path | None = None,
) -> Path:
    """
    Generate test predictions CSV with exact headers: Query,Assessment_url
    """
    df_test = load_test_df()
    if "Query" not in df_test.columns:
        raise ValueError("Test sheet must contain 'Query' column.")

    queries = [str(q).strip() for q in df_test["Query"].unique() if str(q).strip()]
    logger.info("Generating predictions for {} test queries", len(queries))

    preds = predict_for_queries(queries, top_k=k)

    rows: List[Dict[str, str]] = []
    for q in queries:
        urls = preds.get(q, [])
        for url in urls:
            rows.append(
                {
                    "Query": q,
                    "Assessment_url": url,
                }
            )

    if output_path is None:
        output_path = ARTIFACTS_DIR / "test_predictions.csv"

    df_out = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info("Test predictions CSV written to {}", output_path)

    return output_path


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="SHL GenAI Recommender Evaluation")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="What to run: train eval, test predictions, or both.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Top-k cutoff (default 10).",
    )

    args = parser.parse_args()

    if args.mode in {"train", "both"}:
        mr = evaluate_train(k=args.k)
        print(f"Mean Recall@{args.k} (train): {mr:.4f}")

    if args.mode in {"test", "both"}:
        out = make_test_predictions(k=args.k)
        print(f"Test predictions written to: {out}")


if __name__ == "__main__":
    main()
