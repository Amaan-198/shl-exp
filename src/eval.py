# src/eval.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from . import config

# ---------- URL → canonical family slug ----------

_SLUG_RE = re.compile(r"/view/([^/?#]+)")

def _canon_slug(url: str) -> str:
    """
    Convert any SHL product URL into a canonical 'family' slug so
    variants like '-new', '(v2)', '-7.0' all match.
    """
    if not isinstance(url, str) or not url:
        return ""
    u = url.strip().lower()
    m = _SLUG_RE.search(u)
    slug = m.group(1) if m else u
    slug = slug.rstrip("/").replace("%28", "(").replace("%29", ")").replace("_", "-")
    return config.family_slug(slug)

# ---------- IO helpers ----------

def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8")
    cols = {c.lower(): c for c in df.columns}
    qcol, ucol = cols.get("query"), cols.get("assessment_url")
    if not qcol or not ucol:
        raise ValueError(
            f"Expected columns 'Query' and 'Assessment_url'. Found: {list(df.columns)}"
        )
    return df.rename(columns={qcol: "Query", ucol: "Assessment_url"})

def _normalize_query_key(q: str) -> str:
    """
    Normalize query text so the same JD with different whitespace/newlines
    maps to the same key.
    """
    q = str(q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q

# ---------- build gold sets (by family slug) ----------

def build_gold_sets(train_file: Path) -> Dict[str, Set[str]]:
    """
    Collapse identical logical queries into:
        normalized_query -> {canonical_family_slugs}
    """
    df = _read_any(train_file)
    gold: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        q_key = _normalize_query_key(row["Query"])
        slug = _canon_slug(str(row["Assessment_url"]))
        if q_key and slug:
            gold.setdefault(q_key, set()).add(slug)
    return gold

# ---------- metrics (compare slugs) ----------

def recall_at_k(gold: Set[str], pred_slugs: List[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(pred_slugs[:k])
    hits = len(gold.intersection(top))
    return hits / float(len(gold))

def evaluate(
    preds: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    ks=(1, 5, 10),
) -> Dict[int, float]:
    """
    preds: normalized query -> list of *URLs* in rank order
    We convert to family slugs here and de-dup while preserving order.
    """
    scores = {k: 0.0 for k in ks}
    n = 0
    for q_key, pred_urls in preds.items():
        if q_key not in gold:
            continue
        # URL -> slug; de-dup keeping order
        seen = set()
        pred_slugs: List[str] = []
        for u in pred_urls:
            s = _canon_slug(u)
            if s and s not in seen:
                seen.add(s)
                pred_slugs.append(s)

        g = gold[q_key]
        for k in ks:
            scores[k] += recall_at_k(g, pred_slugs, k)
        n += 1
    if n == 0:
        return {k: 0.0 for k in ks}
    return {k: scores[k] / n for k in ks}

# ---------- prediction writer (train/test CSVs) ----------

def write_test_predictions(preds: Dict[str, List[str]], path: Path) -> None:
    """
    preds: normalized_query_text -> [urls in rank order]
    Writes CSV with exact header: Query,Assessment_url
    (keeps incoming order; URLs are assumed already canonicalized upstream).
    """
    rows = []
    for q, urls in preds.items():
        for u in urls:
            if u:
                rows.append({"Query": q, "Assessment_url": u})
    df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=Path, required=True,
                    help="Path to training gold file (gen_ai_train.xlsx or CSV)")
    ap.add_argument("--preds_csv", type=Path, required=True,
                    help="Path to model predictions CSV")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()

    gold = build_gold_sets(args.train_csv)

    dfp = _read_any(args.preds_csv)
    # Group predictions by normalized query; keep URL order as-is
    preds: Dict[str, List[str]] = {}
    for q, frame in dfp.groupby("Query"):
        q_key = _normalize_query_key(q)
        urls = frame["Assessment_url"].astype(str).tolist()

        # de-dup *by slug* while preserving rank order
        seen = set()
        clean_urls: List[str] = []
        for u in urls:
            s = _canon_slug(u)
            if s and s not in seen:
                seen.add(s)
                clean_urls.append(u)
        preds[q_key] = clean_urls

    scores = evaluate(preds, gold, ks=args.k)
    for k in args.k:
        print(f"Recall@{k}: {scores[k]:.4f}")

if __name__ == "__main__":
    main()
