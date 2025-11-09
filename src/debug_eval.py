# src/debug_eval.py
import argparse
import pandas as pd
import re
from typing import List, Set
from . import config

def _norm_col(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")

_SLUG_RE = re.compile(r"/view/([^/?#]+)")

def _canon_slug(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    u = url.strip().lower()
    m = _SLUG_RE.search(u)
    slug = m.group(1) if m else u
    slug = slug.rstrip("/").replace("%28", "(").replace("%29", ")").replace("_", "-")
    return config.family_slug(slug)

def _recall_at_k(gold: Set[str], preds: List[str], k: int) -> float:
    if not gold:
        return 0.0
    top = preds[:k]
    return 1.0 if any(p in gold for p in top) else 0.0

def main(args):
    gold = pd.read_excel(args.train_csv)
    pred = pd.read_csv(args.preds_csv)

    gold.columns = [_norm_col(c) for c in gold.columns]
    pred.columns = [_norm_col(c) for c in pred.columns]

    gq = next(c for c in gold.columns if "query" in c)
    gu = next(c for c in gold.columns if "url" in c)
    pq = next(c for c in pred.columns if "query" in c)
    pu = next(c for c in pred.columns if "url" in c)

    gold["slug"] = gold[gu].map(_canon_slug)
    pred["slug"] = pred[pu].map(_canon_slug)

    # build gold: query -> set(slugs)
    gold_g = gold.groupby(gq)["slug"].apply(lambda s: set([x for x in s if x])).reset_index(name="gold_slugs")

    # build preds: query -> list(slugs) (de-dup by slug preserving order)
    def _dedup_keep_order(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    pred_g = (
        pred.groupby(pq)["slug"]
        .apply(lambda s: _dedup_keep_order([x for x in s if x]))
        .reset_index(name="pred_slugs")
    )

    merged = gold_g.merge(pred_g, left_on=gq, right_on=pq, how="left")
    merged["pred_slugs"] = merged["pred_slugs"].apply(lambda x: x if isinstance(x, list) else [])

    merged["r1"]  = merged.apply(lambda r: _recall_at_k(r["gold_slugs"], r["pred_slugs"], 1),  axis=1)
    merged["r5"]  = merged.apply(lambda r: _recall_at_k(r["gold_slugs"], r["pred_slugs"], 5),  axis=1)
    merged["r10"] = merged.apply(lambda r: _recall_at_k(r["gold_slugs"], r["pred_slugs"], 10), axis=1)

    n = len(merged)
    r1 = merged["r1"].mean(); r5 = merged["r5"].mean(); r10 = merged["r10"].mean()
    print(f"Total queries: {n}")
    print(f"Recall@1  = {r1:.4f}")
    print(f"Recall@5  = {r5:.4f}")
    print(f"Recall@10 = {r10:.4f}\n")

    if args.show_hits:
        print("Sample queries with gold vs predicted (canonical slug view):\n")
        for i, row in merged.head(args.show_hits).iterrows():
            q = row[gq]
            print(f"=== Query #{i+1}/{n} ===")
            print(f"Q: {q}")
            print("Gold (slugs):")
            for s in sorted(list(row["gold_slugs"])): print(f"  - .../view/{s}")
            print("Predicted (top 10, slugs):")
            for s in row["pred_slugs"][:10]: print(f"  - .../view/{s}")
            hits = [p for p in row["pred_slugs"][:10] if p in row["gold_slugs"]]
            if hits:
                print("Hits in top10 (by slug): " + ", ".join(f".../view/{h}" for h in hits))
            else:
                print("Hits in top10 (by slug): <none>")
            print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--preds_csv", required=True)
    ap.add_argument("--show_hits", type=int, default=10)
    main(ap.parse_args())
