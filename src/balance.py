from __future__ import annotations
from math import ceil
from typing import Dict, List

"""
Simple allocation logic for balancing final recommendation sets.

After the reranking and diversification steps the recommender needs to
assemble a short list of assessment IDs that respects the user's
inferred intent (knowledge & skills versus personality & behavior).
This module encodes a handful of heuristics for splitting the list
into K/P buckets and then filling any remaining slots while filtering
out obviously off‑domain items.

The logic here closely follows the original project.  We expose a
single :func:`allocate` function that takes a ranked list of item IDs
with associated class labels and returns a final list of IDs sized
between ``RESULT_MIN`` and ``RESULT_MAX``.  Denied substrings can be
configured via the ``DENY_SUBSTR`` list.
"""

from typing import Dict, List
try:
    from loguru import logger  # type: ignore
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)

    class _FallbackLogger:
        def __init__(self, logger):
            self._logger = logger

        def info(self, msg: str, *args, **kwargs) -> None:
            self._logger.info(msg.format(*args))

        def warning(self, msg: str, *args, **kwargs) -> None:
            self._logger.warning(msg.format(*args))

        def error(self, msg: str, *args, **kwargs) -> None:
            self._logger.error(msg.format(*args))

        def exception(self, msg: str, *args, **kwargs) -> None:
            self._logger.exception(msg.format(*args))

    logger = _FallbackLogger(logging.getLogger(__name__))

from .config import RESULT_MIN, RESULT_MAX

# Very small denylist for SWE‑ish queries; keep conservative
DENY_SUBSTR = [
    "accounts payable",
    "accounts receivable",
    "pharmaceutical",
    "svar",
    "spoken english",
    "spoken french",
    "spoken spanish",
]


def _looks_offdomain(name: str, desc: str) -> bool:
    """Return True if a name/description appears to be off‑domain."""
    text = f"{name} {desc}".lower()
    return any(s in text for s in DENY_SUBSTR)


def allocate(ids: List[int], classes: Dict[int, str], target_size: int, pt: float, pb: float) -> List[int]:
    """
    ids: MMR-ordered list of item_ids
    classes: dict[item_id] -> "K" | "P" | "BOTH"
    target_size: final count to return (e.g., 10)
    pt, pb: intent proportions (technical vs behavior), 0..1
    """

    # decide split policy
    if pt >= 0.45 and pb >= 0.45:
        k_need, p_need = target_size // 2, target_size // 2
    elif max(pt, pb) >= 0.60 and min(pt, pb) >= 0.30:
        if pt >= pb:
            k_need = ceil(0.7 * target_size)
            p_need = target_size - k_need
        else:
            p_need = ceil(0.7 * target_size)
            k_need = target_size - p_need
    else:
        if pt >= pb:
            k_need, p_need = target_size, 0
        else:
            k_need, p_need = 0, target_size

    # split pools
    k_pool, p_pool, both_pool = [], [], []
    for i in ids:
        cls = classes.get(i, "K")
        if cls == "K":
            k_pool.append(i)
        elif cls == "P":
            p_pool.append(i)
        else:
            both_pool.append(i)

    picked: List[int] = []

    def take(pool: List[int], need: int):
        out = pool[:max(0, need)]
        rem = pool[len(out):]
        need_left = max(0, need - len(out))
        return out, rem, need_left

    # fill K (prefers K, then BOTH)
    out, k_pool, k_need = take(k_pool, k_need)
    picked += out
    if k_need > 0:
        out, both_pool, k_need = take(both_pool, k_need)
        picked += out

    # fill P (prefers P, then BOTH)
    out, p_pool, p_need = take(p_pool, p_need)
    picked += out
    if p_need > 0:
        out, both_pool, p_need = take(both_pool, p_need)
        picked += out

    # backfill to target_size using remaining MMR order
    seen = set(picked)
    for i in ids:
        if len(picked) >= target_size:
            break
        if i not in seen:
            picked.append(i)
            seen.add(i)

    return picked

