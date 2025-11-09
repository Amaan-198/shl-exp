"""Typed containers shared across pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredCandidate:
    """Lightweight score triple representing a catalog candidate."""

    item_id: int
    fused_score: float
    rerank_score: float

