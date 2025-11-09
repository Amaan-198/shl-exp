"""Shared query parsing helpers for duration, level, and intent tags."""

from __future__ import annotations

import re
from typing import Tuple, Set

import pandas as pd

from .config import APTITUDE_TRIGGER_PHRASES, BEHAVIOUR_TRIGGER_PHRASES, TECH_KEYWORDS

# ---------------------------------------------------------------------------
# Canonical keyword collections
# ---------------------------------------------------------------------------

_TECH_KEYWORDS = list(TECH_KEYWORDS)
_ENTRY_LEVEL_KEYWORDS = [
    "entry-level",
    "entry level",
    "graduate",
    "fresher",
    "campus",
    "intern",
    "internship",
    "0-2 years",
    "0-2 yrs",
    "0 to 2 years",
    "new graduates",
]
_TYPE_CATEGORY_MAP = {
    "Knowledge & Skills": "technical",
    "Ability & Aptitude": "aptitude",
    "Personality & Behavior": "behaviour",
    "Biodata & Situational Judgement": "behaviour",
    "Simulations": "behaviour",
    "Competencies": "behaviour",
    "Development & 360": "behaviour",
    "Assessment Exercises": "behaviour",
}
_INTENT_KEYWORDS = {
    "technical": _TECH_KEYWORDS,
    "behaviour": [
        "communication",
        "interpersonal",
        "presentation",
        "leadership",
        "teamwork",
        "collaboration",
        "stakeholder",
        "client",
        "customer",
        "soft skills",
        "relationship",
        "partner",
        "consultant",
        "empathy",
        "negotiation",
        "service",
        "orientation",
        "sales",
        "creative",
        "culture fit",
        "cultural fit",
        "values fit",
        "culturally a right fit",
    ],
    "aptitude": [
        "analytical",
        "reasoning",
        "logic",
        "logical",
        "numerical",
        "inductive",
        "aptitude",
        "problem solving",
        "quantitative",
        "cognitive",
        "iq",
    ],
}
_DURATION_RXES = [
    (re.compile(r"\b(\d+)\s*-\s*(\d+)\s*(min|mins|minutes)\b", re.I), 1),
    (re.compile(r"\b(\d+)\s*-\s*(\d+)\s*(hr|hrs|hour|hours)\b", re.I), 60),
    (
        re.compile(
            r"\b(?:at most|no more than|<=?)\s*(\d+)\s*(min|mins|minutes)\b", re.I
        ),
        "MAX_MIN",
    ),
    (
        re.compile(
            r"\b(?:at most|no more than|<=?)\s*(\d+)\s*(hr|hrs|hour|hours)\b", re.I
        ),
        "MAX_HR",
    ),
    (
        re.compile(
            r"\babout an hour\b|\baround an hour\b|\b~?1\s*(hr|hrs|hour|hours)\b", re.I
        ),
        "ABOUT_HR",
    ),
    (re.compile(r"\b(\d+)\s*(min|mins|minutes)\b", re.I), "SINGLE_MIN"),
    (re.compile(r"\b(\d+)\s*(hr|hrs|hour|hours)\b", re.I), "SINGLE_HR"),
]

# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


def _minutes_hint_from_query(q: str) -> Tuple[int | None, int | None]:
    """Return (max_minutes, approx_minutes) hints derived from the text query."""

    ql = q.lower()
    m = re.search(
        r"(?:at\s*most|<=?|no\s*more\s*than)\s*(\d{1,3})\s*(?:min|mins|minutes?)",
        ql,
    )
    if m:
        return int(m.group(1)), int(m.group(1))
    m = re.search(r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s*hour", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return b * 60, int((a + b) / 2 * 60)
    if re.search(r"(?:about|around|~)\s*an?\s*hour", ql) or re.search(
        r"\b1\s*hour\b|\b1\s*hr\b",
        ql,
    ):
        return 90, 60
    m = re.search(r"\b(\d{1,3})\s*(?:min|mins|minutes?)\b", ql)
    if m:
        v = int(m.group(1))
        return v, v
    return None, None


def _parse_duration_window(q: str) -> Tuple[int | None, int | None]:
    """Extract (min_minutes, max_minutes) window declarations from the query."""

    ql = q.lower()
    for rx, kind in _DURATION_RXES:
        m = rx.search(ql)
        if not m:
            continue
        if kind == 1:
            a, b = int(m.group(1)), int(m.group(2))
            return min(a, b), max(a, b)
        if kind == 60:
            a, b = int(m.group(1)) * 60, int(m.group(2)) * 60
            return min(a, b), max(a, b)
        if kind == "MAX_MIN":
            return None, int(m.group(1))
        if kind == "MAX_HR":
            return None, int(m.group(1)) * 60
        if kind == "ABOUT_HR":
            return 50, 70
        if kind == "SINGLE_MIN":
            return None, int(m.group(1))
        if kind == "SINGLE_HR":
            return None, int(m.group(1)) * 60
    return (None, None)


def _duration_adjust(score: float, duration_min: float, q: str) -> float:
    """Apply light duration alignment tweaks based on query hints."""

    max_minutes, approx = _minutes_hint_from_query(q)
    if duration_min <= 0:
        return score - 0.05
    if max_minutes is not None and duration_min > max_minutes + 15:
        return score - 0.08
    if approx is not None:
        if abs(duration_min - approx) <= 10:
            return score + 0.12
        if duration_min > approx + 30:
            return score - 0.05
    return score

# ---------------------------------------------------------------------------
# Query intent helpers
# ---------------------------------------------------------------------------


def _role_level(q: str) -> str:
    """Return a coarse role level bucket (exec, manager, grad, mid)."""

    ql = q.lower()
    if any(
        k in ql
        for k in [
            "cxo",
            "coo",
            "ceo",
            "cto",
            "chief ",
            "vp ",
            "vice president",
            "director",
            "head of",
            "senior leader",
            "senior leadership",
            "executive",
        ]
    ):
        return "exec"
    if any(k in ql for k in ["manager", "lead", "team lead", "senior manager"]):
        return "manager"
    if any(k in ql for k in _ENTRY_LEVEL_KEYWORDS):
        return "grad"
    return "mid"


def _categories_for_item(row: pd.Series) -> Set[str]:
    """Map a catalog row to behaviour/technical/aptitude categories."""

    cats: Set[str] = set()
    types = row.get("test_type", [])
    if isinstance(types, str):
        cleaned = types.replace("[", "").replace("]", "").replace("'", "")
        types_list = [t.strip() for t in cleaned.split(",") if t.strip()]
    elif isinstance(types, (list, tuple)):
        types_list = [str(t).strip() for t in types if str(t).strip()]
    else:
        try:
            types_list = [str(t).strip() for t in list(types) if str(t).strip()]
        except Exception:
            types_list = [str(types).strip()] if types else []
    for t in types_list:
        cat = _TYPE_CATEGORY_MAP.get(t)
        if cat:
            cats.add(cat)
    return cats


def _get_query_intent_categories(query: str) -> Set[str]:
    """Infer coarse intent categories from the query text."""

    q_lower = query.lower()
    cats: Set[str] = set()
    for cat, keywords in _INTENT_KEYWORDS.items():
        if any(k in q_lower for k in keywords):
            cats.add(cat)
    if any(kw in q_lower for kw in BEHAVIOUR_TRIGGER_PHRASES):
        cats.add("behaviour")
    if any(kw in q_lower for kw in APTITUDE_TRIGGER_PHRASES):
        cats.add("aptitude")
    return cats


__all__ = [
    "_TECH_KEYWORDS",
    "_ENTRY_LEVEL_KEYWORDS",
    "_minutes_hint_from_query",
    "_parse_duration_window",
    "_duration_adjust",
    "_role_level",
    "_categories_for_item",
    "_get_query_intent_categories",
]
