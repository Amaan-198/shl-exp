from __future__ import annotations
import re
import warnings
import os
import math
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote

"""
FastAPI application for the SHL recommender.

- Respects result policy: RESULT_MIN ≤ N ≤ RESULT_MAX
- Uses RESULT_DEFAULT_TARGET as a soft target only
- Flat-score aware dynamic cutoff (does NOT over-prune when reranker is offline)
- Family sibling completion can top up toward the soft target without exceeding MAX
- Canonical families pinned by name + slug, with must-include guard after MMR/cutoff
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    RESULT_MAX,
    RESULT_MIN,
    RESULT_DEFAULT_TARGET,
    INTENT_LABEL_TECHNICAL,
    INTENT_LABEL_PERSONALITY,
    ZERO_SHOT_MODEL,
    HealthResponse,
    BEHAVIOUR_TRIGGER_PHRASES,
    APTITUDE_TRIGGER_PHRASES,
    family_slug,
    RecommendResponse,
    MMR_LAMBDA,
    RETRIEVAL_BOOST_SEEDS,
    EXPANSION_LIBRARY,
    COMMUNICATION_TRIGGER_PHRASES,  # imported for future use
)

# optional transformers --------------------------------------------------------
try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None

# logging + noisy warnings guard ----------------------------------------------
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

        def warn(self, msg: str, *args, **kwargs) -> None:
            self._logger.warning(msg.format(*args))

        def error(self, msg: str, *args, **kwargs) -> None:
            self._logger.error(msg.format(*args))

        def exception(self, msg: str, *args, **kwargs) -> None:
            self._logger.exception(msg.format(*args))

    logger = _FallbackLogger(logging.getLogger(__name__))

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

# project modules -------------------------------------------------------------
from .catalog_build import load_catalog_snapshot
from .retrieval import retrieve_candidates
from .rerank import rerank_candidates
from .mmr import load_item_embeddings, mmr_select
from .balance import allocate
from .mapping import map_items_to_response
from .jd_fetch import fetch_and_extract

# =============================================================================
# Family / slug helpers
# =============================================================================


def _slug_from_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    u = url.strip().lower()
    m = re.search(r"/view/([^/?#]+)", u)
    return m.group(1) if m else u.rstrip("/")


def _family_base(slug: str) -> str:
    s = family_slug(slug)
    s = re.sub(
        r"-(essentials|advanced|advanced-level|entry-level|foundation|v\d+)$", "", s
    )
    return s


def _canonicalize_slug_for_eval(url: str) -> str:
    """
    Canonicalize the slug portion of a catalog URL so it matches train/eval gold:
    - Strip '-new', '(new)', '-v1', '-v2', ... suffixes
    - Decode URL-encoded parentheses
    - Collapse duplicate dashes
    """
    if not isinstance(url, str) or not url:
        return url
    m = re.search(r"(/view/)([^/?#]+)", url)
    if not m:
        return url
    slug = m.group(2)
    # Decode %28/%29 etc and normalise case
    slug_dec = unquote(slug).lower()

    # Drop version and "new" noise
    slug_dec = re.sub(r"-v\d+\b", "", slug_dec)
    slug_dec = slug_dec.replace("-new", "")
    slug_dec = slug_dec.replace("(new)", "")
    slug_dec = slug_dec.replace(" (new)", "")
    slug_dec = slug_dec.replace("%28new%29", "")

    # Normalise specific ssas patterns if they sneak through
    slug_dec = slug_dec.replace(
        "sql-server-analysis-services-%28ssas%29-%28new%29",
        "sql-server-analysis-services-(ssas)",
    )
    slug_dec = slug_dec.replace(
        "sql-server-analysis-services-%28ssas%29", "sql-server-analysis-services-(ssas)"
    )

    # Collapse duplicate dashes and trim
    slug_dec = re.sub(r"-+", "-", slug_dec).strip("-")

    # Rebuild URL with canonical slug
    new_url = url[: m.start(2)] + slug_dec + url[m.end(2) :]
    return new_url


def _family_expand_ids(
    catalog_df: pd.DataFrame, seed_ids: List[int], target: int
) -> List[int]:
    if not seed_ids or target <= 0:
        return seed_ids
    id_col = "item_id" if "item_id" in catalog_df.columns else "id"
    df = catalog_df.copy()
    df["slug"] = df["url"].map(_slug_from_url)
    df["fam"] = df["slug"].map(family_slug)
    df["base"] = df["slug"].map(_family_base)

    base_index: dict[str, List[int]] = {}
    for iid, base in df[[id_col, "base"]].itertuples(index=False, name=None):
        base_index.setdefault(str(base), []).append(int(iid))

    have: List[int] = list(seed_ids)
    seen = set(seed_ids)

    for iid in seed_ids:
        if len(have) >= target:
            break
        try:
            base = str(df.loc[df[id_col] == iid, "base"].values[0])
        except Exception:
            continue
        for sib in base_index.get(base, []):
            if len(have) >= target:
                break
            if sib not in seen:
                have.append(sib)
                seen.add(sib)

    return have[:target]


def _find_ids_by_slugs(
    catalog_df: pd.DataFrame, slugs: list[str], limit: int = 3
) -> list[int]:
    """Return item_ids whose URL slug (family-canonicalized) matches any of the given slugs."""
    if not slugs:
        return []
    want = {family_slug(s) for s in slugs if s}
    id_col = "item_id" if "item_id" in catalog_df.columns else "id"
    df = catalog_df.copy()
    df["slug"] = df["url"].map(_slug_from_url).map(family_slug)
    hits: list[int] = []
    for iid, s in df[[id_col, "slug"]].itertuples(index=False, name=None):
        if s in want:
            hits.append(int(iid))
            if len(hits) >= limit:
                break
    return hits


# =============================================================================
# Exec / culture helper (stricter)
# =============================================================================


def _is_exec_culture_query(q: str) -> bool:
    """
    Return True only when the query is genuinely about senior/executive leadership
    and/or cultural fit at that level. Avoid firing just because of 'customer support executives'.
    """
    ql = q.lower()
    qpad = f" {ql} "

    # Hard C-suite / senior markers
    if any(
        k in ql
        for k in [
            "coo",
            "chief operating officer",
            "c-suite",
            "cxo",
            "chief executive officer",
            "ceo",
            "cfo",
            "cto",
            "vp ",
            "vice president",
            "senior leadership",
            "executive leadership",
        ]
    ):
        return True

    # Generic "executive" but only when paired with culture/leadership language
    if " executive " in qpad:
        if any(
            k in ql
            for k in [
                "culture fit",
                "cultural fit",
                "values fit",
                "leadership",
                "senior leader",
                "people leader",
                "right fit for our culture",
                "executive role",
            ]
        ):
            return True

    return False


def _collect_must_include_ids(query: str, catalog_df: pd.DataFrame) -> list[int]:
    """
    Build a small 'must include' set of canonical families per intent (looked up by slug).
    Also includes *backstops* for explicit cognitive/personality asks so category balance
    works even if retrieval misses those families.
    """
    q = query.lower()

    # Use stricter exec / culture detection
    is_exec_culture = _is_exec_culture_query(query)
    is_java = ("java" in q) and any(k in q for k in ["developer", "engineer"])
    wants_40 = ("40" in q and "min" in q) or "40 minutes" in q
    is_qa = any(
        k in q
        for k in [
            "qa",
            "quality assurance",
            "selenium",
            "manual testing",
            "webdriver",
            "test case",
        ]
    )
    is_admin = any(
        k in q
        for k in [
            "assistant admin",
            "administrative assistant",
            "bank admin",
            "bank administrative",
        ]
    )
    is_sales_grad = ("sales" in q) and any(
        k in q for k in ["entry level", "entry-level", "graduate", "fresher", "0-2"]
    )
    # STRICTER: don't let 'community' trigger marketing manager by itself
    is_marketing_mgr = ("marketing manager" in q) or (
        "marketing" in q
        and any(
            k in q for k in ["brand", "campaign", "demand generation", "seo", "content"]
        )
    )
    is_writer_seo = any(
        k in q for k in ["content writer", "content writing", "copywriter", "seo"]
    )
    is_data_analyst = ("data analyst" in q) or any(
        k in q for k in [" sql ", " excel ", " tableau ", " power bi ", " analytics "]
    )
    is_consultant_io = ("consultant" in q) and any(
        k in q
        for k in [
            "industrial/organizational",
            "industrial organizational",
            "i/o",
            "psychometric",
            "talent assessment",
            "validation",
            "job analysis",
        ]
    )
    is_product_manager = ("product manager" in q) or (
        " product " in q and " manager " in q
    )
    is_customer_support = any(
        k in q
        for k in [
            "customer support",
            "customer service",
            "call center",
            "contact center",
        ]
    )
    is_finance_analyst = "finance" in q and "analyst" in q
    is_ai_research = any(
        k in q
        for k in [
            " ai ",
            " artificial intelligence",
            "machine learning",
            " ml ",
            "research engineer",
            " llm ",
            " rag ",
        ]
    )

    must_slug_packs: list[list[str]] = []

    if is_exec_culture:
        must_slug_packs += [
            ["occupational-personality-questionnaire-opq32r"],
            ["enterprise-leadership-report", "enterprise-leadership-report-2-0"],
            ["global-skills-assessment"],
        ]
    if is_java:
        must_slug_packs += [
            ["core-java-entry-level", "java-8"],
            ["core-java-advanced-level"],
            ["interpersonal-communications"],
        ]
        if wants_40:
            must_slug_packs += [["programming-concepts"], ["automata-fix"]]
    if is_qa:
        must_slug_packs += [
            ["selenium"],
            ["manual-testing"],
            ["htmlcss", "css3", "javascript"],
            ["automata-sql", "sql-server"],
        ]
    if is_admin:
        must_slug_packs += [
            ["administrative-professional-short-form"],
            ["bank-administrative-assistant-short-form"],
            ["general-entry-level-data-entry-7-0-solution"],
            ["verify-numerical-ability"],
            ["basic-computer-literacy-windows-10"],
        ]
    if is_sales_grad:
        must_slug_packs += [
            [
                "entry-level-sales-7-1",
                "entry-level-sales-sift-out-7-1",
                "entry-level-sales-solution",
            ],
            ["sales-representative-solution", "technical-sales-associate-solution"],
            ["interpersonal-communications"],
            ["business-communication-adaptive"],
        ]
    if is_marketing_mgr:
        must_slug_packs += [
            ["digital-advertising"],
            ["marketing"],
            ["writex-email-writing-sales"],
            ["manager-8-0-jfa-4310", "manager-8-0"],
            ["business-communication-adaptive"],
        ]
    if is_writer_seo:
        must_slug_packs += [
            ["search-engine-optimization"],
            ["written-english"],
            ["english-comprehension"],
            ["drupal"],
        ]
    if is_data_analyst:
        must_slug_packs += [
            ["automata-sql", "sql-server"],
            ["microsoft-excel-365"],
            ["microsoft-excel-365-essentials"],
            ["python"],
            ["tableau"],
            ["sql-server-analysis-services-ssas"],
            ["data-warehousing-concepts"],
        ]
    if is_consultant_io:
        must_slug_packs += [
            ["occupational-personality-questionnaire-opq32r"],
            ["verify-verbal-ability-next-generation", "verify-verbal-ability"],
            ["verify-numerical-ability"],
            ["professional-7-1-solution", "professional-7-0-solution-3958"],
        ]
    if is_product_manager:
        must_slug_packs += [
            ["agile"],
            ["scrum"],
            ["jira"],
            ["confluence"],
            ["business-communication-adaptive"],
        ]
    if is_customer_support:
        must_slug_packs += [
            ["svar-spoken-english-indian-accent"],
            ["business-communication-adaptive"],
            ["interpersonal-communications"],
            ["customer-service-simulation"],  # will no-op if not found
        ]
    if is_finance_analyst:
        must_slug_packs += [
            ["microsoft-excel-365"],
            ["microsoft-excel-365-essentials"],
            ["verify-numerical-ability"],
            ["verify-verbal-ability-next-generation"],
            ["automata-sql"],
            ["tableau"],
        ]
    if is_ai_research:
        must_slug_packs += [
            ["python"],
            ["programming-concepts"],
            ["machine-learning"],
        ]

    # --- Backstops if the query explicitly asks "cognitive/personality"
    if any(
        k in q
        for k in [
            "cognitive",
            "aptitude",
            "reasoning",
            "verbal ability",
            "numerical ability",
            "inductive",
            "iq",
        ]
    ):
        must_slug_packs += [
            ["verify-verbal-ability-next-generation", "verify-verbal-ability"],
            ["verify-numerical-ability"],
            ["inductive-reasoning", "shl-verify-interactive-inductive-reasoning"],
        ]
    if any(
        k in q
        for k in [
            "personality",
            "culture fit",
            "values fit",
            "behavioral",
            "behavioural",
        ]
    ):
        must_slug_packs += [["occupational-personality-questionnaire-opq32r"]]

    ids: list[int] = []
    seen: set[int] = set()
    for pack in must_slug_packs:
        for iid in _find_ids_by_slugs(catalog_df, pack, limit=2):
            if iid not in seen:
                ids.append(iid)
                seen.add(iid)
    return ids


def _force_pin_presence(
    final_ids: list[int],
    ranked: list["ScoredCandidate"],
    must_ids: list[int],
    max_k: int,
) -> list[int]:
    """Ensure 'must_ids' are present; if missing, inject them near the front and trim to max_k."""
    if not must_ids:
        return final_ids
    out = list(final_ids)
    have = set(out)
    rank_pos = {c.item_id: i for i, c in enumerate(ranked)}
    to_add = [mid for mid in must_ids if mid not in have]
    to_add.sort(key=lambda i: rank_pos.get(i, 10_000))
    if to_add:
        out = to_add + out
    seen = set()
    dedup: list[int] = []
    for iid in out:
        if iid in seen:
            continue
        dedup.append(iid)
        seen.add(iid)
    if len(dedup) > max_k:
        dedup = dedup[:max_k]
    return dedup


# =============================================================================
# Seed injection + duration helpers
# =============================================================================


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _minutes_hint_from_query(q: str) -> tuple[int | None, int | None]:
    """
    Returns (max_minutes, approx_minutes) where:
      - max_minutes: hard upper bound if 'at most/<=/<=90' style is found
      - approx_minutes: single-point hint (e.g., 'about an hour' -> 60), or mid-point for ranges.
    """
    ql = q.lower()
    m = re.search(
        r"(?:at\s*most|<=?|no\s*more\s*than)\s*(\d{1,3})\s*(?:min|mins|minutes?)", ql
    )
    if m:
        return int(m.group(1)), int(m.group(1))
    m = re.search(r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s*hour", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return b * 60, int((a + b) / 2 * 60)
    if re.search(r"(?:about|around|~)\s*an?\s*hour", ql) or re.search(
        r"\b1\s*hour\b|\b1\s*hr\b", ql
    ):
        return 90, 60
    m = re.search(r"\b(\d{1,3})\s*(?:min|mins|minutes?)\b", ql)
    if m:
        v = int(m.group(1))
        return v, v
    return None, None


def _duration_adjust(score: float, duration_min: float, q: str) -> float:
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


def _intent_keys_from_query(query: str) -> list[str]:
    """
    Map the raw query into small intent keys that drive seed injection.
    (Tightened marketing trigger; added explicit keys for aptitude/behavior,
    entry-level sales, product manager, customer support, finance analyst, AI research.)
    """

    def _match_any(ql: str, keys: list[str]) -> bool:
        return any(k in ql for k in keys)

    ql = query.lower()
    keys: list[str] = []

    # Existing buckets
    if _match_any(
        ql,
        [
            "consultant",
            "i/o",
            "industrial/organizational",
            "psychometric",
            "people science",
        ],
    ):
        keys += ["consultant", "industrial organizational", "behavior", "aptitude"]
    if _match_any(
        ql, ["qa engineer", "qa ", "quality assurance", "testing", "selenium"]
    ):
        keys += ["qa engineer", "quality assurance", "qa_testing"]
    if _match_any(
        ql, ["assistant admin", "administrative assistant", "bank admin", "data entry"]
    ):
        keys += ["admin_ops"]
    if _match_any(
        ql, ["content writer", "content writing", "copywriter", "seo", "email writing"]
    ):
        keys += ["content_marketing"]
    # STRICTER marketing manager (don't key on "community" alone)
    if ("marketing manager" in ql) or (
        "marketing" in ql
        and any(
            k in ql
            for k in ["brand", "campaign", "demand generation", "seo", "content"]
        )
    ):
        keys += ["marketing manager", "marketing_mgr", "marketing"]
    if _match_any(
        ql, ["entry level sales", "sales associate", "spoken english", "svar"]
    ) or (
        "sales" in ql
        and any(
            k in ql
            for k in ["entry level", "entry-level", "graduate", "fresher", "0-2"]
        )
    ):
        keys += ["sales_entry"]
    if (
        "analyst" in ql
        or _match_any(
            ql,
            [
                "analytics",
                "business analyst",
                "business intelligence",
                "tableau",
                "power bi",
                "excel ",
            ],
        )
    ):
        keys += ["data_analyst"]
    if _match_any(ql, ["java developer", "java "]):
        keys += ["java_dev"]
    if _match_any(
        ql,
        [
            "coo",
            "chief operating officer",
            "culture fit",
            "culturally",
            "values fit",
            "personality",
            "behavioral",
            "behavioural",
        ],
    ):
        keys += ["behavior"]

    # NEW keys / triggers
    if any(
        k in ql
        for k in [
            "cognitive",
            "aptitude",
            "reasoning",
            "verbal ability",
            "numerical ability",
            "inductive",
            "iq",
        ]
    ):
        keys += ["aptitude"]
    if any(
        k in ql
        for k in [
            "personality",
            "culture fit",
            "values fit",
            "behavioral",
            "behavioural",
        ]
    ):
        keys += ["behavior"]
    if ("product manager" in ql) or (" product " in ql and " manager " in ql):
        keys += ["product_manager"]
    if any(
        k in ql
        for k in [
            "customer support",
            "customer service",
            "call center",
            "contact center",
        ]
    ):
        keys += ["customer_support"]
    if "finance" in ql and "analyst" in ql:
        keys += ["finance_analyst"]
    if any(
        k in ql
        for k in [
            " ai ",
            " artificial intelligence",
            "machine learning",
            " ml ",
            "research engineer",
            " llm",
            " rag ",
        ]
    ):
        keys += ["ai_research_eng", "data_analyst"]  # allow technical pull too

    keys = list(dict.fromkeys(keys))
    return _limit_intent_keys(keys, query)


def _limit_intent_keys(keys: list[str], query: str) -> list[str]:
    """
    For very long JDs, keep only the top 1–2 strongest archetype keys to avoid over-broad seed injection.
    For shorter queries, allow up to 3 archetypes so multi-skill roles are covered.
    """
    if not keys:
        return keys
    ql = query.lower()

    archetype_groups: dict[str, list[str]] = {
        "java_dev": ["java", "developer", "engineer"],
        "qa engineer": [
            "qa engineer",
            "qa ",
            "quality assurance",
            "selenium",
            "testing",
        ],
        "quality assurance": ["quality assurance", "qa ", "testing"],
        "qa_testing": ["qa", "testing", "selenium"],
        "data_analyst": ["data analyst", "analytics", "sql", "excel", "tableau", "bi"],
        "marketing manager": [
            "marketing manager",
            "brand",
            "campaign",
            "demand generation",
            "seo",
            "content",
        ],
        "marketing_mgr": [
            "marketing",
            "brand",
            "campaign",
            "demand generation",
            "seo",
            "content",
        ],
        "marketing": [
            "marketing",
            "brand",
            "campaign",
            "demand generation",
            "seo",
            "content",
        ],
        "content_marketing": [
            "content writer",
            "content writing",
            "copywriter",
            "seo",
            "email writing",
        ],
        "admin_ops": [
            "assistant admin",
            "administrative assistant",
            "bank admin",
            "data entry",
        ],
        "sales_entry": [
            "entry level sales",
            "sales role",
            "sales associate",
            "spoken english",
            "svar",
        ],
        "consultant": [
            "consultant",
            "industrial",
            "psychometric",
            "talent assessment",
            "job analysis",
        ],
        "industrial organizational": [
            "industrial/organizational",
            "industrial organizational",
            "i/o",
        ],
        "behavior": [
            "culture fit",
            "cultural fit",
            "values fit",
            "leadership",
            "personality",
            "behavior",
        ],
        "aptitude": [
            "numerical",
            "verbal",
            "inductive",
            "reasoning",
            "aptitude",
            "cognitive",
            "iq",
        ],
        "product_manager": [
            "product manager",
            "jira",
            "confluence",
            "sdlc",
            "agile",
            "scrum",
        ],
        "customer_support": [
            "customer support",
            "customer service",
            "contact center",
            "call center",
            "spoken english",
        ],
        "finance_analyst": [
            "finance",
            "analyst",
            "excel",
            "kpi",
            "forecast",
            "budget",
            "numerical",
        ],
        "ai_research_eng": [
            "ai",
            "artificial intelligence",
            "machine learning",
            "ml",
            "llm",
            "rag",
            "research engineer",
        ],
    }

    scores: dict[str, int] = {}
    for k, toks in archetype_groups.items():
        scores[k] = sum(1 for tok in toks if tok in ql)

    dedup = list(dict.fromkeys(keys))
    if len(dedup) <= 2:
        return dedup

    length = len(ql)
    if length >= 1600:
        max_keys = 2
    elif length >= 800:
        max_keys = 3
    else:
        max_keys = 3

    dedup.sort(key=lambda k: scores.get(k, 0), reverse=True)
    return dedup[:max_keys]


@dataclass
class ScoredCandidate:
    item_id: int
    fused_score: float
    rerank_score: float


def _inject_seed_candidates(
    query: str, ranked: List[ScoredCandidate], catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Inject catalog items that match intent buckets from RETRIEVAL_BOOST_SEEDS / EXPANSION_LIBRARY."""
    if not ranked:
        base = 0.6
    else:
        top = float(ranked[0].rerank_score)
        base = top - 0.02 if math.isfinite(top) else 0.95

    wanted_phrases: list[str] = []
    for key in _intent_keys_from_query(query):
        wanted_phrases += RETRIEVAL_BOOST_SEEDS.get(key, [])
        wanted_phrases += EXPANSION_LIBRARY.get(key, [])
    wanted_norm = {_normalize(w) for w in wanted_phrases if w}

    have = {c.item_id for c in ranked}
    for iid, row in catalog_df.iterrows():
        name = _normalize(str(row.get("name", "")))
        desc = _normalize(str(row.get("description", "")))
        blob = f"{name} {desc}"
        if any(p in blob for p in wanted_norm):
            if iid not in have:
                ranked.append(
                    ScoredCandidate(
                        item_id=int(iid), fused_score=base, rerank_score=base
                    )
                )
                have.add(int(iid))

    ranked.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return ranked


# =============================================================================
# Canonical "pinned" families + helpers
# =============================================================================


def _lookup_by_name_keywords(
    catalog_df: pd.DataFrame, name_keywords: list[str], limit: int = 10
) -> list[int]:
    """Return item_ids whose (name+desc) contains all tokens of any keyword query (case-insensitive)."""

    hits: list[int] = []
    if not name_keywords:
        return hits

    def _tokens_in(blob: str, phrase: str) -> bool:
        toks = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if t]
        return all(t in blob for t in toks)

    for iid, row in catalog_df.iterrows():
        blob = f"{row.get('name','')} {row.get('description','')}".lower()
        if any(_tokens_in(blob, k) for k in name_keywords):
            hits.append(int(iid))
            if len(hits) >= limit:
                break
    return hits


def _pinned_names_for_query(q: str) -> list[list[str]]:
    """Return an ordered list of keyword-sets; each set maps to a canonical family to pin."""
    ql = q.lower()

    is_exec_culture = _is_exec_culture_query(q)
    is_java = ("java" in ql) and any(k in ql for k in ["developer", "engineer"])
    wants_40 = ("40" in ql and "min" in ql) or "40 minutes" in ql
    is_qa = any(
        k in ql
        for k in [
            "qa",
            "quality assurance",
            "selenium",
            "manual testing",
            "test case",
            "webdriver",
        ]
    )
    is_admin = any(
        k in ql for k in ["assistant admin", "administrative assistant", "bank admin"]
    )
    wants_30_40 = any(k in ql for k in ["30-40", "30 – 40", "30 to 40"]) or (
        "30" in ql and "40" in ql and "min" in ql
    )
    is_sales_grad = ("sales" in ql) and any(
        k in ql for k in ["entry level", "entry-level", "graduate", "fresher", "0-2"]
    )
    # STRICTER: avoid "community" alone
    is_marketing_mgr = ("marketing manager" in ql) or (
        "marketing" in ql
        and any(
            k in ql
            for k in ["brand", "campaign", "demand generation", "seo", "content"]
        )
    )
    is_writer_seo = any(
        k in ql for k in ["content writer", "content writing", "copywriter", "seo"]
    )
    is_data_analyst = ("data analyst" in ql) or any(
        k in ql for k in ["sql", "excel", "tableau", "python"]
    )

    pinned: list[list[str]] = []

    if is_exec_culture:
        pinned += [
            ["occupational personality questionnaire", "opq32r", "opq"],
            [
                "enterprise leadership report",
                "mfs 360 enterprise leadership",
                "leadership report",
            ],
            ["global skills assessment"],
        ]
    if is_java:
        pinned += [
            ["core java entry level", "java 8"],
            ["core java advanced level"],
            ["interpersonal communications"],
        ]
        if wants_40:
            pinned += [["programming concepts"]]
    if is_qa:
        pinned += [
            ["automata sql", "automata-sql", "sql server"],
            ["selenium"],
            ["manual testing"],
            ["htmlcss", "css3", "javascript"],
        ]
    if is_admin:
        pinned += [
            ["administrative professional short form"],
            ["bank administrative assistant short form"],
            ["general entry level data entry 7.0"],
            ["verify numerical ability"],
            ["basic computer literacy windows 10"],
        ]
        if wants_30_40:
            pinned += [["multitasking ability"]]
    if is_sales_grad:
        pinned += [
            ["entry level sales", "entry-level sales sift-out 7.1"],
            ["sales representative solution", "technical sales associate"],
            ["interpersonal communications"],
            ["svar spoken english indian accent"],
            ["business communication adaptive"],
        ]
    if is_marketing_mgr:
        pinned += [
            ["digital advertising"],
            ["marketing"],
            ["writex email writing sales"],
            ["manager 8.0"],
            ["english comprehension", "business communication adaptive"],
        ]
    if is_writer_seo:
        pinned += [
            ["search engine optimization"],
            ["written english"],
            ["english comprehension"],
            ["business communication adaptive"],
            ["drupal"],
        ]
    if is_data_analyst:
        pinned += [
            ["automata sql", "sql server"],
            ["microsoft excel 365"],
            ["microsoft excel 365 essentials"],
            ["python"],
            ["tableau"],
            ["sql server analysis services (ssas)"],
            ["data warehousing concepts"],
        ]

    return pinned


def _prepend_pinned_candidates(
    query: str, ranked: list[ScoredCandidate], catalog_df: pd.DataFrame
) -> list[ScoredCandidate]:
    """
    Pin canonical families at the top with a strong score.
    - Pins by NAME keywords (ordered) and by canonical SLUGs (must-include families)
    - Deterministic; avoids duplicates; lets MMR diversify afterwards.
    """
    sets = _pinned_names_for_query(query)
    have = {c.item_id for c in ranked}
    prepend: list[ScoredCandidate] = []

    for name_set in sets:
        ids = _lookup_by_name_keywords(catalog_df, name_set, limit=2)
        for iid in ids:
            if iid in have:
                continue
            prepend.append(
                ScoredCandidate(item_id=int(iid), fused_score=1.25, rerank_score=1.25)
            )
            have.add(int(iid))

    must_ids = _collect_must_include_ids(query, catalog_df)
    for iid in must_ids:
        if iid not in have:
            prepend.append(
                ScoredCandidate(item_id=int(iid), fused_score=1.26, rerank_score=1.26)
            )
            have.add(int(iid))

    if not prepend:
        return ranked

    merged = prepend + ranked
    merged.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return merged


def _hard_duration_filter(
    query: str, ranked: list[ScoredCandidate], catalog_df: pd.DataFrame
) -> list[ScoredCandidate]:
    max_minutes, _ = _minutes_hint_from_query(query)
    if max_minutes is None:
        return ranked
    kept: list[ScoredCandidate] = []
    for c in ranked:
        try:
            dur = float(catalog_df.loc[c.item_id, "duration"] or 0)
        except Exception:
            dur = 0.0
        if dur and dur > (max_minutes + 12):
            continue
        kept.append(c)
    return kept or ranked


# =============================================================================
# Heuristics (boosted + expanded)
# =============================================================================


def _normalize_basename(name: str) -> str:
    base = name.lower()
    base = re.sub(r"[\s&/\-]+", "", base)
    base = re.sub(r"[^a-z0-9]", "", base)
    return base


_TECH_KEYWORDS = [
    "software",
    "developer",
    "developers",
    "programmer",
    "coder",
    "engineer",
    "engineers",
    "engineering",
    "technician",
    "technology",
    "technical",
    "coding",
    "programming",
    "devops",
    "backend",
    "front-end",
    "frontend",
    "fullstack",
    "full-stack",
    "python",
    "java",
    ".net",
    "c#",
    "c++",
    "javascript",
    "machine learning",
    "ml",
    "neural network",
    "neural_network",
    "deep learning",
    "data engineer",
    "data-engineer",
    "sql",
    "excel",
    "tableau",
    "power bi",
    "power-bi",
    "bi",
    "data warehouse",
    "data warehousing",
]
_TECH_ALLOWED_TYPES = {"Knowledge & Skills", "Ability & Aptitude"}
_GENERIC_PATTERNS = [
    "multitasking ability",
    "360",
    "verify",
    "inductive reasoning",
    "360 feedback",
]
_NON_EN_LANGUAGES = [
    "spanish",
    "french",
    "german",
    "mandarin",
    "chinese",
    "arabic",
    "hindi",
    "japanese",
    "portuguese",
    "italian",
    "sv",
    "svenska",
]
_CLIENT_ALLOWED_TYPES = {
    "Personality & Behavior",
    "Biodata & Situational Judgement",
    "Knowledge & Skills",
}
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
_ENTRY_LEVEL_POSITIVE = [
    "verify g+",
    "inductive",
    "numerical",
    "multitasking",
    "entry-level",
    "entry level",
    "graduate",
    "entry-level sales",
    "entry level sales",
    "sales representative",
    "sales-representative",
    "sales associate",
    "technical sales",
]
_ENTRY_LEVEL_NEGATIVE = [
    "expert",
    "senior",
    "advanced",
    "salesforce",
    "sap",
    "dynamics",
]
_DOMAIN_KEYWORDS = [
    "food",
    "beverage",
    "hospitality",
    "accounting",
    "retail",
    "filing",
    "front office",
    "office management",
    "restaurants",
    "hotel",
    "pharmaceutical",
    "insurance",
    "sales",
    "marketing",
    "customer service",
    "support",
    "filling",
    "warehouse",
    "hipaa",
    "healthcare",
    "medical",
    "medical records",
]
_AI_KEYWORDS = [
    "artificial intelligence",
    "ai",
    "machine learning",
    "ml",
    "deep learning",
    "data science",
    "neural network",
    "computer vision",
    "nlp",
    "natural language",
]
_PYTHON_KEYWORDS = [
    "python",
    "django",
    "flask",
    "pandas",
    "numpy",
    "data structures",
    "data analysis",
    "tensorflow",
    "pytorch",
    "machine learning",
]
_ANALYTICS_KEYWORDS = [
    "excel",
    "tableau",
    "power bi",
    "visualization",
    "visualisation",
    "data viz",
    "reporting",
    "storytelling",
    "analytics",
    "data analytics",
    "business intelligence",
    "ssas",
    "sql server",
    "automata sql",
]
_DOMAIN_FOCUS_KEYWORDS = {
    "analytics": [
        "analytics",
        "data analysis",
        "business data",
        "analyze",
        "analyse",
        "data-driven",
        "reporting",
        "insight",
        "data insights",
        "data interpretation",
        "dashboard",
        "tableau",
        "power bi",
        "excel",
    ],
    "communication": [
        "communication",
        "writing",
        "presentation",
        "interpersonal",
        "client communication",
        "collaboration",
        "stakeholder management",
        "storytelling",
        "english",
        "verbal",
    ],
    "sales": [
        "sales",
        "negotiation",
        "customer",
        "service orientation",
        "customer service",
        "client-facing",
        "selling",
        "retail",
        "marketing",
    ],
}
_COMMON_IRRELEVANT_PATTERNS = [
    "filing - names",
    "filing - numbers",
    "food science",
    "food and beverage",
    "front office management",
    "following instructions",
    "written english",
    "filling",
    "office management",
    "office operations",
    "housekeeping",
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


def _parse_duration_window(q: str) -> tuple[int | None, int | None]:
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


def _role_level(q: str) -> str:
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


# -----------------------------------------------------------------------------
# Category helpers / balance / filters
# -----------------------------------------------------------------------------


def _categories_for_item(row) -> set[str]:
    cats: set[str] = set()
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


def _get_query_intent_categories(query: str) -> set[str]:
    q_lower = query.lower()
    cats: set[str] = set()
    for cat, keywords in _INTENT_KEYWORDS.items():
        if any(k in q_lower for k in keywords):
            cats.add(cat)
    if any(kw in q_lower for kw in BEHAVIOUR_TRIGGER_PHRASES):
        cats.add("behaviour")
    if any(kw in q_lower for kw in APTITUDE_TRIGGER_PHRASES):
        cats.add("aptitude")
    return cats


def _apply_category_balance(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    needed_cats = _get_query_intent_categories(query)
    if not needed_cats or len(needed_cats) == 1:
        return ranked
    present: set[str] = set()
    for c in ranked[:RESULT_MAX]:
        try:
            row = catalog_df.loc[c.item_id]
        except Exception:
            continue
        present |= _categories_for_item(row)
    missing = needed_cats - present
    if not missing:
        return ranked
    to_promote: List[ScoredCandidate] = []
    for cat in missing:
        for c in ranked:
            try:
                row = catalog_df.loc[c.item_id]
            except Exception:
                continue
            if cat in _categories_for_item(row):
                to_promote.append(c)
                break
    new_ranked: List[ScoredCandidate] = []
    promoted_ids = {c.item_id for c in to_promote}
    for c in ranked:
        if c.item_id in promoted_ids and c not in new_ranked:
            new_ranked.append(c)
    for c in ranked:
        if c not in new_ranked:
            new_ranked.append(c)
    return new_ranked


def _apply_category_filter(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    query_cats = _get_query_intent_categories(query)
    filtered: List[ScoredCandidate] = list(ranked)
    if "behaviour" in query_cats and "technical" not in query_cats:
        tmp: List[ScoredCandidate] = []
        for c in filtered:
            try:
                row = catalog_df.loc[c.item_id]
            except Exception:
                tmp.append(c)
                continue
            cats = _categories_for_item(row)
            if cats == {"technical"}:
                continue
            tmp.append(c)
        filtered = tmp
    if "technical" in query_cats and "behaviour" not in query_cats:
        tmp2: List[ScoredCandidate] = []
        for c in filtered:
            try:
                row = catalog_df.loc[c.item_id]
            except Exception:
                tmp2.append(c)
                continue
            cats = _categories_for_item(row)
            if cats == {"behaviour"}:
                continue
            tmp2.append(c)
        filtered = tmp2
    if len(filtered) < max(5, len(ranked) // 3):
        return ranked
    return filtered


# -----------------------------------------------------------------------------
# Rank shaping helpers (generic penalty, post adjustments, domain drops)
# -----------------------------------------------------------------------------


def _apply_generic_penalty(
    ranked: List[ScoredCandidate], catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    penalised: List[ScoredCandidate] = []
    for c in ranked:
        item_id = int(c.item_id)
        fused_score = float(getattr(c, "fused_score", getattr(c, "rerank_score", 0.0)))
        score = float(getattr(c, "rerank_score", fused_score))
        try:
            name = str(catalog_df.loc[item_id, "name"]).lower()
        except Exception:
            name = ""
        if any(pat in name for pat in _GENERIC_PATTERNS):
            score *= 0.7
        penalised.append(
            ScoredCandidate(
                item_id=item_id, fused_score=fused_score, rerank_score=score
            )
        )
    penalised.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return penalised


def _post_rank_adjustments(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    q_lower = query.lower()

    if "collaborat" in q_lower or "business team" in q_lower:
        query_cats = _get_query_intent_categories(query) | {"behaviour"}
    else:
        query_cats = _get_query_intent_categories(query)

    role = _role_level(query)
    dur_min, dur_max = _parse_duration_window(query)

    is_entry = role == "grad"
    is_client = any(
        k in q_lower
        for k in [
            "client",
            "customer",
            "stakeholder",
            "presentation",
            "communication",
            "teamwork",
            "collaboration",
        ]
    )
    is_strong_tech = (
        "technical" in query_cats and "behaviour" not in query_cats
    ) or any(k in q_lower for k in _TECH_KEYWORDS)

    is_content_writer = any(
        k in q_lower
        for k in [
            "content writer",
            "content-writing",
            "content writing",
            "copywriter",
            "copy writer",
            "blog writer",
            "seo",
            "search engine optimization",
        ]
    )
    wants_english = any(
        k in q_lower
        for k in [
            "english",
            "spoken english",
            "written english",
            "english test",
            "english comprehension",
            "business communication",
            "communication skills",
            "verbal ability",
            "verbal test",
        ]
    )
    is_exec = role == "exec"
    cares_culture = any(
        k in q_lower
        for k in [
            "culture fit",
            "cultural fit",
            "values fit",
            "right fit for our culture",
        ]
    )

    is_qa_query = any(
        k in q_lower
        for k in [
            "qa engineer",
            "qa",
            "quality assurance",
            "software testing",
            "tester",
            "manual testing",
            "selenium",
            "webdriver",
            "test case",
            "test plan",
            "regression test",
        ]
    )
    is_sales_grad = ("sales" in q_lower) and role == "grad"
    is_analyst = any(
        k in q_lower
        for k in [
            "data analyst",
            "analyst",
            "analytics",
            "sql",
            "excel",
            "tableau",
            "bi",
            "business intelligence",
        ]
    )
    is_fin_ops_analyst = "finance" in q_lower and "analyst" in q_lower

    is_consultant_io = ("consultant" in q_lower) and any(
        k in q_lower
        for k in [
            "industrial/organizational",
            "industrial organizational",
            "i/o",
            "psychometric",
            "talent assessment",
            "validation",
            "job analysis",
        ]
    )
    is_marketing_mgr = ("marketing manager" in q_lower) or (
        "marketing" in q_lower
        and any(
            k in q_lower
            for k in [
                "brand",
                "campaign",
                "demand generation",
                "events",
                "seo",
                "content",
            ]
        )
    )
    is_product_manager = ("product manager" in q_lower) or (
        " product " in q_lower and " manager" in q_lower
    ) or ("product management" in q_lower)
    is_presales = "presales" in q_lower or "pre-sales" in q_lower
    is_non_tech_role = (
        is_sales_grad
        or is_marketing_mgr
        or is_consultant_io
        or is_presales
        or any(
            k in q_lower
            for k in ["admin", "administrative assistant", "bank administrative"]
        )
    )
    pm_allows_dev = is_product_manager and any(
        k in q_lower for k in ["coding", "developer", "programming", "hands-on", "technical"]
    )
    js_in_query = any(
        term in q_lower
        for term in ["javascript", "java script", " js ", " nodejs", " front end", "frontend"]
    )
    wants_stack_python_sql_js = "python" in q_lower and "sql" in q_lower and js_in_query
    has_analytics_terms = any(
        k in q_lower
        for k in ["tableau", "power bi", "excel", "analytics", "business intelligence", "bi "]
    )
    stack_focus_no_analytics = wants_stack_python_sql_js and not has_analytics_terms
    presales_tool_signal = is_presales and any(
        k in q_lower
        for k in [
            "canva",
            "adobe",
            "synthesia",
            "presentation",
            "presentations",
            "proposal",
            "proposals",
            "demo",
            "demos",
            "pitch",
            "rfp",
            "storytelling",
        ]
    )
    presales_allows_dev = is_presales and any(
        k in q_lower
        for k in ["developer", "engineering", "coding", "technical", "solution engineer"]
    )

    DEV_NOISE = {
        "java",
        "framework",
        "programming",
        "developer",
        "c++",
        "linux",
        "spring",
        "hibernate",
        "salesforce development",
    }

    # Screening vs development/360
    is_screening = any(
        k in q_lower
        for k in [
            "screen",
            "screening",
            "screen applications",
            "applications to screen",
            "shortlist",
            "short-list",
            "short listing",
            "filter candidates",
            "hiring",
            "recruit",
            "recruitment",
        ]
    )
    is_cog_plus_personality = (
        "cognitive" in q_lower or "aptitude" in q_lower or "reasoning" in q_lower
    ) and ("personality" in q_lower or "behaviour" in q_lower or "behavior" in q_lower)

    # Customer support / call-center
    is_contact_centre = any(
        k in q_lower
        for k in [
            "customer support",
            "customer service",
            "contact center",
            "contact centre",
            "call center",
            "call centre",
            "bpo",
            "voice process",
        ]
    )

    adjusted: List[ScoredCandidate] = []
    seen_bases: dict[str, bool] = {}

    for c in ranked:
        iid = c.item_id
        try:
            row = catalog_df.loc[iid]
        except Exception:
            row = {}

        score = float(c.rerank_score)

        name = str(row.get("name", "") or "")
        desc = str(row.get("description", "") or "")
        name_desc = (name + " " + desc).lower()
        lname = name.lower()

        try:
            duration = float(row.get("duration", 0) or 0)
        except Exception:
            duration = 0.0
        if duration == 0:
            score -= 0.10
        if dur_max is not None:
            if duration > (dur_max + 5):
                score -= 0.20
            elif duration >= (dur_max - 5):
                score -= 0.05
            else:
                score += 0.03
        if dur_min is not None and duration > 0 and duration < (dur_min - 5):
            score -= 0.05
        score = _duration_adjust(score, duration, query)

        # Explicit demotion of long 360/behavioral when tight budgets (≤45m)
        if dur_max is not None and dur_max <= 45:
            if (
                any(
                    kw in name_desc
                    for kw in ["360", "enterprise leadership report", "manager 8.0"]
                )
                and duration
                and duration > dur_max
            ):
                score -= 0.25

        if (
            any(word in lname for word in ["report", "guide", "profile"])
            and "opq" not in name_desc
            and "leadership" not in name_desc
        ):
            score -= 0.10

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

        if any(k in q_lower for k in _TECH_KEYWORDS) and any(
            t in _TECH_ALLOWED_TYPES for t in types_list
        ):
            score += 0.08
        if is_client and any(
            t
            in {
                "Personality & Behavior",
                "Biodata & Situational Judgement",
                "Knowledge & Skills",
            }
            for t in types_list
        ):
            score += 0.08

        if not any(lang in q_lower for lang in _NON_EN_LANGUAGES):
            if any(lang in name_desc for lang in _NON_EN_LANGUAGES):
                score -= 0.08

        if is_entry or is_sales_grad:
            if any(pat in lname for pat in _ENTRY_LEVEL_POSITIVE):
                score += 0.08
            if any(pat in lname for pat in _ENTRY_LEVEL_NEGATIVE):
                score -= 0.04

        if name:
            for kw in _DOMAIN_KEYWORDS:
                if kw in lname and kw not in q_lower:
                    score -= 0.05
                    break

        base = _normalize_basename(name)
        if base:
            if base in seen_bases:
                score -= 0.05
            else:
                seen_bases[base] = True

        ai_query = any(kw in q_lower for kw in _AI_KEYWORDS)
        if ai_query and any(kw in name_desc for kw in _AI_KEYWORDS):
            score += 0.08
        elif ai_query:
            score -= 0.05

        if any(kw in q_lower for kw in _PYTHON_KEYWORDS) and any(
            kw in name_desc for kw in _PYTHON_KEYWORDS
        ):
            score += 0.12

        if any(kw in q_lower for kw in _ANALYTICS_KEYWORDS) and any(
            kw in name_desc for kw in _ANALYTICS_KEYWORDS
        ):
            score += 0.12

        if stack_focus_no_analytics:
            if any(
                kw in name_desc
                for kw in [
                    "javascript",
                    "java script",
                    "nodejs",
                    "node.js",
                    "front end",
                    "frontend",
                    "web developer",
                    "web development",
                ]
            ):
                score += 0.12
            if any(
                kw in name_desc
                for kw in [
                    "microsoft excel",
                    "excel 365",
                    "tableau",
                    "power bi",
                    "business intelligence",
                    "ssas",
                    "data warehouse",
                    "data warehousing",
                ]
            ):
                score -= 0.12
            if "python" in name_desc or "programming concepts" in name_desc:
                score += 0.06
            if "sql" in name_desc:
                score += 0.06

        query_domains: set[str] = set()
        for dom, kws in _DOMAIN_FOCUS_KEYWORDS.items():
            if any(k in q_lower for k in kws):
                query_domains.add(dom)
        if query_domains:
            matches_domain = any(
                any(kw in name_desc for kw in _DOMAIN_FOCUS_KEYWORDS.get(dom, []))
                for dom in query_domains
            )
            score += 0.06 if matches_domain else -0.05

        if not any(pat in q_lower for pat in _COMMON_IRRELEVANT_PATTERNS):
            for pat in _COMMON_IRRELEVANT_PATTERNS:
                if pat in lname:
                    score -= 0.25 if is_strong_tech else 0.07
                    break

        if is_content_writer or wants_english:
            if any(
                kw in name_desc
                for kw in [
                    "english comprehension",
                    "written english",
                    "business communication adaptive",
                    "svar spoken english",
                    "writex email writing",
                ]
            ):
                score += 0.14
            if "search engine optimization" in name_desc or "seo" in name_desc:
                score += 0.10

        if is_exec or cares_culture:
            if any(
                kw in name_desc
                for kw in [
                    "occupational personality questionnaire",
                    "opq",
                    "enterprise leadership report",
                    "manager 8.0",
                    "global skills assessment",
                    "team types and leadership styles",
                ]
            ):
                score += 0.18
            if any(
                kw in name_desc
                for kw in [
                    "contact center",
                    "call simulation",
                    "count out the money",
                    "conversational multichat simulation",
                    "cashier",
                    "retail sales",
                    "warehouse",
                    "data entry",
                ]
            ):
                score -= 0.30
            if "technical checking" in name_desc:
                score -= 0.35
            if (
                "behaviour" in _categories_for_item(row)
                and "leadership" not in name_desc
                and "opq" not in name_desc
            ):
                score -= 0.03

        # Java dev: prefer Core/Java8 & interpersonal; demote EE/Linux and automata-front-end
        if "java" in q_lower and any(k in q_lower for k in ["developer", "engineer"]):
            if "core java" in name_desc or "java 8" in name_desc:
                score += 0.14
            if (
                "java platform enterprise edition" in name_desc
                or "java ee" in name_desc
                or "linux" in name_desc
            ):
                score -= 0.12
            if "automata front-end" in name_desc or "front end" in name_desc:
                score -= 0.12
            if (
                "interpersonal communications" in name_desc
                or "business communication" in name_desc
            ):
                score += 0.08

        if is_qa_query:
            if any(
                kw in name_desc
                for kw in [
                    "selenium",
                    "manual testing",
                    "htmlcss",
                    "javascript",
                    "css3",
                    "automata sql",
                    "sql server",
                ]
            ):
                score += 0.18
            if any(
                kw in name_desc
                for kw in ["automata front-end", "front end", "front-end"]
            ):
                score -= 0.10
            if "verify" in name_desc and not any(
                kw in name_desc
                for kw in [
                    "selenium",
                    "manual testing",
                    "qa",
                    "test case",
                    "software testing",
                ]
            ):
                score -= 0.06

        if is_sales_grad and any(
            kw in name_desc
            for kw in [
                "entry-level sales",
                "entry level sales",
                "sales representative",
                "sales-representative",
                "technical sales associate",
                "business communication adaptive",
                "svar spoken english",
            ]
        ):
            score += 0.16
        if (
            is_sales_grad
            and not is_contact_centre
            and "svar spoken english" in name_desc
        ):
            score -= 0.10

        if any(
            k in q_lower
            for k in [
                "admin",
                "administrative assistant",
                "assistant admin",
                "bank administrative",
            ]
        ):
            if any(
                kw in name_desc
                for kw in [
                    "administrative professional short form",
                    "bank administrative assistant short form",
                    "basic computer literacy",
                    "verify numerical ability",
                    "general entry level data entry",
                ]
            ):
                score += 0.18
            if "multitasking ability" in name_desc:
                score += 0.06

        if is_analyst:
            if any(
                kw in name_desc
                for kw in [
                    "automata sql",
                    "sql server",
                    "sql-server",
                    "microsoft excel 365",
                    "excel 365 essentials",
                    "tableau",
                    "python",
                    "data warehousing",
                    "ssas",
                ]
            ):
                score += 0.14
            if "appdynamics" in name_desc or "cisco" in name_desc:
                score -= 0.10

        # Finance & Ops Analyst: push down heavy ETL/SSAS when not explicitly asked
        if is_fin_ops_analyst and any(
            k in name_desc for k in ["data warehousing", "ssas", "etl", "spark"]
        ):
            score -= 0.18

        if is_consultant_io:
            if any(
                kw in name_desc
                for kw in [
                    "occupational personality questionnaire",
                    "opq",
                    "verify verbal ability",
                    "verify numerical ability",
                    "inductive reasoning",
                    "professional 7.1",
                    "professional 7.0",
                ]
            ):
                score += 0.20
            if any(
                kw in name_desc
                for kw in [
                    "data warehousing",
                    "ssas",
                    "python",
                    "sql server programming",
                    "tableau",
                    "excel 365",
                    "data visualization",
                ]
            ):
                score -= 0.22

        if is_product_manager:
            if any(
                kw in name_desc
                for kw in [
                    "agile",
                    "scrum",
                    "product management",
                    "product manager",
                    "project management",
                    "stakeholder",
                    "requirements",
                    "user story",
                    "roadmap",
                    "jira",
                    "confluence",
                    "business communication",
                    "time management",
                ]
            ):
                score += 0.14
            if not pm_allows_dev and any(
                kw in name_desc
                for kw in [
                    "automata",
                    "developer",
                    "programming",
                    "java",
                    "python",
                    "c++",
                    "c#",
                    "linux",
                    "spark",
                    "hadoop",
                    "spring",
                    "hibernate",
                ]
            ):
                score -= 0.14

        if is_marketing_mgr:
            if any(
                kw in name_desc
                for kw in [
                    "digital advertising",
                    "writex email writing sales",
                    "manager 8.0",
                    "business communication adaptive",
                ]
            ):
                score += 0.18
            # Strongly demote heavy analytics/ETL items for marketing leadership roles
            if any(
                kw in name_desc
                for kw in [
                    "data warehousing",
                    "data warehouse",
                    "ssas",
                    "sql server",
                    "automata sql",
                    "etl",
                    "tableau",
                    "business intelligence",
                    "analytics",
                ]
            ):
                score -= 0.18
            if any(
                kw in name_desc
                for kw in [
                    "salesforce development",
                    "java",
                    "frameworks",
                    "programming",
                    "linux",
                ]
            ):
                score -= 0.18

        if is_non_tech_role and any(kw in name_desc for kw in DEV_NOISE):
            score -= 0.15

        # Screening vs development/360: avoid 360/HiPo when explicitly screening applications
        if is_screening:
            if any(
                kw in name_desc
                for kw in [
                    "360 ",
                    "360°",
                    "360 feedback",
                    "mfs 360",
                    "enterprise leadership report",
                    "hipo assessment",
                    "high potential",
                    "development report",
                    "coaching report",
                ]
            ):
                score -= 0.22

        if is_cog_plus_personality:
            if any(
                kw in name_desc
                for kw in [
                    "occupational personality questionnaire",
                    "opq",
                    "opq32r",
                ]
            ):
                score += 0.18
            if any(
                kw in name_desc
                for kw in [
                    "verify verbal ability",
                    "verify numerical ability",
                    "interactive inductive reasoning",
                    "verify g+",
                ]
            ):
                score += 0.16

        # Presales: emphasise comms + solutioning; downweight ETL/warehouse + contact-centre sims
        if is_presales:
            if any(
                kw in name_desc
                for kw in [
                    "business communication adaptive",
                    "english comprehension",
                    "written english",
                    "writex email writing",
                    "interpersonal communications",
                    "presentation",
                    "presentation skills",
                    "proposal",
                    "customer presentation",
                ]
            ):
                score += 0.14
            if presales_tool_signal and any(
                kw in name_desc
                for kw in ["presentation", "proposal", "demo", "pitch", "storytelling"]
            ):
                score += 0.10
            if (not has_analytics_terms) and any(
                kw in name_desc
                for kw in [
                    "data warehousing",
                    "data warehouse",
                    "sql server",
                    "automata sql",
                    "ssas",
                    "etl",
                    "tableau",
                    "business intelligence",
                    "analytics",
                ]
            ):
                score -= 0.16
            if (not presales_allows_dev) and any(
                kw in name_desc
                for kw in [
                    "automata",
                    "developer",
                    "programming",
                    "java",
                    "python",
                    "linux",
                    "spark",
                    "data engineer",
                ]
            ):
                score -= 0.14
            if any(
                kw in name_desc
                for kw in [
                    "contact center",
                    "contact centre",
                    "call center",
                    "call centre",
                    "multichat simulation",
                    "call simulation",
                ]
            ):
                score -= 0.08

        # Customer support / call centre: favour English/communication; demote leadership/DB
        if is_contact_centre:
            if any(
                kw in name_desc
                for kw in [
                    "business communication adaptive",
                    "interpersonal communications",
                    "svar spoken english",
                    "spoken english",
                    "writex email writing",
                    "english comprehension",
                    "written english",
                ]
            ):
                score += 0.16
            if any(
                kw in name_desc
                for kw in [
                    "enterprise leadership report",
                    "mfs 360",
                    "360 feedback",
                    "leadership report",
                    "hipo assessment",
                    "high potential",
                ]
            ):
                score -= 0.22
            if any(
                kw in name_desc
                for kw in [
                    "sql server programming",
                    "data warehousing",
                    "data warehouse",
                    "ssas",
                    "etl",
                ]
            ):
                score -= 0.18

        adjusted.append(
            ScoredCandidate(
                item_id=c.item_id, fused_score=c.fused_score, rerank_score=score
            )
        )

    adjusted.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return adjusted


def _hard_drop_if_strong_tech(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    q_lower = query.lower()
    tech_hit = any(k in q_lower for k in _TECH_KEYWORDS)
    ai_hit = any(k in q_lower for k in _AI_KEYWORDS)
    analytics_hit = any(k in q_lower for k in _ANALYTICS_KEYWORDS)
    strong = (tech_hit or ai_hit or analytics_hit) and not any(
        k in q_lower for k in _INTENT_KEYWORDS.get("behaviour", [])
    )
    if not strong:
        return ranked
    hard_drop_patterns = [
        "following instructions",
        "reviewing forms",
        "filing - names",
        "filing - numbers",
        "written english",
        "written spanish",
        "ms office basic computer literacy",
    ]
    out: List[ScoredCandidate] = []
    for c in ranked:
        try:
            name = str(catalog_df.loc[c.item_id, "name"]).lower()
        except Exception:
            out.append(c)
            continue
        if any(pat in name for pat in hard_drop_patterns):
            continue
        out.append(c)
    return out or ranked


def _filter_domain_candidates(
    query: str, ranked: List[ScoredCandidate], catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    q_lower = query.lower()
    if not any(k in q_lower for k in _TECH_KEYWORDS):
        return ranked
    filtered: List[ScoredCandidate] = []
    for c in ranked:
        try:
            types = catalog_df.loc[c.item_id, "test_type"]
        except Exception:
            types = []
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
        if any(t in _TECH_ALLOWED_TYPES for t in types_list):
            filtered.append(c)
    return filtered or ranked


# -----------------------------------------------------------------------------
# Domain vetoes (off-domain cleanup)
# -----------------------------------------------------------------------------


def _apply_domain_vetoes(
    query: str, ranked_list: List[ScoredCandidate], catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    ql = query.lower()
    is_consultant_io = ("consultant" in ql) and any(
        k in ql
        for k in [
            "industrial/organizational",
            "industrial organizational",
            "i/o",
            "psychometric",
            "talent assessment",
            "validation",
            "job analysis",
        ]
    )
    is_marketing_mgr = ("marketing manager" in ql) or (
        ("marketing" in ql) and ("brand" in ql)
    )
    is_sales_grad = ("sales" in ql) and any(
        k in ql for k in ["entry level", "entry-level", "graduate", "fresher", "0-2"]
    )
    is_presales = "presales" in ql or "pre-sales" in ql
    is_admin = any(
        k in ql
        for k in [
            "administrative assistant",
            "assistant admin",
            "bank admin",
            "bank administrative",
        ]
    )
    is_qa = any(
        k in ql
        for k in ["qa", "quality assurance", "selenium", "manual testing", "webdriver"]
    )
    is_fin_ops_analyst = "finance" in ql and "analyst" in ql
    is_product_manager = ("product manager" in ql) or (
        " product " in ql and " manager" in ql
    ) or ("product management" in ql)
    pm_allows_dev = is_product_manager and any(
        k in ql for k in ["coding", "developer", "programming", "hands-on", "technical"]
    )
    presales_allows_dev = is_presales and any(
        k in ql for k in ["developer", "engineering", "coding", "technical", "solution engineer"]
    )
    presales_mentions_analytics = any(
        k in ql for k in ["tableau", "power bi", "excel", "analytics", "business intelligence", "bi "]
    )

    non_tech = (
        is_consultant_io or is_marketing_mgr or is_sales_grad or is_admin or is_presales
    )
    dev_noise = {
        "java",
        "framework",
        "programming",
        "developer",
        "c++",
        "linux",
        "spring",
        "hibernate",
        "salesforce development",
        "automata",
    }

    cleaned: List[ScoredCandidate] = []
    for c in ranked_list:
        row = catalog_df.loc[c.item_id] if c.item_id in catalog_df.index else {}
        blob = f"{row.get('name','')} {row.get('description','')} {row.get('search_text','')}".lower()
        score = float(c.rerank_score)

        if non_tech and any(k in blob for k in dev_noise):
            score -= 0.20

        if is_consultant_io and any(
            k in blob
            for k in [
                "data warehousing",
                "ssas",
                "tableau",
                "python",
                "sql server programming",
                "spark",
            ]
        ):
            score -= 0.25

        if is_qa and ("automata front-end" in blob or "front end" in blob):
            score -= 0.12

        if is_marketing_mgr and any(
            k in blob
            for k in [
                "data warehousing",
                "data warehouse",
                "sql server",
                "automata sql",
                "ssas",
                "etl",
                "tableau",
                "business intelligence",
                "analytics",
            ]
        ):
            score -= 0.18

        if is_fin_ops_analyst and any(
            k in blob for k in ["data warehousing", "ssas", "etl", "spark"]
        ):
            score -= 0.18

        if is_presales and any(
            k in blob
            for k in [
                "business communication",
                "written english",
                "writex",
                "interpersonal communications",
                "presentation",
                "proposal",
                "demo",
                "pitch",
            ]
        ):
            score += 0.12
        if is_presales and (not presales_mentions_analytics) and any(
            k in blob
            for k in [
                "tableau",
                "microsoft excel",
                "power bi",
                "ssas",
                "data warehouse",
                "data warehousing",
            ]
        ):
            score -= 0.18
        if is_presales and (not presales_allows_dev) and any(
            k in blob
            for k in [
                "automata",
                "developer",
                "programming",
                "java",
                "python",
                "spark",
                "data engineer",
                "sql server programming",
            ]
        ):
            score -= 0.18

        if is_product_manager and any(
            k in blob
            for k in [
                "agile",
                "scrum",
                "product management",
                "product manager",
                "project management",
                "stakeholder",
                "requirements",
                "user story",
                "roadmap",
                "jira",
                "confluence",
                "business communication",
            ]
        ):
            score += 0.12
        if is_product_manager and (not pm_allows_dev) and any(
            k in blob
            for k in [
                "automata",
                "developer",
                "programming",
                "java",
                "python",
                "c++",
                "c#",
                "linux",
                "spark",
                "hadoop",
                "spring",
                "hibernate",
            ]
        ):
            score -= 0.18

        cleaned.append(
            ScoredCandidate(
                item_id=c.item_id, fused_score=c.fused_score, rerank_score=score
            )
        )

    cleaned.sort(key=lambda x: (-float(x.rerank_score), x.item_id))
    return cleaned or ranked_list


# -----------------------------------------------------------------------------
# Dynamic cutoff / diversity
# -----------------------------------------------------------------------------


def _apply_dynamic_cutoff(
    final_ids: List[int], ranked_scores: dict[int, float], soft_target: int, query: str
) -> List[int]:
    """
    Keep between RESULT_MIN..RESULT_MAX. Use a sharper knee detector and duration hints.
    - Short budget queries (~≤60m) tend to benefit from 5–7 focused results.
    - Flat score distributions fall back to soft_target, otherwise use knee/quantiles.
    """
    scores = [ranked_scores.get(i, 0.0) for i in final_ids]
    if not scores:
        return final_ids

    mn, mx = min(scores), max(scores)
    rng = mx - mn
    if rng <= 1e-6:
        return final_ids[:soft_target]

    norm = [(s - mn) / (rng if rng > 0 else 1.0) for s in scores]
    drops = [norm[i - 1] - norm[i] for i in range(1, len(norm))]
    knee_idx = None
    best_drop = 0.0
    for i, d in enumerate(drops, 1):
        if d > best_drop and d >= 0.18:
            best_drop = d
            knee_idx = i

    tail_avg = sum(norm[-3:]) / min(3, len(norm))
    tail_weak = tail_avg < 0.35

    keep = len(final_ids)
    if knee_idx:
        keep = knee_idx + 1
    if tail_weak:
        keep = min(keep, max(soft_target - 2, RESULT_MIN))

    max_minutes, approx = _minutes_hint_from_query(query)
    if (approx or 0) <= 60 or (max_minutes or 0) <= 60:
        keep = min(keep, 7)

    keep = max(RESULT_MIN, min(keep, RESULT_MAX))
    return final_ids[:keep]


def _ensure_min_category_diversity(
    final_ids: List[int],
    ranked: List[ScoredCandidate],
    catalog_df: pd.DataFrame,
    min_categories: int = 2,
) -> List[int]:
    present: set[str] = set()
    for iid in final_ids:
        try:
            row = catalog_df.loc[iid]
        except Exception:
            continue
        present |= _categories_for_item(row)
    if len(present) >= min_categories:
        return final_ids
    for c in ranked:
        if len(present) >= min_categories:
            break
        if c.item_id in final_ids:
            continue
        try:
            row = catalog_df.loc[c.item_id]
        except Exception:
            continue
        cats = _categories_for_item(row)
        new_cats = cats - present
        if new_cats:
            final_ids.append(c.item_id)
            present |= new_cats
    if len(final_ids) > RESULT_MAX:
        final_ids = final_ids[:RESULT_MAX]
    return final_ids


# =============================================================================
# Pipeline
# =============================================================================


def run_full_pipeline(
    query: str,
    catalog_df: pd.DataFrame,
    intent_model: object | None = None,
) -> RecommendResponse:

    # 0) URL fetch
    if re.match(r"^https?://", query.strip(), re.IGNORECASE):
        logger.info("Query is a URL. Fetching content from: {}", query)
        fetched_text = fetch_and_extract(query)
        if not fetched_text:
            logger.warning("Failed to fetch or extract text from URL.")
            return RecommendResponse(recommended_assessments=[])
        query = fetched_text

    if not query or not query.strip():
        return RecommendResponse(recommended_assessments=[])

    # 1) Retrieval
    raw_query, cleaned_query, fused = retrieve_candidates(query)

    # 2) Rerank
    ranked = rerank_candidates(cleaned_query, fused)

    # 3) Heuristics BEFORE MMR
    if (
        (len(cleaned_query) > 320)
        or ("job description" in cleaned_query.lower())
        or ("responsibilit" in cleaned_query.lower())
    ):
        ranked = _filter_domain_candidates(cleaned_query, ranked, catalog_df)
    else:
        if not any(
            k in cleaned_query.lower()
            for k in ["leadership", "communication", "conflict", "teamwork", "empathy"]
        ):
            ranked = _filter_domain_candidates(cleaned_query, ranked, catalog_df)

    ranked = _apply_generic_penalty(ranked, catalog_df)
    ranked = _post_rank_adjustments(ranked, cleaned_query, catalog_df)
    ranked = _hard_drop_if_strong_tech(ranked, cleaned_query, catalog_df)

    # Canonical family pinning (name + slug)
    ranked = _prepend_pinned_candidates(cleaned_query, ranked, catalog_df)

    # Track must-include IDs for guard after MMR/cutoff
    must_ids = _collect_must_include_ids(cleaned_query, catalog_df)

    # Seed injection
    ranked = _inject_seed_candidates(cleaned_query, ranked, catalog_df)
    logger.info(
        "Seed injection active for keys: {}", _intent_keys_from_query(cleaned_query)
    )

    # Duration guard
    ranked = _hard_duration_filter(cleaned_query, ranked, catalog_df)

    # Off-domain cleanup
    ranked = _apply_domain_vetoes(cleaned_query, ranked, catalog_df)

    # Category balance / filter
    ranked = _apply_category_balance(ranked, cleaned_query, catalog_df)
    ranked = _apply_category_filter(ranked, cleaned_query, catalog_df)

    # 4) MMR
    embeddings, ids = load_item_embeddings()
    mmr_ids = mmr_select(
        candidates=[(c.item_id, c.rerank_score) for c in ranked],
        embeddings=embeddings,
        ids=ids,
        k=RESULT_MAX,
        lambda_=MMR_LAMBDA,
    )

    # 5) Intent
    if intent_model is not None:
        try:
            intent_labels = [INTENT_LABEL_TECHNICAL, INTENT_LABEL_PERSONALITY]
            intent_result = intent_model(
                cleaned_query, intent_labels, multi_label=False
            )
            score_map = {
                label: score
                for label, score in zip(
                    intent_result["labels"], intent_result["scores"]
                )
            }
            pt = float(score_map.get(INTENT_LABEL_TECHNICAL, 0.5))
            pb = float(score_map.get(INTENT_LABEL_PERSONALITY, 0.5))
            _q = f" {cleaned_query.lower()} "
            if any(
                k in _q
                for k in [
                    " leadership ",
                    " employee_engagement ",
                    " conflict_management ",
                    " interpersonal ",
                ]
            ):
                pb = min(0.80, pb + 0.20)
            if any(
                k in _q
                for k in [
                    " python ",
                    " backend ",
                    " data_structures ",
                    " machine_learning ",
                    " neural_network ",
                    " excel ",
                    " tableau ",
                    " power_bi ",
                    " visualization ",
                    " analytics ",
                    " sql ",
                ]
            ):
                pt = min(0.80, pt + 0.15)
            s = pt + pb
            if s > 0:
                pt, pb = pt / s, pb / s
            logger.info("Intent scores: technical={:.2f}, behavior={:.2f}", pt, pb)
        except Exception as e:
            logger.warning("Intent classification failed: {}", e)
            pt = pb = 0.5
    else:
        pt = pb = 0.5

    # 6) K/P/BOTH mapping
    def _kp_class(types_list: list[str]) -> str:
        K_SET = {
            "Knowledge & Skills",
            "Ability & Aptitude",
            "Simulations",
            "Assessment Exercises",
        }
        P_SET = {
            "Personality & Behavior",
            "Competencies",
            "Development & 360",
            "Biodata & Situational Judgement",
        }
        has_k = any(t in K_SET for t in types_list)
        has_p = any(t in P_SET for t in types_list)
        if has_k and has_p:
            return "BOTH"
        if has_p:
            return "P"
        return "K"

    classes: dict[int, str] = {}
    for iid in mmr_ids:
        row = catalog_df.loc[iid]
        raw_types = row.get("test_type")
        if raw_types is None or (
            isinstance(raw_types, float) and raw_types != raw_types
        ):
            types_list = []
        elif isinstance(raw_types, str):
            stripped = raw_types.replace("[", "").replace("]", "").replace("'", "")
            types_list = [t.strip() for t in stripped.split(",") if t.strip()]
        elif isinstance(raw_types, (list, tuple)):
            types_list = [str(t).strip() for t in raw_types if str(t).strip()]
        else:
            try:
                types_list = [str(t).strip() for t in list(raw_types) if t.strip()]
            except Exception:
                types_list = [str(raw_types).strip()] if raw_types else []
        classes[iid] = _kp_class(types_list)

    # 7) Allocation
    try:
        final_ids = allocate(
            mmr_ids, classes, RESULT_MAX, pt=pt, pb=pb, catalog_df=catalog_df
        )
    except TypeError:
        final_ids = allocate(mmr_ids, classes, RESULT_MAX, pt=pt, pb=pb)

    # 8) Dynamic cutoff to soft target
    soft_target = min(max(RESULT_MIN, RESULT_DEFAULT_TARGET), RESULT_MAX)
    score_lookup = {c.item_id: c.rerank_score for c in ranked}
    final_ids = _apply_dynamic_cutoff(
        final_ids, score_lookup, soft_target=soft_target, query=cleaned_query
    )

    # Ensure category diversity
    final_ids = _ensure_min_category_diversity(
        final_ids, ranked, catalog_df, min_categories=2
    )

    # Family top-up towards soft target
    if len(final_ids) < soft_target:
        final_ids = _family_expand_ids(
            catalog_df, final_ids, min(soft_target, RESULT_MAX)
        )

    # Ensure minimum results
    if len(final_ids) < RESULT_MIN:
        seen = set(final_ids)
        for c in ranked:
            if len(final_ids) >= RESULT_MIN:
                break
            if c.item_id not in seen:
                final_ids.append(c.item_id)
                seen.add(c.item_id)

    # Must-include guard (OPQ, digital advertising, core-java, etc.)
    final_ids = _force_pin_presence(
        final_ids,
        ranked,
        _collect_must_include_ids(cleaned_query, catalog_df),
        max_k=RESULT_MAX,
    )

    # Final clamp
    if len(final_ids) > RESULT_MAX:
        final_ids = final_ids[:RESULT_MAX]

    # 9) Map to API schema
    response = map_items_to_response(final_ids, catalog_df)
    if len(response.recommended_assessments) > RESULT_MAX:
        response.recommended_assessments = response.recommended_assessments[:RESULT_MAX]

    # Canonicalise URLs for eval (alias collapsing)
    for item in response.recommended_assessments:
        try:
            if getattr(item, "url", None):
                item.url = _canonicalize_slug_for_eval(item.url)
        except Exception:
            continue

    return response


# =============================================================================
# FastAPI app + startup
# =============================================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_catalog_df = None  # type: ignore
intent_classifier = None


@app.on_event("startup")
def startup_event() -> None:
    global _catalog_df, intent_classifier
    logger.info("Starting app warmup...")
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer  # type: ignore  # noqa: F401
        except Exception:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
            logger.warning("Disabled hf_transfer acceleration (package not installed).")
    _catalog_df = load_catalog_snapshot().set_index("item_id", drop=False)
    logger.info("Loaded catalog snapshot with {} rows", len(_catalog_df))
    try:
        from .retrieval import _load_retrieval_components

        _load_retrieval_components()
        from .mmr import load_item_embeddings as _load_emb

        _load_emb()
    except Exception as e:
        logger.warning("Warmup partial failure: {}", e)
    if pipeline is not None:
        try:
            intent_classifier = pipeline(
                "zero-shot-classification", model=ZERO_SHOT_MODEL
            )  # type: ignore
            logger.info(
                "Loaded zero-shot intent classifier with model {}", ZERO_SHOT_MODEL
            )
        except Exception as e:
            intent_classifier = None
            logger.warning("Failed to load zero-shot classifier: {}", e)
    else:
        intent_classifier = None
    logger.info("Warmup complete.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


@app.post("/recommend")
def recommend(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must be non-empty")
    if _catalog_df is None:
        raise HTTPException(status_code=500, detail="Catalog not loaded")
    response = run_full_pipeline(query, _catalog_df, intent_classifier)
    return response


# =============================================================================
# CLI convenience
# =============================================================================


def recommend_single_query(query: str) -> list[str]:
    global _catalog_df, intent_classifier
    if _catalog_df is None:
        _catalog_df = load_catalog_snapshot().set_index("item_id", drop=False)
    response = run_full_pipeline(query, _catalog_df, intent_classifier)
    return [
        item.url
        for item in response.recommended_assessments
        if getattr(item, "url", None)
    ]
