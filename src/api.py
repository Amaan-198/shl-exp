from __future__ import annotations
import re
import warnings
import os
import math
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
    INTENT_KEY_ALIASES,
)
from .constants import (
    ADMIN_TERMS,
    AI_RESEARCH_TERMS,
    CUSTOMER_SUPPORT_TERMS_CORE,
    CUSTOMER_SUPPORT_TERMS_EXTENDED,
    DATA_ANALYST_INTENT_TERMS,
    DATA_ANALYST_PIN_TERMS,
    MARKETING_MANAGER_CONTEXT_TERMS,
    QA_TERMS,
    SALES_ENTRY_LEVEL_TERMS,
    WRITER_SEO_TERMS,
)
from .intent_utils import (
    _intent_keys_from_query as _intent_keys_from_query_impl,
    _limit_intent_keys as _limit_intent_keys_impl,
    _is_dev_query as _is_dev_query_impl,
    _is_exec_culture_query as _is_exec_culture_query_impl,
)
from .pipeline_types import ScoredCandidate
from .post_processing import (
    _apply_category_balance,
    _apply_category_filter,
    _apply_domain_vetoes,
    _apply_dynamic_cutoff,
    _apply_generic_penalty,
    _ensure_min_category_diversity,
    _filter_domain_candidates,
    _hard_drop_if_strong_tech,
    _hard_duration_filter,
    _post_rank_adjustments,
)

_is_exec_culture_query = _is_exec_culture_query_impl
_is_dev_query = _is_dev_query_impl
_intent_keys_from_query = _intent_keys_from_query_impl
_limit_intent_keys = _limit_intent_keys_impl

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
    """Delegate to the shared intent helper for exec/culture detection."""

    return _is_exec_culture_query_impl(q)


def _is_dev_query(q: str) -> bool:
    """Delegate to the shared intent helper for broad developer detection."""

    return _is_dev_query_impl(q)


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
    is_dev_generic = _is_dev_query(query)
    wants_40 = ("40" in q and "min" in q) or "40 minutes" in q
    is_qa = any(k in q for k in QA_TERMS)
    is_admin = any(k in q for k in ADMIN_TERMS)
    is_sales_grad = ("sales" in q) and any(k in q for k in SALES_ENTRY_LEVEL_TERMS)
    # STRICTER: don't let 'community' trigger marketing manager by itself
    is_marketing_mgr = ("marketing manager" in q) or (
        "marketing" in q and any(k in q for k in MARKETING_MANAGER_CONTEXT_TERMS)
    )
    is_writer_seo = any(k in q for k in WRITER_SEO_TERMS)
    is_data_analyst = ("data analyst" in q) or any(k in q for k in DATA_ANALYST_INTENT_TERMS)
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
    is_customer_support = any(k in q for k in CUSTOMER_SUPPORT_TERMS_CORE)
    is_finance_analyst = "finance" in q and "analyst" in q
    is_ai_research = any(k in q for k in AI_RESEARCH_TERMS)

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
    elif is_dev_generic:
        must_slug_packs += [
            ["programming-concepts"],
            ["core-java-entry-level", "core-java-advanced-level"],
        ]
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




def _intent_keys_from_query(query: str) -> list[str]:
    """Delegate to the shared intent helper for intent key extraction."""

    return _intent_keys_from_query_impl(query)


def _limit_intent_keys(keys: list[str], query: str) -> list[str]:
    """Delegate to the shared intent helper for key clamping."""

    return _limit_intent_keys_impl(keys, query)
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
        canonical = INTENT_KEY_ALIASES.get(key, key)
        wanted_phrases += RETRIEVAL_BOOST_SEEDS.get(canonical, [])
        wanted_phrases += EXPANSION_LIBRARY.get(canonical, [])
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
    is_dev_generic = _is_dev_query(q)
    wants_40 = ("40" in ql and "min" in ql) or "40 minutes" in ql
    is_qa = any(k in ql for k in QA_TERMS)
    is_admin = any(k in ql for k in ADMIN_TERMS)
    wants_30_40 = any(k in ql for k in ["30-40", "30 – 40", "30 to 40"]) or (
        "30" in ql and "40" in ql and "min" in ql
    )
    is_sales_grad = ("sales" in ql) and any(k in ql for k in SALES_ENTRY_LEVEL_TERMS)
    # STRICTER: avoid "community" alone
    is_marketing_mgr = ("marketing manager" in ql) or (
        "marketing" in ql and any(k in ql for k in MARKETING_MANAGER_CONTEXT_TERMS)
    )
    is_writer_seo = any(k in ql for k in WRITER_SEO_TERMS)
    is_data_analyst = ("data analyst" in ql) or any(k in ql for k in DATA_ANALYST_PIN_TERMS)

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
    elif is_dev_generic:
        pinned += [
            ["programming concepts"],
            ["core java entry level", "core java advanced level"],
        ]
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
