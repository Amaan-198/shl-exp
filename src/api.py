from __future__ import annotations
import re

"""
FastAPI application for the SHL recommender.

- Respects result policy: RESULT_MIN ≤ N ≤ RESULT_MAX
- Uses RESULT_DEFAULT_TARGET as a soft target only
- Flat-score aware dynamic cutoff (does NOT over-prune when reranker is offline)
- Family sibling completion can top up toward the soft target without exceeding MAX
"""

from typing import List
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from loguru import logger  # type: ignore
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)

    class _FallbackLogger:
        def __init__(self, logger):
            self._logger = logger
        def info(self, msg: str, *args, **kwargs) -> None: self._logger.info(msg.format(*args))
        def warning(self, msg: str, *args, **kwargs) -> None: self._logger.warning(msg.format(*args))
        def error(self, msg: str, *args, **kwargs) -> None: self._logger.error(msg.format(*args))
        def exception(self, msg: str, *args, **kwargs) -> None: self._logger.exception(msg.format(*args))
    logger = _FallbackLogger(logging.getLogger(__name__))

from pydantic import BaseModel, Field
import pandas as pd
import os

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
)

# optional transformers
try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None

from .catalog_build import load_catalog_snapshot
from .retrieval import retrieve_candidates
from .rerank import rerank_candidates
from .mmr import load_item_embeddings, mmr_select
from .balance import allocate
from .mapping import map_items_to_response
from .jd_fetch import fetch_and_extract


# -----------------------
# Family helpers (soft completion)
# -----------------------

def _slug_from_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    u = url.strip().lower()
    m = re.search(r"/view/([^/?#]+)", u)
    return m.group(1) if m else u.rstrip("/")


def _family_base(slug: str) -> str:
    s = family_slug(slug)
    s = re.sub(r"-(essentials|advanced|advanced-level|entry-level|foundation|v\d+)$", "", s)
    return s


def _family_expand_ids(catalog_df: pd.DataFrame, seed_ids: List[int], target: int) -> List[int]:
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


# -----------------------
# Pipeline
# -----------------------

def _hard_drop_if_strong_tech(ranked, cleaned_query, catalog_df):
    q = f" {cleaned_query.lower()} "
    tech_signals = [
        " python ", " backend ", " data_structures ", " data structure ",
        " machine_learning ", " neural_network ", " ml ", " ai ",
        " excel ", " tableau ", " power_bi ", " visualization ", " analytics ", " sql "
    ]
    if not any(s in q for s in tech_signals):
        return ranked

    drop_name_rx = re.compile(
        r"(?:\bfiling\b|\breviewing forms\b|\bfollowing instructions\b|"
        r"\bwrit(?:ten )?(?:english|spanish)\b|"
        r"\bretail sales\b|\bcontact center\b|"
        r"\bsales(?:\s*&|\s*and)\s*service(?:\s*phone|\s*simulation)?)",
        re.I,
    )

    keep = []
    for c in ranked:
        row = catalog_df.iloc[c.item_id]
        name = (row.get("name") or "").strip()
        if drop_name_rx.search(name):
            continue
        keep.append(c)
    return keep


def run_full_pipeline(
    query: str,
    catalog_df: pd.DataFrame,
    intent_model: object | None = None,
) -> RecommendResponse:
    # --- 0) URL detection & fetch ---
    if re.match(r"^https?://", query.strip(), re.IGNORECASE):
        logger.info("Query is a URL. Fetching content from: {}", query)
        fetched_text = fetch_and_extract(query)
        if not fetched_text:
            logger.warning("Failed to fetch or extract text from URL.")
            return RecommendResponse(recommended_assessments=[])
        query = fetched_text

    if not query or not query.strip():
        return RecommendResponse(recommended_assessments=[])

    # --- 1) Retrieval (BM25 + dense fusion) ---
    raw_query, cleaned_query, fused = retrieve_candidates(query)

    # --- 2) Cross-encoder reranking (robust to offline) ---
    ranked = rerank_candidates(cleaned_query, fused)

    # --- 3) Heuristics before MMR ---
    if not any(k in cleaned_query.lower() for k in ["leadership","communication","conflict","teamwork","empathy"]):
        ranked = _filter_domain_candidates(cleaned_query, ranked, catalog_df)
    ranked = _apply_generic_penalty(ranked, catalog_df)
    ranked = _post_rank_adjustments(ranked, cleaned_query, catalog_df)
    ranked = _hard_drop_if_strong_tech(ranked, cleaned_query, catalog_df)

    # Category balance / filter prior to MMR
    ranked = _apply_category_balance(ranked, cleaned_query, catalog_df)
    ranked = _apply_category_filter(ranked, cleaned_query, catalog_df)

    # --- 4) MMR diversification ---
    embeddings, ids = load_item_embeddings()
    mmr_ids = mmr_select(
        candidates=[(c.item_id, c.rerank_score) for c in ranked],
        embeddings=embeddings,
        ids=ids,
        k=RESULT_MAX,
        lambda_=0.7,
    )

    # --- 5) Intent classification (zero-shot if available) + priors ---
    if intent_model is not None:
        try:
            intent_labels = [INTENT_LABEL_TECHNICAL, INTENT_LABEL_PERSONALITY]
            intent_result = intent_model(cleaned_query, intent_labels, multi_label=False)
            score_map = {label: score for label, score in zip(intent_result["labels"], intent_result["scores"])}
            pt = float(score_map.get(INTENT_LABEL_TECHNICAL, 0.5))
            pb = float(score_map.get(INTENT_LABEL_PERSONALITY, 0.5))
            _q = f" {cleaned_query.lower()} "
            if any(k in _q for k in [" leadership ", " employee_engagement ", " conflict_management ", " interpersonal "]):
                pb = min(0.80, pb + 0.20)
            if any(k in _q for k in [" python ", " backend ", " data_structures ", " machine_learning ", " neural_network ", " excel ", " tableau ", " power_bi ", " visualization ", " analytics ", " sql "]):
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

    # --- 6) Build K/P/BOTH class map from test_type ---
    def _kp_class(types_list: list[str]) -> str:
        K_SET = {"Knowledge & Skills", "Ability & Aptitude", "Simulations", "Assessment Exercises"}
        P_SET = {"Personality & Behavior", "Competencies", "Development & 360", "Biodata & Situational Judgement"}
        has_k = any(t in K_SET for t in types_list)
        has_p = any(t in P_SET for t in types_list)
        if has_k and has_p: return "BOTH"
        if has_p: return "P"
        return "K"

    classes: dict[int, str] = {}
    for iid in mmr_ids:
        row = catalog_df.loc[iid]
        raw_types = row.get("test_type")
        types_list: list[str] = []
        if raw_types is None or (isinstance(raw_types, float) and raw_types != raw_types):
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

    # --- 7) Allocation toward MAX (then backfill in MMR order) ---
    try:
        final_ids = allocate(mmr_ids, classes, RESULT_MAX, pt=pt, pb=pb, catalog_df=catalog_df)
    except TypeError:
        final_ids = allocate(mmr_ids, classes, RESULT_MAX, pt=pt, pb=pb)

    # --- 8) Sizing policy (MIN/MAX respected; soft target used) ---
    soft_target = min(max(RESULT_MIN, RESULT_DEFAULT_TARGET), RESULT_MAX)

    score_lookup = {c.item_id: c.rerank_score for c in ranked}
    final_ids = _apply_dynamic_cutoff(final_ids, score_lookup, soft_target=soft_target)

    final_ids = _ensure_min_category_diversity(final_ids, ranked, catalog_df, min_categories=2)

    if len(final_ids) < soft_target and len(final_ids) >= RESULT_MIN:
        seed_take = min(max(soft_target, 20), len(ranked))
        seed_ids = [c.item_id for c in ranked[:seed_take]]
        seen = set(final_ids)
        ordered = list(final_ids) + [i for i in seed_ids if i not in seen]
        final_ids = _family_expand_ids(catalog_df, ordered, target=soft_target)

    if len(final_ids) < RESULT_MIN:
        needed = RESULT_MIN - len(final_ids)
        seen = set(final_ids)
        for c in ranked:
            if len(final_ids) >= RESULT_MIN: break
            if c.item_id not in seen:
                final_ids.append(c.item_id); seen.add(c.item_id)

    if len(final_ids) > RESULT_MAX:
        final_ids = final_ids[:RESULT_MAX]

    # --- 9) Map to API schema ---
    response = map_items_to_response(final_ids, catalog_df)
    if len(response.recommended_assessments) > RESULT_MAX:
        response.recommended_assessments = response.recommended_assessments[:RESULT_MAX]
    return response


# ================= Heuristics (unchanged structure; kept for continuity) ==============

@dataclass
class ScoredCandidate:
    item_id: int
    fused_score: float
    rerank_score: float

def _normalize_basename(name: str) -> str:
    base = name.lower()
    base = re.sub(r"[\s&/\-]+", "", base)
    base = re.sub(r"[^a-z0-9]", "", base)
    return base

_TECH_KEYWORDS = [
    "software","developer","developers","programmer","coder","engineer","engineers","engineering","technician",
    "technology","technical","coding","programming","devops","backend","front-end","frontend","fullstack","full-stack",
    "python","java",".net","c#","c++","javascript","machine learning","ml","neural network","neural_network","deep learning",
    "data engineer","data-engineer",
]
_TECH_ALLOWED_TYPES = {"Knowledge & Skills", "Ability & Aptitude"}
_GENERIC_PATTERNS = ["multitasking ability","360","verify","inductive reasoning","360 feedback"]
_NON_EN_LANGUAGES = ["spanish","french","german","mandarin","chinese","arabic","hindi","japanese","portuguese","italian","sv","svenska"]
_CLIENT_ALLOWED_TYPES = {"Personality & Behavior","Biodata & Situational Judgement","Knowledge & Skills"}
_ENTRY_LEVEL_KEYWORDS = ["entry-level","entry level","graduate","fresher","campus","intern","internship","0-2 years","0-2 yrs","0 to 2 years"]
_ENTRY_LEVEL_POSITIVE = ["verify g+","inductive","numerical","multitasking","entry-level","entry level","graduate",
                         "entry-level sales","entry level sales","sales representative","sales-representative",
                         "sales associate","technical sales"]
_ENTRY_LEVEL_NEGATIVE = ["expert","senior","advanced","salesforce","sap","dynamics"]
_DOMAIN_KEYWORDS = ["food","beverage","hospitality","accounting","retail","filing","front office","office management",
                    "restaurants","hotel","pharmaceutical","insurance","sales","marketing","customer service","support",
                    "filling","warehouse"]
_AI_KEYWORDS = ["artificial intelligence","ai","machine learning","ml","deep learning","data science","neural network",
                "computer vision","nlp","natural language"]
_PYTHON_KEYWORDS = ["python","django","flask","pandas","numpy","data structures","machine learning","data analysis",
                    "tensorflow","pytorch"]
_ANALYTICS_KEYWORDS = ["excel","tableau","power bi","visualization","visualisation","data viz","reporting","storytelling",
                       "analytics","data analytics","business intelligence"]
_DOMAIN_FOCUS_KEYWORDS = {
    "analytics": ["analytics","data analysis","business data","analyze","analyse","data-driven","reporting","insight",
                  "data insights","data interpretation"],
    "communication": ["communication","writing","presentation","interpersonal","client communication","collaboration",
                      "stakeholder management","storytelling","articulation"],
    "sales": ["sales","negotiation","customer","service orientation","customer service","client-facing","selling","retail","marketing"],
}
_COMMON_IRRELEVANT_PATTERNS = ["filing - names","filing - numbers","food science","food and beverage","front office management",
                               "following instructions","written english","filling","office management","office operations"]
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
    "behaviour": ["communication","interpersonal","presentation","leadership","teamwork","collaboration","stakeholder",
                  "client","customer","soft skills","relationship","partner","consultant","empathy","negotiation","service",
                  "orientation","sales","creative"],
    "aptitude": ["analytical","reasoning","logic","logical","numerical","inductive","aptitude","problem solving",
                 "quantitative","cognitive"],
}

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
        if cat: cats.add(cat)
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

def _apply_category_balance(ranked: List, query: str, catalog_df) -> List:
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
    to_promote: List = []
    for cat in missing:
        for c in ranked:
            try:
                row = catalog_df.loc[c.item_id]
            except Exception:
                continue
            if cat in _categories_for_item(row):
                to_promote.append(c)
                break
    new_ranked: List = []
    promoted_ids = {c.item_id for c in to_promote}
    for c in ranked:
        if c.item_id in promoted_ids and c not in new_ranked:
            new_ranked.append(c)
    for c in ranked:
        if c not in new_ranked:
            new_ranked.append(c)
    return new_ranked

def _apply_category_filter(ranked: List, query: str, catalog_df) -> List:
    query_cats = _get_query_intent_categories(query)
    filtered: List = list(ranked)
    if "behaviour" in query_cats and "technical" not in query_cats:
        tmp: List = []
        for c in filtered:
            try: row = catalog_df.loc[c.item_id]
            except Exception: tmp.append(c); continue
            cats = _categories_for_item(row)
            if cats == {"technical"}: continue
            tmp.append(c)
        filtered = tmp
    if "technical" in query_cats and "behaviour" not in query_cats:
        tmp2: List = []
        for c in filtered:
            try: row = catalog_df.loc[c.item_id]
            except Exception: tmp2.append(c); continue
            cats = _categories_for_item(row)
            if cats == {"behaviour"}: continue
            tmp2.append(c)
        filtered = tmp2
    if len(filtered) < max(5, len(ranked) // 3):
        return ranked
    return filtered

def _apply_dynamic_cutoff(final_ids: List[int], ranked_scores: dict[int, float], soft_target: int) -> List[int]:
    """Adaptive count: keep ≥ RESULT_MIN and ≤ RESULT_MAX; skip pruning on flat scores."""
    scores = [ranked_scores.get(i, 0.0) for i in final_ids]
    if not scores:
        return final_ids
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    # NEW: if scores are (near) flat (e.g., reranker offline -> all zeros), do NOT prune.
    if rng <= 1e-6:
        return final_ids[:soft_target]
    normalized = [(s - mn) / (rng if rng > 0 else 1.0) for s in scores]
    kept: List[int] = [iid for iid, n in zip(final_ids, normalized) if n >= 0.60]
    if len(kept) < RESULT_MIN:
        kept = final_ids[:RESULT_MIN]
    elif len(kept) > RESULT_MAX:
        kept = kept[:RESULT_MAX]
    return kept

def _ensure_min_category_diversity(final_ids: List[int], ranked: List, catalog_df, min_categories: int = 2) -> List[int]:
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

def _apply_generic_penalty(ranked: List, catalog_df) -> List[ScoredCandidate]:
    penalised: List[ScoredCandidate] = []
    for c in ranked:
        if hasattr(c, "item_id"):
            item_id = int(c.item_id)
            fused_score = float(getattr(c, "fused_score", getattr(c, "rerank_score", 0.0)))
            score = float(getattr(c, "rerank_score", fused_score))
        else:
            try:
                item_id_raw, score_raw = c
            except Exception:
                continue
            item_id = int(item_id_raw)
            score = float(score_raw)
            fused_score = score
        try:
            name = str(catalog_df.loc[item_id, "name"]).lower()
        except Exception:
            name = ""
        if any(pat in name for pat in _GENERIC_PATTERNS):
            score *= 0.7
        penalised.append(ScoredCandidate(item_id=item_id, fused_score=fused_score, rerank_score=score))
    penalised.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return penalised

def _post_rank_adjustments(ranked: List, query: str, catalog_df) -> List:
    q_lower = query.lower()
    query_cats = _get_query_intent_categories(query)

    is_entry = any(k in q_lower for k in _ENTRY_LEVEL_KEYWORDS)
    is_client = any(k in q_lower for k in ["client","customer","stakeholder","presentation","communication","teamwork","collaboration"])
    is_strong_tech = "technical" in query_cats and "behaviour" not in query_cats

    is_content_writer = any(k in q_lower for k in [
        "content writer","content-writing","content writing","copywriter","copy writer",
        "blog writer","seo","search engine optimization",
    ])
    wants_english = any(k in q_lower for k in [
        "english","spoken english","written english","english test","english comprehension",
        "business communication","communication skills","verbal ability","verbal test",
    ])
    is_exec = any(k in q_lower for k in [
        "coo","chief operating officer","cxo","executive","senior leadership","senior leader","leadership role",
    ])
    cares_culture = any(k in q_lower for k in [
        "culture fit","cultural fit","culturally a right fit","values fit","right fit for our culture",
    ])
    is_qa_query = any(k in q_lower for k in [
        "qa engineer","qa","quality assurance","software testing","tester","manual testing","selenium",
        "webdriver","test case","test plan","regression test",
    ])
    is_sales_grad = ("sales" in q_lower) and any(k in q_lower for k in [
        "entry level","entry-level","graduate","fresher","new graduates","0-2 years","0-2 yrs","0 to 2 years",
    ])

    adjusted: List = []
    seen_bases: dict[str, bool] = {}

    for c in ranked:
        iid = c.item_id
        try:
            row = catalog_df.loc[iid]
        except Exception:
            row = {}

        score = c.rerank_score

        try:
            name = str(row.get("name", ""))
        except Exception:
            name = ""

        try:
            desc = str(row.get("description", ""))
        except Exception:
            desc = ""

        lname = name.lower()
        name_desc = (name + " " + desc).lower()

        try:
            duration = float(row.get("duration", 0))
        except Exception:
            duration = 0.0
        if duration == 0:
            score -= 0.10

        if any(word in lname for word in ["report", "guide", "profile"]):
            score -= 0.10

        try:
            types = row.get("test_type", [])
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

        if any(k in q_lower for k in _TECH_KEYWORDS):
            if any(t in _TECH_ALLOWED_TYPES for t in types_list):
                score += 0.08

        if is_client:
            if any(t in {"Personality & Behavior","Biodata & Situational Judgement","Knowledge & Skills"} for t in types_list):
                score += 0.08

        if not any(lang in q_lower for lang in _NON_EN_LANGUAGES):
            if any(lang in name_desc for lang in _NON_EN_LANGUAGES):
                score -= 0.08

        if is_entry or is_sales_grad:
            if any(pat in lname for pat in _ENTRY_LEVEL_POSITIVE):
                score += 0.06
            if any(pat in lname for pat in _ENTRY_LEVEL_NEGATIVE):
                score -= 0.03

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
        if ai_query:
            if any(kw in name_desc for kw in _AI_KEYWORDS):
                score += 0.08
            else:
                score -= 0.05

        if any(kw in q_lower for kw in _PYTHON_KEYWORDS):
            if any(kw in name_desc for kw in _PYTHON_KEYWORDS):
                score += 0.10

        if any(kw in q_lower for kw in _ANALYTICS_KEYWORDS):
            if any(kw in name_desc for kw in _ANALYTICS_KEYWORDS):
                score += 0.10

        query_domains: set[str] = set()
        for dom, kws in _DOMAIN_FOCUS_KEYWORDS.items():
            if any(k in q_lower for k in kws):
                query_domains.add(dom)
        if query_domains:
            matches_domain = False
            for dom in query_domains:
                if any(kw in name_desc for kw in _DOMAIN_FOCUS_KEYWORDS.get(dom, [])):
                    matches_domain = True
                    break
            if matches_domain:
                score += 0.05
            else:
                score -= 0.05

        if not any(pat in q_lower for pat in _COMMON_IRRELEVANT_PATTERNS):
            for pat in _COMMON_IRRELEVANT_PATTERNS:
                if pat in lname:
                    if is_strong_tech:
                        score -= 0.25
                    else:
                        score -= 0.07
                    break

        if is_content_writer or wants_english:
            if any(kw in name_desc for kw in [
                "written english","english comprehension","english-comprehension","english language",
                "grammar","vocabulary","business communication",
            ]):
                score += 0.10
            elif any(kw in name_desc for kw in ["seo","search engine optimization","search-engine-optimization"]):
                score += 0.05

        if is_exec or cares_culture:
            if "opq" in name_desc or "occupational personality questionnaire" in name_desc:
                score += 0.10
            if "enterprise leadership" in name_desc:
                score += 0.10
            elif "leadership" in name_desc and "report" in name_desc:
                score += 0.07

        if is_qa_query:
            if any(kw in name_desc for kw in [
                "selenium","automata","manual testing","software testing","qa engineer","quality assurance",
            ]):
                score += 0.12
            if ("verify" in name_desc
                and any(kw in name_desc for kw in ["numerical","verbal","inductive"])
                and not any(kw in name_desc for kw in ["qa","quality assurance","selenium","test","testing"])
            ):
                score -= 0.05

        if is_sales_grad:
            if any(kw in name_desc for kw in [
                "entry-level sales","entry level sales","entry-level sales-7-1",
                "entry-level sales sift-out","sales representative solution",
                "sales representative","sales-representative","technical sales associate",
            ]):
                score += 0.10
            if any(kw in name_desc for kw in ["salesforce","sap","dynamics"]):
                score -= 0.07

        adjusted.append(type(c)(item_id=c.item_id, fused_score=c.fused_score, rerank_score=score))
    adjusted.sort(key=lambda c: (-float(c.rerank_score), c.item_id))
    return adjusted

def _hard_drop_if_strong_tech(ranked: List, query: str, catalog_df) -> List:
    q_lower = query.lower()
    tech_hit = any(k in q_lower for k in _TECH_KEYWORDS)
    ai_hit = any(k in q_lower for k in _AI_KEYWORDS)
    analytics_hit = any(k in q_lower for k in _ANALYTICS_KEYWORDS)
    strong = (tech_hit or ai_hit or analytics_hit) and not any(k in q_lower for k in _INTENT_KEYWORDS.get("behaviour", []))
    if not strong:
        return ranked
    hard_drop_patterns = [
        "following instructions","reviewing forms","filing - names","filing - numbers",
        "written english","written spanish","ms office basic computer literacy",
    ]
    out: List = []
    for c in ranked:
        try:
            name = str(catalog_df.loc[c.item_id, "name"]).lower()
        except Exception:
            out.append(c); continue
        if any(pat in name for pat in hard_drop_patterns):
            continue
        out.append(c)
    return out or ranked

def _filter_domain_candidates(query: str, ranked: List, catalog_df) -> List:
    q_lower = query.lower()
    if not any(k in q_lower for k in _TECH_KEYWORDS):
        return ranked
    filtered: List = []
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


# -----------------------
# FastAPI app + startup
# -----------------------

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
            intent_classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)  # type: ignore
            logger.info("Loaded zero-shot intent classifier with model {}", ZERO_SHOT_MODEL)
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


# -----------------------
# CLI convenience
# -----------------------

def recommend_single_query(query: str) -> list[str]:
    global _catalog_df, intent_classifier
    if _catalog_df is None:
        _catalog_df = load_catalog_snapshot().set_index("item_id", drop=False)
    response = run_full_pipeline(query, _catalog_df, intent_classifier)
    return [item.url for item in response.recommended_assessments if getattr(item, "url", None)]
