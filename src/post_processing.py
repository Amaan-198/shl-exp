"""Post-ranking heuristics and domain-specific adjustments."""

from __future__ import annotations

import re
from typing import List

import pandas as pd

from .config import RESULT_MAX, RESULT_MIN
from .constants import (
    ADMIN_TERMS,
    CUSTOMER_SUPPORT_TERMS_EXTENDED,
    QA_TERMS,
    SALES_ENTRY_LEVEL_TERMS,
)
from .intent_utils import _is_dev_query as _is_dev_query_impl
from .pipeline_types import ScoredCandidate
from .query_analysis import (
    _TECH_KEYWORDS,
    _ENTRY_LEVEL_KEYWORDS,
    _INTENT_KEYWORDS,
    _categories_for_item,
    _duration_adjust,
    _get_query_intent_categories,
    _minutes_hint_from_query,
    _parse_duration_window,
    _role_level,
)

_is_dev_query = _is_dev_query_impl


# ---------------------------------------------------------------------------
# Keyword banks specific to post-ranking heuristics
# ---------------------------------------------------------------------------

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
_STACK_FOCUS_TECH_TERMS = [
    "python",
    "sql",
    "javascript",
    "java",
    "nodejs",
    "node.js",
    "frontend",
    "front end",
    "fullstack",
]
_STACK_FOCUS_REQUIRED_TERMS = [
    "python",
    "sql",
    "java",
    "javascript",
]
_STACK_FOCUS_SOFTENERS = [
    "excel",
    "tableau",
    "power bi",
    "analytics",
    "visualization",
]
_AI_HEAVY_NOISE = [
    "excel",
    "tableau",
    "power bi",
    "word",
    "microsoft office",
    "excel 365",
]
_TOOL_NOISE_TERMS = [
    "data warehousing",
    "data warehouse",
    "ssas",
    "etl",
    "spark",
    "hadoop",
    "cognos",
    "ssis",
]
_COMMERCIAL_ROLE_TERMS = [
    "sales",
    "marketing",
    "presales",
    "customer",
    "client",
    "proposal",
    "demo",
]
_CLIENT_CONTACT_TERMS = [
    "client",
    "customer",
    "stakeholder",
    "presentation",
    "communication",
    "teamwork",
    "collaboration",
]
_AI_SOFTENERS = [
    "storytelling",
    "communication",
    "presentation",
    "stakeholder",
    "business",
]
_WRITER_SOFTENERS = [
    "content writer",
    "copywriter",
    "copy writer",
    "blog",
    "seo",
    "search engine",
]



def _hard_duration_filter(
    query: str, ranked: List[ScoredCandidate], catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Remove items that exceed the query's hard duration hints."""

    max_minutes, _ = _minutes_hint_from_query(query)
    if max_minutes is None:
        return ranked
    kept: List[ScoredCandidate] = []
    for c in ranked:
        try:
            dur = float(catalog_df.loc[c.item_id, "duration"] or 0)
        except Exception:
            dur = 0.0
        if dur and dur > (max_minutes + 12):
            continue
        kept.append(c)
    return kept or ranked



def _normalize_basename(name: str) -> str:
    """Normalise an assessment name for deduplication checks."""

    base = name.lower()
    base = re.sub(r"[\s&/\-]+", "", base)
    base = re.sub(r"[^a-z0-9]", "", base)
    return base



def _apply_generic_penalty(
    ranked: List[ScoredCandidate], catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Down-weight catalogue staples that dominate when reranker ties occur."""

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
    penalised.sort(key=lambda cand: (-float(cand.rerank_score), cand.item_id))
    return penalised



def _post_rank_adjustments(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Apply context-aware boosts and demotions after reranking."""

    q_lower = query.lower()

    if "collaborat" in q_lower or "business team" in q_lower:
        query_cats = _get_query_intent_categories(query) | {"behaviour"}
    else:
        query_cats = _get_query_intent_categories(query)

    role = _role_level(query)
    dur_min, dur_max = _parse_duration_window(query)

    is_entry = role == "grad"
    is_client = any(k in q_lower for k in _CLIENT_CONTACT_TERMS)
    is_strong_tech = (
        "technical" in query_cats and "behaviour" not in query_cats
    ) or any(k in q_lower for k in _TECH_KEYWORDS)
    is_dev_query = _is_dev_query(query)

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
    is_customer_support = any(k in q_lower for k in CUSTOMER_SUPPORT_TERMS_EXTENDED)
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
                "performance marketing",
            ]
        )
    )
    is_product_manager = ("product manager" in q_lower) or (
        " product " in q_lower and " manager" in q_lower
    ) or ("product management" in q_lower)
    is_presales = "presales" in q_lower or "pre-sales" in q_lower

    ai_query = any(kw in q_lower for kw in _AI_KEYWORDS)
    has_analytics_terms = any(kw in q_lower for kw in _ANALYTICS_KEYWORDS)
    stack_focus_terms = [kw for kw in _STACK_FOCUS_REQUIRED_TERMS if kw in q_lower]
    stack_focus = len(stack_focus_terms) >= 2
    stack_focus_no_analytics = stack_focus and not any(
        soft in q_lower for soft in _STACK_FOCUS_SOFTENERS
    )
    multi_lang_requested: List[str] = []
    language_query_map = {
        "python": ["python"],
        "sql": ["sql", "sql server"],
        "javascript": ["javascript", "java script", "nodejs", "node.js", " js "],
        "java": ["core java", " java ", " java,", " java.", " java-", " java/", " java)"]
    }
    for lang, tokens in language_query_map.items():
        if any(tok in q_lower for tok in tokens):
            multi_lang_requested.append(lang)
    multi_lang_requested = list(dict.fromkeys(multi_lang_requested))
    multi_lang_focus = len(multi_lang_requested) >= 2
    has_multi_lang_without_analytics = multi_lang_focus and not has_analytics_terms
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
    explicit_behaviour_request = any(
        k in q_lower
        for k in ["personality", "behaviour", "behavior", "culture fit", "values fit"]
    )
    wants_cognitive_signal = any(
        k in q_lower for k in ["iq", "smart", "intelligent", "cognitive", "aptitude", "reasoning"]
    )
    wants_dev_cognitive = is_dev_query and wants_cognitive_signal
    ai_query = any(kw in q_lower for kw in _AI_KEYWORDS)
    long_query = len(q_lower) >= 800
    heavy_analytics_terms = [
        "data warehousing",
        "data warehouse",
        "ssas",
        "etl",
        "spark",
        "hadoop",
        "microsoft excel 365",
        "excel 365",
        "appdynamics",
        "cisco",
    ]
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
    is_contact_centre = any(
        k in q_lower
        for k in (CUSTOMER_SUPPORT_TERMS_EXTENDED + ["contact centre", "call centre", "bpo"])
    )

    adjusted: List[ScoredCandidate] = []
    seen_bases: dict[str, bool] = {}
    sales_spoken_count = 0

    for c in ranked:
        iid = c.item_id
        try:
            row = catalog_df.loc[iid]
        except Exception:
            row = {}

        base_score = float(c.rerank_score)
        score = base_score

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
        if is_client and any(t in _CLIENT_ALLOWED_TYPES for t in types_list):
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
                if is_sales_grad or is_customer_support:
                    score -= 0.15
                else:
                    score -= 0.05
            else:
                seen_bases[base] = True

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

        if has_multi_lang_without_analytics:
            matches = 0
            for lang in multi_lang_requested:
                if any(kw in name_desc for kw in language_query_map.get(lang, [])):
                    matches += 1
            if matches >= 2:
                score += 0.12
            elif matches == 1:
                score += 0.06
            if any(kw in name_desc for kw in _STACK_FOCUS_SOFTENERS):
                score -= 0.08

        if wants_dev_cognitive:
            if any(
                kw in name_desc
                for kw in [
                    "verify g+",
                    "verify numerical",
                    "verify verbal",
                    "inductive",
                    "aptitude",
                ]
            ):
                score += 0.10
            if "programming" in name_desc or "developer" in name_desc:
                score += 0.10
            if any(
                kw in name_desc
                for kw in ["occupational personality", "culture fit", "values", "behavioral"]
            ):
                score -= 0.08

        if wants_english and any(
            kw in name_desc
            for kw in [
                "spoken english",
                "business communication",
                "writex",
                "interpersonal communications",
                "written english",
            ]
        ):
            score += 0.12

        if wants_english and "svar" in lname:
            sales_spoken_count += 1
            if sales_spoken_count >= 2:
                score -= 0.10

        if is_contact_centre and any(
            kw in name_desc
            for kw in [
                "customer service",
                "customer support",
                "contact centre",
                "call centre",
                "spoken english",
                "business communication",
                "writex",
            ]
        ):
            score += 0.16

        if explicit_behaviour_request and any(
            kw in name_desc
            for kw in [
                "occupational personality questionnaire",
                "behavioral",
                "behavioural",
                "culture",
                "values",
                "situational judgement",
            ]
        ):
            score += 0.12

        if is_exec and ("leadership" in name_desc or "manager" in name_desc):
            score += 0.12
        if is_exec and any(
            kw in name_desc
            for kw in ["entry-level", "graduate", "intern", "campus", "foundation"]
        ):
            score -= 0.20

        if is_client and any(kw in name_desc for kw in ["client", "customer", "stakeholder"]):
            score += 0.08
        if is_client and not any(k in name_desc for k in _CLIENT_ALLOWED_TYPES):
            score -= 0.05

        if is_screening and any(
            kw in name_desc
            for kw in [
                "screen",
                "screening",
                "assessment battery",
                "shortlist",
                "hiring",
            ]
        ):
            score += 0.05

        if is_cog_plus_personality:
            if any(
                kw in name_desc
                for kw in [
                    "verify",
                    "numerical ability",
                    "verbal ability",
                    "inductive",
                    "occupational personality",
                ]
            ):
                score += 0.08

        if wants_english and not any(
            kw in name_desc
            for kw in [
                "spoken english",
                "business communication",
                "writex",
                "interpersonal communications",
                "english comprehension",
            ]
        ):
            score -= 0.08

        if is_content_writer:
            if any(kw in name_desc for kw in ["written english", "copywriting", "seo"]):
                score += 0.12
            if not any(kw in name_desc for kw in _WRITER_SOFTENERS):
                score -= 0.08

        if ai_query and not any(kw in name_desc for kw in _AI_KEYWORDS):
            if any(kw in name_desc for kw in _AI_HEAVY_NOISE):
                score -= 0.08

        if ai_query and any(kw in name_desc for kw in ["python", "machine learning"]):
            score += 0.10

        if ai_query and any(kw in name_desc for kw in _AI_SOFTENERS):
            score += 0.05

        if ai_query and any(kw in name_desc for kw in _TOOL_NOISE_TERMS):
            score -= 0.08

        if ai_query and any(kw in name_desc for kw in _COMMERCIAL_ROLE_TERMS):
            score += 0.05

        if ai_query and any(kw in name_desc for kw in _STACK_FOCUS_TECH_TERMS):
            score += 0.06

        if ai_query and "automata" in name_desc and "sql" not in q_lower:
            score -= 0.10

        if is_presales:
            if presales_tool_signal and any(
                kw in name_desc
                for kw in [
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

        if long_query:
            score = base_score + (score - base_score) * 0.6

        adjusted.append(
            ScoredCandidate(
                item_id=c.item_id, fused_score=c.fused_score, rerank_score=score
            )
        )

    adjusted.sort(key=lambda cand: (-float(cand.rerank_score), cand.item_id))
    return adjusted



def _hard_drop_if_strong_tech(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Strip clearly behavioural tests when the brief is purely technical."""

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
    """Keep technical categories when the brief is clearly technical."""

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



def _apply_domain_vetoes(
    query: str, ranked_list: List[ScoredCandidate], catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Apply role-specific demotions to keep off-domain items out of the slate."""

    ql = query.lower()
    is_dev_query = _is_dev_query(query)
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
    is_sales_grad = ("sales" in ql) and any(k in ql for k in SALES_ENTRY_LEVEL_TERMS)
    is_customer_support = any(k in ql for k in CUSTOMER_SUPPORT_TERMS_EXTENDED)
    is_presales = "presales" in ql or "pre-sales" in ql
    is_admin = any(k in ql for k in ADMIN_TERMS)
    is_qa = any(k in ql for k in QA_TERMS)
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
    has_analytics_terms = presales_mentions_analytics or any(
        k in ql for k in ["dashboard", "reporting", "data warehouse", "ssas", "etl"]
    )
    explicit_behaviour_request = any(
        k in ql for k in ["personality", "behaviour", "behavior", "culture fit", "values fit"]
    )
    wants_cognitive_signal = any(
        k in ql for k in ["iq", "smart", "intelligent", "cognitive", "aptitude", "reasoning"]
    )
    wants_dev_cognitive = is_dev_query and wants_cognitive_signal
    ai_query = any(k in ql for k in _AI_KEYWORDS)
    heavy_analytics_terms = [
        "data warehousing",
        "data warehouse",
        "ssas",
        "etl",
        "spark",
        "hadoop",
        "microsoft excel 365",
        "excel 365",
        "appdynamics",
        "cisco",
    ]

    non_tech = (
        is_consultant_io
        or is_marketing_mgr
        or is_sales_grad
        or is_admin
        or is_customer_support
        or is_presales
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
    analytics_noise = [
        "data warehousing",
        "data warehouse",
        "sql server analysis services",
        "ssas",
        "etl",
        "spark",
        "hadoop",
        "tableau",
    ]

    cleaned: List[ScoredCandidate] = []
    for c in ranked_list:
        row = catalog_df.loc[c.item_id] if c.item_id in catalog_df.index else {}
        blob = f"{row.get('name','')} {row.get('description','')} {row.get('search_text','')}".lower()
        score = float(c.rerank_score)

        if non_tech and any(k in blob for k in dev_noise):
            score -= 0.30

        if (is_customer_support or is_marketing_mgr or is_sales_grad or is_admin) and any(
            k in blob for k in analytics_noise
        ):
            score -= 0.30

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
            score -= 0.25

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

    cleaned.sort(key=lambda cand: (-float(cand.rerank_score), cand.item_id))
    return cleaned or ranked_list



def _apply_category_balance(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Ensure behaviour vs technical coverage when both intents are present."""

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
    if not to_promote:
        return ranked
    new_ranked: List[ScoredCandidate] = []
    seen: set[int] = set()
    for c in to_promote + ranked:
        if c.item_id in seen:
            continue
        new_ranked.append(c)
        seen.add(c.item_id)
    return new_ranked



def _apply_category_filter(
    ranked: List[ScoredCandidate], query: str, catalog_df: pd.DataFrame
) -> List[ScoredCandidate]:
    """Trim items when the query is clearly single-intent."""

    filtered: List[ScoredCandidate] = list(ranked)
    query_cats = _get_query_intent_categories(query)

    if "behaviour" in query_cats and "technical" in query_cats:
        return ranked

    if "behaviour" in query_cats and "technical" not in query_cats:
        tmp: List[ScoredCandidate] = []
        for c in filtered:
            try:
                row = catalog_df.loc[c.item_id]
            except Exception:
                tmp.append(c)
                continue
            cats = _categories_for_item(row)
            if "behaviour" in cats or "aptitude" in cats:
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
            if "technical" in cats or "aptitude" in cats:
                tmp2.append(c)
        filtered = tmp2

    if len(filtered) < max(5, len(ranked) // 3):
        return ranked

    return filtered



def _apply_dynamic_cutoff(
    final_ids: List[int], ranked_scores: dict[int, float], soft_target: int, query: str
) -> List[int]:
    """Clamp the final list using knee detection and duration hints."""

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
    """Top up final ids to guarantee at least *min_categories* distinct types."""

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


__all__ = [
    "_apply_category_balance",
    "_apply_category_filter",
    "_apply_domain_vetoes",
    "_apply_dynamic_cutoff",
    "_apply_generic_penalty",
    "_ensure_min_category_diversity",
    "_filter_domain_candidates",
    "_hard_drop_if_strong_tech",
    "_hard_duration_filter",
    "_normalize_basename",
    "_post_rank_adjustments",
]

