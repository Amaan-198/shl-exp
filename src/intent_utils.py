from __future__ import annotations

from typing import List

from .constants import (
    ADMIN_TERMS,
    AI_RESEARCH_TERMS,
    CUSTOMER_SUPPORT_TERMS_CORE,
    DATA_ANALYST_INTENT_TERMS,
    DATA_ANALYST_PIN_TERMS,
    FINANCE_ANALYST_TOKENS,
    MARKETING_MANAGER_CONTEXT_TERMS,
    QA_TERMS,
    SALES_ENTRY_LEVEL_TERMS,
    WRITER_SEO_TERMS,
)


def _match_any(ql: str, keys: List[str]) -> bool:
    return any(k in ql for k in keys)


def _is_exec_culture_query(q: str) -> bool:
    """Return True only when the query clearly targets executive culture fit."""

    ql = q.lower()
    qpad = f" {ql} "

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


def _is_dev_query(q: str) -> bool:
    """Detect broad software-development asks without overfitting to a single stack."""

    ql = q.lower()
    if not ql.strip():
        return False

    direct_hits = [
        "software developer",
        "software engineer",
        "software development",
        "software dev",
        "full stack developer",
        "full-stack developer",
        "full stack engineer",
        "full-stack engineer",
        "frontend developer",
        "front-end developer",
        "backend developer",
        "back-end developer",
        "web developer",
        "application developer",
        "applications developer",
        "mobile developer",
        "android developer",
        "ios developer",
        "devops engineer",
    ]
    if any(term in ql for term in direct_hits):
        return True

    if "developer" in ql and any(
        ctx in ql
        for ctx in [
            "software",
            "application",
            "product",
            "full stack",
            "full-stack",
            "frontend",
            "front-end",
            "backend",
            "back-end",
            "web",
            "mobile",
            "cloud",
            "devops",
            "engineering",
            "technology",
            "tech",
        ]
    ):
        return True

    if any(token in ql for token in ["programmer", "coding", "programming", "software cod"]):
        if any(ctx in ql for ctx in ["software", "developer", "engineer", "technology", "tech"]):
            return True

    return False


def _intent_keys_from_query(query: str) -> List[str]:
    """Map the raw query into small intent keys that drive seed injection."""

    ql = query.lower()
    keys: List[str] = []

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
    if _match_any(ql, ["qa engineer", "qa ", "quality assurance", "testing", "selenium"]):
        keys += ["qa engineer", "quality assurance", "qa_testing"]
    if _match_any(ql, ADMIN_TERMS + ["data entry"]):
        keys += ["admin_ops"]
    if _match_any(ql, WRITER_SEO_TERMS + ["email writing"]):
        keys += ["content_marketing"]
    if ("marketing manager" in ql) or (
        "marketing" in ql and any(k in ql for k in MARKETING_MANAGER_CONTEXT_TERMS)
    ):
        keys += ["marketing_manager", "marketing"]
    if _match_any(
        ql,
        ["entry level sales", "sales associate", "spoken english", "svar"]
        + [f"sales {term}" for term in SALES_ENTRY_LEVEL_TERMS],
    ) or (
        "sales" in ql and any(k in ql for k in SALES_ENTRY_LEVEL_TERMS)
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
            "software developer",
            "software engineer",
            "software development",
            "fullstack developer",
            "full stack developer",
        ],
    ):
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

    if _is_dev_query(query):
        keys += ["dev_generic"]
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
    if any(k in ql for k in CUSTOMER_SUPPORT_TERMS_CORE):
        keys += ["customer_support"]
    if "finance" in ql and "analyst" in ql:
        keys += ["finance_analyst"]
    if any(k in ql for k in AI_RESEARCH_TERMS):
        keys += ["ai_research_eng", "data_analyst"]

    deduped = list(dict.fromkeys(keys))
    return _limit_intent_keys(deduped, query)


def _limit_intent_keys(keys: List[str], query: str) -> List[str]:
    """Clamp intent keys for very long job descriptions to avoid over-seeding."""

    if not keys:
        return keys

    ql = query.lower()
    archetype_groups: dict[str, List[str]] = {
        "java_dev": ["java", "developer", "engineer"],
        "dev_generic": [
            "software developer",
            "software engineer",
            "programmer",
            "coding",
            "programming",
        ],
        "qa engineer": QA_TERMS + ["qa "],
        "quality assurance": QA_TERMS,
        "qa_testing": QA_TERMS,
        "data_analyst": DATA_ANALYST_INTENT_TERMS + DATA_ANALYST_PIN_TERMS,
        "marketing_manager": MARKETING_MANAGER_CONTEXT_TERMS + ["marketing manager"],
        "marketing": [
            "marketing",
            "brand",
            "campaign",
            "demand generation",
            "seo",
            "content",
        ],
        "content_marketing": WRITER_SEO_TERMS + ["email writing"],
        "admin_ops": ADMIN_TERMS + ["data entry"],
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
        "customer_support": CUSTOMER_SUPPORT_TERMS_CORE + ["spoken english"],
        "finance_analyst": FINANCE_ANALYST_TOKENS,
        "ai_research_eng": AI_RESEARCH_TERMS,
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
        max_keys = 4

    dedup.sort(key=lambda k: scores.get(k, 0), reverse=True)
    return dedup[:max_keys]
