from __future__ import annotations

"""Shared keyword lists used across the recommender heuristics.

These constants intentionally mirror the legacy inline literals so behaviour remains
unchanged while allowing different modules to reuse the same vocabulary without
copy/paste drift.
"""

CUSTOMER_SUPPORT_TERMS_CORE = [
    "customer support",
    "customer service",
    "call center",
    "contact center",
]

CUSTOMER_SUPPORT_TERMS_EXTENDED = CUSTOMER_SUPPORT_TERMS_CORE + [
    "voice process",
    "international call center",
]

SALES_ENTRY_LEVEL_TERMS = [
    "entry level",
    "entry-level",
    "graduate",
    "fresher",
    "0-2",
]

MARKETING_MANAGER_CONTEXT_TERMS = [
    "brand",
    "campaign",
    "demand generation",
    "seo",
    "content",
]

WRITER_SEO_TERMS = [
    "content writer",
    "content writing",
    "copywriter",
    "seo",
]

ADMIN_TERMS = [
    "assistant admin",
    "administrative assistant",
    "bank admin",
    "bank administrative",
]

QA_TERMS = [
    "qa",
    "quality assurance",
    "selenium",
    "manual testing",
    "webdriver",
    "test case",
]

DATA_ANALYST_INTENT_TERMS = [
    " sql ",
    " excel ",
    " tableau ",
    " power bi ",
    " analytics ",
]

DATA_ANALYST_PIN_TERMS = [
    "sql",
    "excel",
    "tableau",
    "python",
]

AI_RESEARCH_TERMS = [
    " ai ",
    " artificial intelligence",
    "machine learning",
    " ml ",
    "research engineer",
    " llm ",
    " rag ",
]

FINANCE_ANALYST_TOKENS = [
    "finance",
    "analyst",
    "excel",
    "kpi",
    "forecast",
    "budget",
    "numerical",
]
