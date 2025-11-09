from __future__ import annotations

"""
Text normalisation helpers shared across catalog building and retrieval.

The goal is to have a single, well-defined place that turns free-form
HTML / marketing copy / user queries into something reasonably clean
for both lexical and dense search.

Public helpers:

* basic_clean(text) -> str
    Light-weight clean used when building the catalog snapshot.

* normalize_for_lexical_index(text) -> str
    Heavier normalisation used before indexing / matching.

* lexical_tokens_for_bm25(text) -> List[str]
    Tokeniser for BM25 / lexical retrieval that mirrors the above
    normalisation so queries and documents see the same view of text.
"""

from typing import Dict, Iterable, List
import re
import unicodedata

from . import config

try:  # optional HTML stripping dependency
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - bs4 is optional
    BeautifulSoup = None  # type: ignore

# ---------------------------------------------------------------------------
# Config shims – work with both the original config.py and the patched one.
# ---------------------------------------------------------------------------

MAX_INPUT_CHARS: int = int(getattr(config, "MAX_INPUT_CHARS", 8000))

# In the original project this is called SYNONYM_MAP; in the patchset it is
# called SYNONYMS.  We support either name.
if hasattr(config, "SYNONYM_MAP"):
    _CONFIG_SYNONYMS: Dict[str, str] = dict(getattr(config, "SYNONYM_MAP"))
elif hasattr(config, "SYNONYMS"):
    _CONFIG_SYNONYMS = dict(getattr(config, "SYNONYMS"))
else:
    _CONFIG_SYNONYMS = {}

# Phrase locks: if config.PHRASE_LOCKS is present we use it; otherwise we use
# a small local default just to keep key phrases together.
_DEFAULT_PHRASE_LOCKS: Dict[str, str] = {
    # programming / data science
    "data structures": "data_structures",
    "machine learning": "machine_learning",
    "neural network": "neural_network",
    "deep learning": "deep_learning",
    "data analysis": "data_analysis",
    "data visualization": "data_visualization",
    "visualisation": "visualization",
    "business intelligence": "business_intelligence",
    # soft-skills
    "critical thinking": "critical_thinking",
    "problem solving": "problem_solving",
    "conflict management": "conflict_management",
    "employee engagement": "employee_engagement",
    "customer service": "customer_service",
}
_PHRASE_LOCKS: Dict[str, str] = dict(
    getattr(config, "PHRASE_LOCKS", _DEFAULT_PHRASE_LOCKS)
)

# Optional domain-specific vocabulary injection used in the patchset.  If it
# does not exist we simply do not use it.
_ESCO_INJECTIONS: Dict[str, Iterable[str]] = dict(
    getattr(config, "ESCO_INJECTIONS", {})
)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    if not text:
        return ""
    if BeautifulSoup is None:
        # Fallback – remove very simple tags
        return re.sub(r"<[^>]+>", " ", text)
    try:
        soup = BeautifulSoup(text, "html.parser")  # type: ignore[call-arg]
        return soup.get_text(" ", strip=True)
    except Exception:
        # If bs4 misbehaves, fall back to a crude strip
        return re.sub(r"<[^>]+>", " ", text)


def _normalise_unicode(text: str) -> str:
    # Normalise quotes, accents etc. into a consistent representation.
    text = unicodedata.normalize("NFKC", text)
    # Replace fancy quotes / dashes with ASCII variants
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def _apply_phrase_locks(text: str) -> str:
    """Replace multi-word phrases with single underscore-joined tokens.

    This is applied *before* lower-casing so we match regardless of case.
    """
    if not _PHRASE_LOCKS:
        return text
    out = text
    # Longer phrases first to avoid partial overlaps.
    for phrase, locked in sorted(_PHRASE_LOCKS.items(), key=lambda kv: -len(kv[0])):
        if not phrase:
            continue
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        out = pattern.sub(locked, out)
    return out


def _apply_synonyms(text: str) -> str:
    """Apply synonym replacements in a token-aware way.

    We treat keys as whole tokens – 'ml' will not rewrite 'email'.
    """
    if not _CONFIG_SYNONYMS:
        return text
    out = text
    for src, dst in _CONFIG_SYNONYMS.items():
        if not src:
            continue
        # Word-boundary style pattern that still allows underscores / + / #
        pattern = re.compile(
            r"(?<![A-Za-z0-9_+#.])" + re.escape(src) + r"(?![A-Za-z0-9_+#.])",
            flags=re.IGNORECASE,
        )
        out = pattern.sub(dst, out)
    return out


def _inject_esco_tokens(norm_text: str) -> List[str]:
    """Optionally inject ESCO / domain tokens based on simple triggers.

    If ESCO_INJECTIONS is not configured this simply returns [].
    """
    if not _ESCO_INJECTIONS:
        return []
    extra: List[str] = []
    lower = norm_text.lower()
    for trigger, tokens in _ESCO_INJECTIONS.items():
        try:
            trig = trigger.lower()
        except Exception:
            trig = str(trigger).lower()
        if trig and trig in lower:
            for t in tokens:
                tok = str(t).strip().lower()
                if tok:
                    extra.append(tok.replace(" ", "_"))
    return extra


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def basic_clean(text: str | None) -> str:
    """Light-weight clean for catalog fields.

    * strips HTML
    * normalises unicode and whitespace
    * truncates excessively long inputs
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Guard against pathological inputs
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    text = _strip_html(text)
    text = _normalise_unicode(text)

    # Normalise whitespace
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_lexical_index(text: str | None) -> str:
    """Heavier normalisation used for both documents and queries."""
    norm = basic_clean(text)
    if not norm:
        return ""

    # Apply phrase locks before lower-casing so matching is case-insensitive.
    norm = _apply_phrase_locks(norm)

    norm = norm.lower()
    norm = _apply_synonyms(norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def lexical_tokens_for_bm25(text: str | None) -> List[str]:
    """Tokenise text in a BM25-friendly way.

    The tokeniser deliberately keeps things like 'c#', 'c++', 'asp.net'
    and 'machine_learning' as single tokens, and preserves term
    frequencies for BM25 scoring.
    """
    norm = normalize_for_lexical_index(text)
    if not norm:
        return []

    # Base tokens from the normalised string
    tokens = re.findall(r"[a-z0-9_+#.]+", norm)

    # Optional ESCO / domain token injection (can de-dup just these if you want)
    extra = _inject_esco_tokens(norm)
    tokens.extend(extra)

    return tokens


def normalize_query(text: str | None) -> str:
    """Alias used in a couple of tests; kept for backwards compatibility."""
    return normalize_for_lexical_index(text)


