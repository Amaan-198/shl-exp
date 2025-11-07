from src.normalize import (
    apply_synonyms,
    basic_clean,
    lexical_tokens_for_bm25,
    normalize_for_lexical_index,
    normalize_query,
    simple_tokenize,
    strip_html,
    clamp_text_length,
)
from src.config import MAX_INPUT_CHARS


def test_strip_html_basic():
    html = "<p>Hello <b>world</b>!</p>"
    assert strip_html(html) == "Hello world!"


def test_basic_clean_trims_whitespace_and_html():
    raw = "   <div>Hello   world</div>\n"
    cleaned = basic_clean(raw)
    assert cleaned == "Hello world"


def test_simple_tokenize_splits_punctuation():
    text = "Senior JS/TS Engineer (Node.js, C#)!"
    tokens = simple_tokenize(text)
    # we're not too strict, but we expect core pieces
    assert "senior" in tokens
    assert "js" in tokens
    assert "ts" in tokens
    assert "node.js" in tokens or "node" in tokens
    assert "c#" in tokens


def test_apply_synonyms_expands_expected_terms():
    tokens = ["senior", "js", "node.js", "ml", "pm"]
    expanded = apply_synonyms(tokens)
    # We care about canonical replacements
    joined = " ".join(expanded)
    assert "javascript" in joined
    assert "nodejs" in joined
    assert "machine" in joined and "learning" in joined
    # project manager synonym
    assert "project" in joined and "manager" in joined


def test_lexical_tokens_for_bm25_uses_synonyms():
    text = "Looking for a JS/Node engineer with ML experience"
    tokens = lexical_tokens_for_bm25(text)
    # JS => javascript
    assert "javascript" in tokens
    # Node => nodejs (depending on exact tokenization)
    assert any(t.startswith("node") for t in tokens)
    # ML => machine learning
    assert "machine" in tokens and "learning" in tokens


def test_normalize_query_and_index_are_consistent():
    text = "Remote JS engineer role"
    q = normalize_query(text)
    idx = normalize_for_lexical_index(text)
    assert q == idx


def test_clamp_text_length():
    text = "x" * (MAX_INPUT_CHARS + 100)
    clamped = clamp_text_length(text)
    assert len(clamped) == MAX_INPUT_CHARS
