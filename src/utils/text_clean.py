# src/utils/text_clean.py
from __future__ import annotations
import re

def clean_query_text(q: str, max_len: int = 20000) -> str:
    """
    Minimal, safe query normaliser used by batch CLI:
    - collapse whitespace/newlines
    - trim
    - hard cap (defensive)
    """
    q = "" if q is None else str(q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > max_len:
        q = q[:max_len]
    return q
