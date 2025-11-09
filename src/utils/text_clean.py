    # src/utils/text_clean.py
import re

__all__ = ["clean_query_text"]

def clean_query_text(q: str) -> str:
    """
    Flatten multi-line job descriptions into one line and
    remove redundant whitespace.

    Examples:
    >>> clean_query_text("About Recro\\n\\nLead marketing")
    'About Recro Lead marketing'
    """
    if not q:
        return ""
    q = str(q)
    # collapse multiple whitespace / newlines into a single space
    q = re.sub(r"\s+", " ", q)
    # strip leading/trailing spaces
    q = q.strip()
    return q


