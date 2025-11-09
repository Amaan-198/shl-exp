# src/utils/urls.py
from __future__ import annotations
from urllib.parse import urlparse
import re
from typing import Iterable, List
from .. import config

_SLUG_RE = re.compile(r"/view/([^/?#]+)")
__all__ = ["canon_url", "canon_urls"]


def canon_url(u: str) -> str:
    """
    Canonicalize an SHL URL for equality checks.

    Strategy:
    - parse the URL
    - take ONLY the last non-empty path segment (the slug)
    - rebuild as: https://shl.com/products/product-catalog/view/<slug>

    This makes the following equivalent:
    - https://shl.com/solutions/products/product-catalog/view/java-8-new
    - https://shl.com/products/product-catalog/view/java-8-new
    """
    if not u:
        return ""
    u = str(u).strip()
    if not u:
        return ""

    p = urlparse(u)

    # Last non-empty path segment = slug
    parts = [seg for seg in p.path.split("/") if seg]
    if not parts:
        return ""
    slug = parts[-1]

    # Normalise host
    host = (p.netloc or "shl.com").strip().lower()
    if host.startswith("www."):
        host = host[4:]

    return f"https://{host}/products/product-catalog/view/{slug}"


def canon_urls(urls):
    return [canon_url(u) for u in (urls or [])]


def _slug(u: str) -> str:
    if not isinstance(u, str) or not u:
        return ""
    u = u.strip().lower()
    m = _SLUG_RE.search(u)
    slug = m.group(1) if m else u
    slug = slug.rstrip("/").replace("%28", "(").replace("%29", ")").replace("_", "-")
    return config.family_slug(slug)


def canon_urls(urls: Iterable[str]) -> List[str]:
    """
    - Normalise by family slug so aliases collapse (…-new, (v2), …).
    - Preserve first-seen order.
    """
    seen_slugs = set()
    out: List[str] = []
    for u in urls:
        s = _slug(u)
        if not s or s in seen_slugs:
            continue
        seen_slugs.add(s)
        out.append(u)
    return out
