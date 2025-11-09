# src/utils/urls.py
from urllib.parse import urlparse

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


