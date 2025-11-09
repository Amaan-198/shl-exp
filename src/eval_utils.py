# NEW / REPLACE
import re, urllib.parse as up

def canonical_slug(url: str) -> str:
    if not url:
        return ""
    p = up.urlparse(url.strip())
    path = (p.path or "").lower()
    if "/view/" in path:
        slug = path.split("/view/", 1)[1]
    else:
        slug = path.strip("/").split("/")[-1]
    slug = up.unquote(slug).rstrip("/")
    slug = slug.replace("–", "-").replace("—", "-")  # unify dashes
    slug = re.sub(r"[^a-z0-9()\-/]+", "-", slug)     # keep (), -, /
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug
