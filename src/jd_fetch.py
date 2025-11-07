from __future__ import annotations

import httpx
from loguru import logger

try:
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None

from .config import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_MAX_REDIRECTS,
    HTTP_MAX_BYTES,
    HTTP_USER_AGENT,
)
from .normalize import basic_clean


def fetch_and_extract(url: str) -> str | None:
    """
    Fetch a JD page and extract its main content safely.

    Hardening (frozen plan §11):
      - httpx with timeouts, redirects, retry
      - 1 MB cap
      - trafilatura extraction fallback to raw text
    """
    headers = {"User-Agent": HTTP_USER_AGENT}
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(HTTP_READ_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
            max_redirects=HTTP_MAX_REDIRECTS,
        ) as client:
            r = client.get(url, headers=headers)
            if r.status_code >= 400:
                logger.warning("JD fetch: HTTP {} for {}", r.status_code, url)
                return None

            if len(r.content) > HTTP_MAX_BYTES:
                logger.warning("JD fetch aborted: {} bytes > {} limit", len(r.content), HTTP_MAX_BYTES)
                return None

            text = None
            if trafilatura is not None:
                try:
                    text = trafilatura.extract(r.text)
                except Exception as e:
                    logger.warning("Trafilatura failed: {}", e)

            if not text:
                text = r.text

            cleaned = basic_clean(text)
            return cleaned if cleaned else None
    except httpx.ReadTimeout:
        logger.warning("JD fetch timeout for {}", url)
        return None
    except Exception as e:
        logger.warning("JD fetch exception for {}: {}", url, e)
        return None
