from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import urljoin

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from .config import (
    DATA_DIR,
    HTTP_CONNECT_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_MAX_REDIRECTS,
    HTTP_MAX_BYTES,
    HTTP_USER_AGENT,
)
from .normalize import basic_clean
from .jd_fetch import fetch_and_extract

BASE_CATALOG_URL = "https://www.shl.com/products/product-catalog/"
BASE_SHL_URL = "https://www.shl.com"

LEGEND_MAP: Dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _http_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": HTTP_USER_AGENT},
        follow_redirects=True,
        timeout=httpx.Timeout(HTTP_READ_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
        max_redirects=HTTP_MAX_REDIRECTS,
    )


def _fetch_html(client: httpx.Client, url: str) -> str:
    logger.info("Fetching catalog page: {}", url)
    r = client.get(url)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    if len(r.content) > HTTP_MAX_BYTES:
        raise RuntimeError(f"Page too large ({len(r.content)} bytes) for {url}")
    return r.text


def _parse_individual_table(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")

    target_table = None
    for tbl in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in tbl.find_all("th")]
        joined = " ".join(headers)
        if "Individual Test Solutions" in joined:
            target_table = tbl
            break

    if target_table is None:
        logger.warning("No 'Individual Test Solutions' table found on page")
        return []

    body = target_table.find("tbody") or target_table

    rows: List[Dict] = []
    for tr in body.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        link = tds[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "")
        url = urljoin(BASE_SHL_URL, href)

        remote_text = tds[1].get_text(" ", strip=True) if len(tds) > 1 else ""
        adaptive_text = tds[2].get_text(" ", strip=True) if len(tds) > 2 else ""
        type_text = tds[-1].get_text(" ", strip=True) if len(tds) >= 1 else ""

        rows.append(
            {
                "name": name,
                "url": url,
                "remote_support_raw": remote_text,
                "adaptive_support_raw": adaptive_text,
                "legend_codes": type_text,
            }
        )

    logger.info("Parsed {} rows from Individual Test Solutions table", len(rows))
    return rows


def _legend_codes_to_test_types(codes: str) -> List[str]:
    parts = [c.strip() for c in codes.replace(",", " ").split() if c.strip()]
    labels: List[str] = []
    for c in parts:
        lbl = LEGEND_MAP.get(c)
        if lbl and lbl not in labels:
            labels.append(lbl)
    return labels


def _extract_duration_minutes(text: str) -> int:
    if not text:
        return 0

    candidates: list[int] = []
    lower = text.lower()

    for m in re.findall(r"minutes?\s*=\s*(\d+)", lower):
        val = int(m)
        if 1 <= val <= 240:
            candidates.append(val)

    for m in re.findall(r"(\d+)\s+minutes?\b", lower):
        val = int(m)
        if 1 <= val <= 240:
            candidates.append(val)

    return max(candidates) if candidates else 0


def _strip_boilerplate(text: str) -> str:
    # Remove duplicated "Title Description" prefix
    text = re.sub(r"^([\w\s\-\(\)&]+)\s+Description\s+\1\s+", "", text, flags=re.I)
    text = re.sub(r"^([\w\s\-\(\)&]+)\s+Description\s+", "", text, flags=re.I)
    # Remove catalog boilerplate lines
    kill = [
        r"Test Type:.*$",
        r"Remote Testing:.*$",
        r"Accelerate Your Talent Strategy.*$",
        r"-\s*[ABCDKPS]\b.*$",  # legend bullets
        r"Your use of this assessment product may be subject to .*Law 144.*$",
    ]
    for pat in kill:
        text = re.sub(pat, "", text, flags=re.I | re.M)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shorten_description(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    text = _strip_boilerplate(text)
    if len(text) > max_chars:
        trunc = text[:max_chars]
        last_space = trunc.rfind(" ")
        if last_space > 0:
            trunc = trunc[:last_space]
        text = trunc.strip()
    return text


def _enrich_with_product_details(rows: List[Dict]) -> None:
    for row in rows:
        url = row["url"]
        try:
            text = fetch_and_extract(url)
        except Exception as e:
            logger.warning("Failed to fetch product details for {}: {}", url, e)
            text = None

        if not text:
            row["description"] = ""
            row["duration"] = 0
        else:
            cleaned = basic_clean(text)
            row["description"] = _shorten_description(cleaned, max_chars=500)
            row["duration"] = _extract_duration_minutes(cleaned)


def _normalize_flags(rows: List[Dict]) -> None:
    def flag(text: str) -> str:
        t = (text or "").lower()
        if not t:
            return "No"
        if "yes" in t or t.startswith("y"):
            return "Yes"
        if "✓" in text or "check" in t:
            return "Yes"
        return "No"

    for row in rows:
        row["remote_support"] = flag(row.get("remote_support_raw", ""))
        row["adaptive_support"] = flag(row.get("adaptive_support_raw", ""))


def _build_search_text(name: str, desc: str, test_type: List[str], adaptive: str, remote: str) -> str:
    parts: List[str] = []
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    if test_type:
        parts.append(" ".join(test_type))
    flags: List[str] = []
    if adaptive == "Yes":
        flags.append("adaptive")
    if remote == "Yes":
        flags.append("remote")
    if flags:
        parts.append(" ".join(flags))
    out = ". ".join(p for p in parts if p)
    out = re.sub(r"\s+", " ", out.lower()).strip()
    return out


def crawl_individual_test_solutions() -> pd.DataFrame:
    client = _http_client()
    all_rows: List[Dict] = []
    seen_urls = set()
    page_size = 12
    start = 0

    while True:
        url = BASE_CATALOG_URL if start == 0 else f"{BASE_CATALOG_URL}?start={start}&type=1"
        try:
            html = _fetch_html(client, url)
        except Exception as e:
            logger.warning("Stopping crawl at start={} due to error: {}", start, e)
            break

        page_rows = _parse_individual_table(html)
        fresh = [r for r in page_rows if r["url"] not in seen_urls]
        for r in fresh:
            seen_urls.add(r["url"])
            all_rows.append(r)

        logger.info("Crawl page start={}, got {} rows ({} unique so far)", start, len(page_rows), len(all_rows))
        if not page_rows or len(page_rows) < page_size:
            break
        start += page_size

    if not all_rows:
        logger.warning("No catalog rows crawled from Individual Test Solutions.")
        return pd.DataFrame(columns=["item_id", "name", "url"])

    _enrich_with_product_details(all_rows)
    _normalize_flags(all_rows)
    for row in all_rows:
        row["test_type"] = _legend_codes_to_test_types(row.get("legend_codes", ""))

    for idx, row in enumerate(all_rows):
        row["item_id"] = idx
        row["search_text"] = _build_search_text(
            name=row.get("name", ""),
            desc=row.get("description", ""),
            test_type=row.get("test_type", []),
            adaptive=row.get("adaptive_support", "No"),
            remote=row.get("remote_support", "No"),
        )

    df = pd.DataFrame(all_rows)
    cols = ["item_id", "name", "url", "description", "duration", "adaptive_support", "remote_support", "test_type", "search_text"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    logger.info("Crawl complete. Final catalog rows: {}", len(df))
    return df


def build_catalog_snapshot_from_crawl() -> None:
    df = crawl_individual_test_solutions()
    output = DATA_DIR / "catalog_snapshot.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Catalog snapshot written to {}", output)


if __name__ == "__main__":
    build_catalog_snapshot_from_crawl()
