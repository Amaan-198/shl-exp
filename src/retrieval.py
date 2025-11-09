from __future__ import annotations
"""
Retrieval module for the SHL recommender.

Hybrid retrieval = BM25 + dense (BGE+FAISS) with score fusion.
This version adds:
- cached loaders (no per-query reloads)
- intent-based light BM25 expansion
- soft domain overlap gating (penalty/boost)
- seed boosts for hard domains (guarantees at least a few right-family items)
- soft duration handling (no hard filter)
"""

import math
import pickle
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np

from . import config
from .text_utils import extract_time_budget_mins
from .intent import detect_intents, expansion_seeds
from .catalog_build import load_catalog_snapshot
from .embed_index import (
    load_bm25 as load_bm25_cache,
    load_dense_components,
    embed_query_with_chunking,
)

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback logger
    import logging
    logging.basicConfig(level=logging.INFO)
    class _FallbackLogger:
        def __init__(self, logger):
            self._logger = logger
        def info(self, msg: str, *args, **kwargs) -> None: self._logger.info(msg.format(*args))
        def warning(self, msg: str, *args, **kwargs) -> None: self._logger.warning(msg.format(*args))
        def error(self, msg: str, *args, **kwargs) -> None: self._logger.error(msg.format(*args))
        def exception(self, msg: str, *args, **kwargs) -> None: self._logger.exception(msg.format(*args))
    logger = _FallbackLogger(logging.getLogger(__name__))

from .config import (
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    IDS_MAPPING_PATH,
    FUSION_TOP_K,
    FUSION_WINSORIZE_MIN,
    FUSION_WINSORIZE_MAX,
    FUSION_EPS,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    BM25_TOP_N,
    DENSE_TOP_N,
    EXPANSION_LIBRARY,
    RETRIEVAL_BOOST_SEEDS,
)

# Optional pandas (not required)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# =============================================================================
# Query expansion (intent → BM25 only)
# =============================================================================

# --- in src/retrieval.py ---

from . import config

_SALES_TRIGGERS = {"sales", "account executive", "bd", "business development", "associate"}
_COMMS_TRIGGERS = {"communication", "spoken english", "english", "email writing", "presentation"}
_DATA_TRIGGERS  = {"analyst", "analytics", "excel", "tableau", "sql", "power bi"}
_QA_TRIGGERS    = {"qa", "testing", "selenium", "manual testing", "automation"}
_JAVA_TRIGGERS  = {"java", "spring", "spring boot"}
_BEHAV_TRIGGERS = {"leadership", "opq", "personality", "behavior", "behaviour"}

def _expand_query_with_intents(raw_query: str) -> str:
    """
    Light intent-based expansion + deterministic SHL seeds.

    - Keep your existing intent detector.
    - Add explicit expansions when clear domain triggers are present.
    - Triple-duplicate exact 'must-have' literals to push BM25.
    """
    from . import config
    base = raw_query.strip()
    expanded_bits: List[str] = []

    # 1) Zero-shot / heuristic intent → soft seeds
    try:
        scores = detect_intents(base)  # your existing detector
        seeds = expansion_seeds(scores)  # your existing mapping
        for term, w in seeds.items():
            expanded_bits.extend([term] * max(1, int(round(w * 2))))
    except Exception:
        pass

    # 2) Deterministic expansions mapped to gold-leaning SHL families
    low = base.lower()

    def add_seed(key: str, times: int = 1):
        for _ in range(times):
            expanded_bits.extend(config.EXPANSION_LIBRARY.get(key, []))

    # Sales / communication
    if any(t in low for t in ("sales","customer service","account executive","bd","business development")):
        add_seed("sales_entry", times=2)
        add_seed("behavior", times=1)
    if any(t in low for t in tuple(config.COMMUNICATION_TRIGGER_PHRASES)):
        add_seed("behavior", times=2)
        add_seed("sales_entry", times=1)

    # Data / analytics / python
    if any(t in low for t in ("data analyst","analytics","excel","power bi","tableau","sql","python","pandas","numpy")):
        add_seed("data_analyst", times=2)

    # QA / testing
    if any(t in low for t in ("qa","quality assurance","selenium","testing","test case","test plan","regression")):
        add_seed("qa_testing", times=2)

    # NEW: marketing / manager / leadership
    if "marketing" in low:
        add_seed("marketing_manager", times=3)
    if any(t in low for t in ("manager","leadership","team lead","supervisor","head of")):
        add_seed("leadership_manager", times=3)

    # 3) Must-have literal boosts (tripled) to bias BM25 properly
    MUSTS = []
    if "opq" in low or "personality" in low:
        MUSTS += ["opq","occupational personality questionnaire"]
    if "marketing" in low:
        MUSTS += ["digital advertising","business communication adaptive","interpersonal communications"]
    if "excel" in low:
        MUSTS += ["microsoft excel 365 new","microsoft excel 365 essentials new"]
    if "manager" in low or "leadership" in low:
        MUSTS += ["manager 8.0","enterprise leadership"]

    expanded_bits.extend([m for m in MUSTS for _ in range(3)])

    # Final expanded query string
    expanded = base + " " + " ".join(expanded_bits)
    return expanded.strip()


# =============================================================================
# Duration (soft preference)
# =============================================================================

def _duration_ok(item_duration_min: Optional[int], budget_min: Optional[int]) -> bool:
    if budget_min is None or item_duration_min is None:
        return True
    return item_duration_min <= (budget_min + config.DURATION_TOLERANCE_MIN)

def _duration_penalty(item_duration_min: Optional[int], budget_min: Optional[int]) -> float:
    if budget_min is None or item_duration_min is None:
        return 0.0
    if item_duration_min <= budget_min + config.DURATION_TOLERANCE_MIN:
        return 0.0
    ratio = (item_duration_min - budget_min) / max(budget_min, 1)
    return min(0.30, 0.30 * ratio)

def _apply_duration_postfilter(
    scored_items: List[Tuple[int, float]],
    budget_min: Optional[int],
    durations: Optional[Dict[int, int]],
    k: int,
) -> List[Tuple[int, float]]:
    if budget_min is None or not scored_items or not durations:
        return scored_items[:k]
    within, over = [], []
    for iid, sc in scored_items:
        dur = int(durations.get(iid)) if (durations and iid in durations) else None
        if _duration_ok(dur, budget_min):
            within.append((iid, sc))
        else:
            over.append((iid, sc * (1.0 - _duration_penalty(dur, budget_min))))
    over.sort(key=lambda x: x[1], reverse=True)
    return (within + over)[:k]


# =============================================================================
# Score normalization
# =============================================================================

def _winsorize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)

def _to_z(arr: np.ndarray, eps: float = FUSION_EPS) -> np.ndarray:
    if arr.size == 0:
        return arr.astype("float32")
    m, s = arr.mean(), arr.std()
    if s < eps: s = eps
    return (arr - m) / s

def _normalize_slice(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    ids, scores = zip(*pairs)
    scores_arr = np.asarray(scores, dtype="float32")
    scores_arr = _winsorize(scores_arr, FUSION_WINSORIZE_MIN, FUSION_WINSORIZE_MAX)
    scores_arr = _to_z(scores_arr)
    return {i: float(s) for i, s in zip(ids, scores_arr)}


# =============================================================================
# Domain gating (soft) + seed boosts
# =============================================================================

_NO_OVERLAP_PENALTY: float = 0.18   # subtract if candidate has zero marker overlap
_MUST_HAVE_BOOST: float   = 0.18    # add if candidate contains must-have literal
_SEED_FUSED_BONUS: float  = 0.25    # small deterministic push for seed-set items

# token groups (low ambiguity)
_DOMAIN_MARKERS: Dict[str, set[str]] = {
    "python_backend": {"python","django","flask","fastapi","api","backend",
                       "data_structures","algorithms","system_design"},
    "ml_deployment": {"machine_learning","deep_learning","neural_network","neural_networks",
                      "mlops","model_deployment","mlflow","onnx","docker","kubernetes",
                      "pytorch","tensorflow"},
    "business_analyst": {"excel","power_bi","tableau","data_visualization","storytelling",
                         "attention_to_detail","pivot_tables"},
    "hr_leadership": {"leadership","employee_engagement","conflict_management",
                      "interpersonal","people_management","opq"},
    "sales_entry": {"sales","negotiation","multitasking","communication",
                    "customer_service","persuasion"},
    "qa_testing": {"qa","testing","test_automation","selenium","cypress",
                   "test_cases","defect","regression"},
    "cloud_devops": {"aws","azure","gcp","kubernetes","docker","terraform",
                     "ci_cd","jenkins","sre"},
}

# lookups cached after first call
_ID_TO_TEXT: Optional[Dict[int, str]] = None
_ID_TO_DURATION: Optional[Dict[int, int]] = None
_ID_TO_URL: Optional[Dict[int, str]] = None
_CURRENT_QUERY_CLEANED: str = ""

def _ensure_catalog_lookup() -> Tuple[Optional[Dict[int, str]], Optional[Dict[int, int]], Optional[Dict[int, str]]]:
    """Build lookups from catalog snapshot for domain gating / durations / URL slug mapping."""
    global _ID_TO_TEXT, _ID_TO_DURATION, _ID_TO_URL
    if _ID_TO_TEXT is not None:
        return _ID_TO_TEXT, _ID_TO_DURATION, _ID_TO_URL
    try:
        df = load_catalog_snapshot()
        id_col = "item_id" if "item_id" in df.columns else ("id" if "id" in df.columns else None)
        if not id_col: return None, None, None
        txt_col = "search_text" if "search_text" in df.columns else None
        dur_col = "duration" if "duration" in df.columns else None
        url_col = "url" if "url" in df.columns else None

        if txt_col:
            pairs = df[[id_col, txt_col]].astype({id_col:"int64"}).itertuples(index=False, name=None)
            _ID_TO_TEXT = {int(i): str(t).lower() for i, t in pairs}
        else:
            _ID_TO_TEXT = {}

        if dur_col:
            dur_pairs = df[[id_col, dur_col]].dropna().astype({id_col:"int64"}).itertuples(index=False, name=None)
            _ID_TO_DURATION = {int(i): int(d) for i, d in dur_pairs}
        else:
            _ID_TO_DURATION = {}

        if url_col:
            url_pairs = df[[id_col, url_col]].astype({id_col:"int64"}).itertuples(index=False, name=None)
            _ID_TO_URL = {int(i): str(u) for i, u in url_pairs}
        else:
            _ID_TO_URL = {}

        logger.info("Catalog lookup initialised ({} items)", len(_ID_TO_TEXT or {}))
        return _ID_TO_TEXT, _ID_TO_DURATION, _ID_TO_URL
    except Exception as e:
        logger.warning("Failed to init catalog lookup: {}", e)
        return None, None, None

def _extract_markers(cleaned_query: str) -> tuple[set[str], set[str]]:
    q = f" {cleaned_query} "
    markers: set[str] = set()
    musts: set[str] = set()
    for toks in _DOMAIN_MARKERS.values():
        for t in toks:
            if f" {t} " in q:
                markers.add(t); musts.add(t)
    return markers, musts

def _apply_domain_adjustments(fused_list: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    if not fused_list or not _CURRENT_QUERY_CLEANED:
        return fused_list
    id2text, _, _ = _ensure_catalog_lookup()
    if id2text is None:
        return fused_list
    markers, musts = _extract_markers(_CURRENT_QUERY_CLEANED)
    if not markers:
        return fused_list
    adjusted: List[Tuple[int, float]] = []
    for iid, score in fused_list:
        st = f" {id2text.get(iid, '')} "
        overlap = sum(1 for m in markers if f" {m} " in st)
        if overlap == 0:
            score -= _NO_OVERLAP_PENALTY
        if any(f" {m} " in st for m in musts):
            score += _MUST_HAVE_BOOST
        adjusted.append((iid, score))
    adjusted.sort(key=lambda x: (-x[1], x[0]))
    return adjusted

def _family_slug(s: str) -> str:
    s = s.strip().lower().strip("/")
    s = s.replace("%28","(").replace("%29",")").replace("_","-")
    s = re.sub(r"-+", "-", s)
    for pat in (r"-new$", r"\(new\)$", r"-v\d+$", r"\(\s*v\d+\s*\)$", r"-\d+\.\d+$"):
        s = re.sub(pat, "", s).rstrip("-").strip()
    return s

def _seed_boost_ids(cleaned_query: str) -> List[int]:
    """Map deterministic seed slugs → item_ids based on the current catalog."""
    id2text, _, id2url = _ensure_catalog_lookup()
    if id2url is None:
        return []
    q = cleaned_query.lower()
    seeds: List[str] = []

    # Heuristic triggers → expansion library
    if any(k in q for k in ["freshers", "entry level", "sales "]):
        seeds += EXPANSION_LIBRARY.get("sales_entry", [])
    if any(k in q for k in ["qa ", "quality assurance", "selenium", "testing"]):
        seeds += EXPANSION_LIBRARY.get("qa_testing", [])
    if any(k in q for k in ["content", "seo", "writer", "marketing manager"]):
        seeds += EXPANSION_LIBRARY.get("content_marketing", [])
    if any(k in q for k in ["assistant admin", "back office", "admin "]):
        seeds += EXPANSION_LIBRARY.get("admin_ops", [])
    if any(k in q for k in ["coo", "cxo", "executive", "culture fit", "leadership", "opq"]):
        seeds += RETRIEVAL_BOOST_SEEDS.get("leadership", []) + RETRIEVAL_BOOST_SEEDS.get("communication", [])

    slugset = { _family_slug(s.replace(" ", "-")) for s in seeds if s }
    hits: List[int] = []
    for iid, url in id2url.items():
        slug = url.split("/view/")[-1] if "/view/" in url else url.split("/")[-1]
        fam = _family_slug(slug)
        if fam in slugset:
            hits.append(int(iid))
    return hits


# =============================================================================
# Fusion
# =============================================================================

def fuse_scores(
    bm25: List[Tuple[int, float]],
    dense: List[Tuple[int, float]],
    top_k: int = FUSION_TOP_K,
) -> List[Tuple[int, float]]:
    nz_bm25 = _normalize_slice(bm25)
    nz_dense = _normalize_slice(dense)
    all_ids = set(nz_bm25) | set(nz_dense)

    fused: List[Tuple[int, float]] = []
    for i in all_ids:
        s = BM25_WEIGHT * nz_bm25.get(i, 0.0) + DENSE_WEIGHT * nz_dense.get(i, 0.0)
        fused.append((i, s))
    fused.sort(key=lambda x: (-x[1], x[0]))

    # Domain soft boost/penalty
    fused = _apply_domain_adjustments(fused)

    # Seed boosts after basic fusion (deterministic small lift)
    seeds = set(_seed_boost_ids(_CURRENT_QUERY_CLEANED))
    if seeds:
        boosted = []
        for iid, sc in fused:
            sc2 = sc + (_SEED_FUSED_BONUS if iid in seeds else 0.0)
            boosted.append((iid, sc2))
        boosted.sort(key=lambda x: (-x[1], x[0]))
        fused = boosted

    return fused[:top_k]


# =============================================================================
# Cached loaders
# =============================================================================

@lru_cache(maxsize=1)
def _get_bm25():
    return load_bm25_cache(BM25_INDEX_PATH)

@lru_cache(maxsize=1)
def _get_dense():
    # returns (model, faiss_index, id_map, dense_stale)
    return load_dense_components(FAISS_INDEX_PATH, IDS_MAPPING_PATH)

@lru_cache(maxsize=1)
def _get_catalog_df():
    return load_catalog_snapshot()


# =============================================================================
# Retrieval primitives
# =============================================================================

def retrieve_bm25(cleaned_query: str) -> List[Tuple[int, float]]:
    bm25 = _get_bm25()
    results = bm25.query(cleaned_query)
    return results[:BM25_TOP_N] if results else []

def retrieve_dense(model, faiss_index, id_map, query_vec) -> List[Tuple[int, float]]:
    D, I = faiss_index.search(query_vec[np.newaxis, :], DENSE_TOP_N)
    out: List[Tuple[int, float]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: continue
        item_id = id_map[str(idx)]
        out.append((int(item_id), float(dist)))
    return out


# =============================================================================
# Public API
# =============================================================================

def retrieve_candidates(raw_query: str) -> Tuple[str, str, List[Tuple[int, float]]]:
    """
    Returns: (raw_query, cleaned_query, [(item_id, fused_score), ...])
    """
    from .normalize import normalize_for_lexical_index as clean_text
    global _CURRENT_QUERY_CLEANED

    budget_min = extract_time_budget_mins(raw_query)
    cleaned_query = clean_text(raw_query)
    _CURRENT_QUERY_CLEANED = cleaned_query

    expanded_for_bm25 = _expand_query_with_intents(cleaned_query)

    # cached components
    bm25 = _get_bm25()
    model, faiss_index, id_map, dense_stale = _get_dense()

    # BM25
    bm25_results = bm25.query(expanded_for_bm25) if bm25 else []
    if bm25_results:
        bm25_results = bm25_results[:BM25_TOP_N]

    # Dense
    dense_results: List[Tuple[int, float]] = []
    if not dense_stale and model is not None and faiss_index is not None and id_map:
        try:
            query_vec = embed_query_with_chunking(model, cleaned_query)
            dense_results = retrieve_dense(model, faiss_index, id_map, query_vec)
        except Exception as e:  # pragma: no cover
            logger.warning("Dense retrieval failed; BM25 only: {}", e)

    fused = fuse_scores(bm25_results, dense_results)  # → top FUSION_TOP_K

    # Duration-aware gentle reorder within top-K
    _, duration_lookup, _ = _ensure_catalog_lookup()
    fused = _apply_duration_postfilter(fused, budget_min, duration_lookup, FUSION_TOP_K)

    logger.info(
        "retrieve_candidates: raw='{}' cleaned='{}' bm25N={} denseN={} -> {} fused",
        raw_query, cleaned_query, len(bm25_results), len(dense_results), len(fused),
    )
    return raw_query, cleaned_query, fused
