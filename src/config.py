from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


# ---------------------------
# Paths
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CATALOG_RAW_DIR = DATA_DIR / "catalog_raw"
CATALOG_SNAPSHOT_PATH = DATA_DIR / "catalog_snapshot.parquet"
TRAIN_PATH = DATA_DIR / "gen_ai_train.xlsx"
TEST_PATH = DATA_DIR / "gen_ai_test.xlsx"

INDICES_DIR = PROJECT_ROOT / "indices"
BM25_INDEX_PATH = INDICES_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDICES_DIR / "faiss.index"
EMBEDDINGS_PATH = INDICES_DIR / "item_embeddings.npy"
IDS_MAPPING_PATH = INDICES_DIR / "ids.json"

MODELS_DIR = PROJECT_ROOT / "models"  # for HF cache if you want to mount it


# ---------------------------
# Model names (pinned)
# ---------------------------

# Dense encoder
BGE_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"

# Cross-encoder reranker
BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Zero-shot intent classifier
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"  # bart-mnli; name as in HF hub


# ---------------------------
# Retrieval & fusion settings
# ---------------------------

BM25_TOP_N = 200          # per retriever
DENSE_TOP_N = 200
FUSION_TOP_K = 60         # after fusion

BM25_WEIGHT = 0.60
DENSE_WEIGHT = 0.40

FUSION_WINSORIZE_MIN = -3.0
FUSION_WINSORIZE_MAX = 3.0
FUSION_EPS = 1e-8

# MMR diversification
MMR_LAMBDA = 0.60         # 70% relevance / 30% diversity


# ---------------------------
# Result size policy
# ---------------------------

RESULT_MIN = 5
RESULT_MAX = 10
RESULT_DEFAULT_TARGET = 10  # aim for 10 when possible, but never < 5


# ---------------------------
# Rerank settings & env toggles
# ---------------------------

DEFAULT_RERANK_CUTOFF = 60
RERANK_CUTOFF = int(os.getenv("RERANK_CUTOFF", str(DEFAULT_RERANK_CUTOFF)))

# HF cache / offline mode (we don't set env vars here; just define names)
HF_ENV_VARS = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TRANSFORMERS_CACHE": str(MODELS_DIR),
    # HF_HUB_OFFLINE to be optionally set to "1" by the runtime after first pull
}


# ---------------------------
# Text processing
# ---------------------------

MAX_INPUT_CHARS = 20_000  # input size cap
ENCODER_TOKEN_BUDGET = 512

# Chunking for long JDs (dense side only)
CHUNK_SIZE_TOKENS = 220
CHUNK_STRIDE_TOKENS = 110
CHUNK_TOP_K = 3  # keep top-3 chunk vectors and average


# Small, deterministic synonym map
SYNONYM_MAP: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "node.js": "nodejs",
    "asp.net": "aspnet",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "pm": "project manager",
    "stakeholder mgmt": "stakeholder management",
}

# Optional ESCO augment flag (we can stub this at first)
ENABLE_ESCO_AUGMENT = False


# ---------------------------
# JD fetch / HTTP hardening
# ---------------------------

HTTP_CONNECT_TIMEOUT = 3.0
HTTP_READ_TIMEOUT = 7.0
HTTP_MAX_REDIRECTS = 2
HTTP_MAX_BYTES = 1_000_000  # 1 MB cap

HTTP_USER_AGENT = (
    "shl-rag-recommender/1.0 (+https://example.com; contact=genai@placeholder.com)"
)

# --- Zero-shot smoothing (balance calibration) ---
INTENT_TEMP = 1.5           # temperature for softmax
INTENT_SMOOTH_EPS = 0.15    # mix with uniform prior
INTENT_CLIP_MIN = 0.20      # floor after smoothing
INTENT_CLIP_MAX = 0.80      # ceiling after smoothing

# ---------------------------
# K/P balance policy
# ---------------------------

INTENT_LABEL_TECHNICAL = "technical skills / knowledge"
INTENT_LABEL_PERSONALITY = "personality / behavior"
INTENT_LABELS: List[str] = [INTENT_LABEL_TECHNICAL, INTENT_LABEL_PERSONALITY]

# --- Soft bonus for items whose test_type matches dominant intent ---
INTENT_SOFT_BONUS = 0.06

# --- Crawl controls ---
MAX_CATALOG_PAGES = 100
HTTP_RETRY_READS = 1

# thresholds from the plan
BOTH_HIGH_THRESHOLD = 0.45  # pt >= 0.45 and pb >= 0.45 -> 5/5
DOMINANT_THRESHOLD = 0.60   # max(pt, pb) >= 0.60 and other >= 0.30 -> 7/3
SECONDARY_MIN_FOR_SPLIT = 0.30

BALANCE_5_5_SIZE = 10
BALANCE_7_3_SIZE = 10


# ---------------------------
# Logging / observability
# ---------------------------

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------
# Pydantic models shared around the app
# ---------------------------

class AssessmentItem(BaseModel):
    """
    Canonical schema for a single recommended assessment.
    This matches the API contract exactly.
    """

    url: str
    name: str
    description: str
    duration: int = Field(ge=0)
    adaptive_support: str  # "Yes"/"No"
    remote_support: str    # "Yes"/"No"
    test_type: List[str]

    def ensure_flags_are_literal(self) -> None:
        """
        Safety check: normalize adaptive/remote flags strictly to 'Yes'/'No'.
        """
        self.adaptive_support = "Yes" if self.adaptive_support == "Yes" else "No"
        self.remote_support = "Yes" if self.remote_support == "Yes" else "No"


class RecommendResponse(BaseModel):
    """
    Response body for POST /recommend.
    """

    recommended_assessments: List[AssessmentItem]


class HealthResponse(BaseModel):
    """
    Response body for GET /health.
    """

    status: str
