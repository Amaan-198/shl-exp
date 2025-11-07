from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger

from .config import CATALOG_RAW_DIR, CATALOG_SNAPSHOT_PATH
from .normalize import basic_clean


# ---------------------------
# Column detection / standardization
# ---------------------------

# We don't know the exact XLSX schema, so we support multiple likely variants.
COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "name": [
        "name",
        "Name",
        "Assessment Name",
        "assessment_name",
        "Assessment",
        "Title",
    ],
    "url": [
        "url",
        "URL",
        "Url",
        "Link",
        "Assessment URL",
        "assessment_url",
    ],
    "description": [
        "description",
        "Description",
        "Assessment Description",
        "Long Description",
        "Overview",
        "Summary",
    ],
    "duration_raw": [
        "duration",
        "Duration",
        "Duration (mins)",
        "Duration (Minutes)",
        "Time (minutes)",
        "time_minutes",
    ],
    "adaptive_raw": [
        "adaptive",
        "Adaptive",
        "Adaptive/IRT",
        "Adaptive / IRT",
        "Adaptive Flag",
    ],
    "remote_raw": [
        "remote",
        "Remote",
        "Remote Testing",
        "Remote flag",
        "Remote/Online",
    ],
    "test_type_raw": [
        "type",
        "Type",
        "Legend",
        "K/P",
        "Test Type",
        "test_type",
        "primary_dimension",
    ],
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from the raw catalog to a canonical internal schema.

    We keep raw columns:
    - name
    - url
    - description
    - duration_raw
    - adaptive_raw
    - remote_raw
    - test_type_raw
    """
    col_map: Dict[str, str] = {}
    lower_to_original = {c.lower(): c for c in df.columns}

    for canon, candidates in COLUMN_CANDIDATES.items():
        for candidate in candidates:
            # Try exact, then case-insensitive
            if candidate in df.columns:
                col_map[candidate] = canon
                break
            cand_lower = candidate.lower()
            if cand_lower in lower_to_original:
                col_map[lower_to_original[cand_lower]] = canon
                break

    logger.info("Standardizing columns with map: {}", col_map)

    df_std = df.rename(columns=col_map)

    # Log if key columns are missing; we still proceed but downstream may drop rows.
    required = ["name", "url", "description"]
    missing = [c for c in required if c not in df_std.columns]
    if missing:
        logger.warning("Raw catalog is missing required columns: {}", missing)

    return df_std


# ---------------------------
# Field parsing helpers
# ---------------------------

def parse_duration_to_minutes(value) -> int:
    """
    Parse a duration field into an integer number of minutes.

    Rules:
    - If it's already numeric, clamp to int >= 0.
    - If it's a string containing numbers, return the upper bound of any range.
      e.g., "20-30 minutes" -> 30
    - If no numbers are found, return 0.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0

    if isinstance(value, (int, float)):
        return max(0, int(value))

    text = str(value)
    nums = re.findall(r"\d+", text)
    if not nums:
        return 0

    try:
        ints = [int(n) for n in nums]
        return max(ints)
    except ValueError:
        return 0


def _canonicalize_yes_no(value) -> str:
    """
    Normalize a flag to literal 'Yes' or 'No'.

    Truthy patterns -> 'Yes', everything else -> 'No'.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "No"

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "y", "true", "1", "adaptive", "remote", "supported"}:
            return "Yes"
        if v in {"no", "n", "false", "0"}:
            return "No"
        # Unknown string -> default No, but log once upstream if needed.
        return "No"

    # For non-strings, fall back to bool semantics.
    return "Yes" if bool(value) else "No"


def parse_test_type_field(value) -> List[str]:
    """
    Parse the raw test_type / legend field into a list of humanized labels.

    Mapping rules:
    - 'K' or equivalents -> 'Knowledge & Skills'
    - 'P' or equivalents -> 'Personality & Behavior'
    - Already-humanized strings are normalized to the canonical forms.
    - Other tokens are kept as-is (trimmed).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    tokens: List[str]

    if isinstance(value, list):
        tokens = [str(v) for v in value]
    else:
        text = str(value)
        # Split on common separators
        parts = re.split(r"[;,/|]+", text)
        tokens = [p.strip() for p in parts if p.strip()]

    labels: List[str] = []
    for tok in tokens:
        upper = tok.upper()
        lowered = tok.strip().lower()

        if upper == "K":
            label = "Knowledge & Skills"
        elif upper == "P":
            label = "Personality & Behavior"
        elif lowered in {"knowledge & skills", "knowledge and skills"}:
            label = "Knowledge & Skills"
        elif lowered in {"personality & behavior", "personality and behavior"}:
            label = "Personality & Behavior"
        else:
            label = tok.strip()

        if label and label not in labels:
            labels.append(label)

    return labels


def build_search_text(
    name: str,
    description: str,
    test_type: Iterable[str],
    adaptive_support: str,
    remote_support: str,
) -> str:
    """
    Build the search_text field:

    name + ". " + description + ". " + human test_type words + flag hints ('adaptive', 'remote')

    Final text is lowercased and whitespace-normalized.
    """
    name = (name or "").strip()
    description = (description or "").strip()
    test_words = " ".join(test_type) if test_type else ""

    flag_bits: List[str] = []
    if adaptive_support == "Yes":
        flag_bits.append("adaptive")
    if remote_support == "Yes":
        flag_bits.append("remote")

    parts: List[str] = []
    if name:
        parts.append(name)
    if description:
        parts.append(description)
    if test_words:
        parts.append(test_words)
    if flag_bits:
        parts.append(" ".join(flag_bits))

    # Join with ". " between logical segments.
    combined = ". ".join(parts)
    combined = combined.strip().lower()
    combined = re.sub(r"\s+", " ", combined)
    return combined


# ---------------------------
# Catalog normalization
# ---------------------------

def normalize_catalog_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Main normalization pipeline for the SHL catalog.

    Input: raw DataFrame with unknown column names.
    Output: canonical schema:

    - item_id (int)
    - url (str)
    - name (str)
    - description (str)
    - duration (int minutes, >=0)
    - adaptive_support ("Yes"/"No")
    - remote_support ("Yes"/"No")
    - test_type (List[str])
    - search_text (str; lowercased)
    """
    logger.info("Normalizing catalog dataframe with {} raw rows", len(df_raw))

    df = _standardize_columns(df_raw.copy())

    # Drop rows with empty URL
    if "url" not in df.columns:
        logger.error("No URL column found after standardization; resulting catalog will be empty.")
        return pd.DataFrame(
            columns=[
                "item_id",
                "url",
                "name",
                "description",
                "duration",
                "adaptive_support",
                "remote_support",
                "test_type",
                "search_text",
            ]
        )

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # Basic fields
    df["name"] = df.get("name", "").fillna("").astype(str)
    df["description"] = df.get("description", "").fillna("").astype(str)

    # Clean name/description using shared normalizer (preserve casing for UI)
    df["name_clean"] = df["name"].apply(basic_clean)
    df["description_clean"] = df["description"].apply(basic_clean)

    # Duration
    if "duration_raw" in df.columns:
        df["duration"] = df["duration_raw"].apply(parse_duration_to_minutes)
    else:
        df["duration"] = 0

    # Flags
    df["adaptive_support"] = df.get("adaptive_raw", "No").apply(_canonicalize_yes_no)
    df["remote_support"] = df.get("remote_raw", "No").apply(_canonicalize_yes_no)

    # Test type
    if "test_type_raw" in df.columns:
        df["test_type"] = df["test_type_raw"].apply(parse_test_type_field)
    else:
        df["test_type"] = [[] for _ in range(len(df))]

    # Build search_text
    df["search_text"] = df.apply(
        lambda row: build_search_text(
            name=row["name_clean"],
            description=row["description_clean"],
            test_type=row["test_type"],
            adaptive_support=row["adaptive_support"],
            remote_support=row["remote_support"],
        ),
        axis=1,
    )

    # Final canonical frame
    df_out = df[
        [
            "url",
            "name_clean",
            "description_clean",
            "duration",
            "adaptive_support",
            "remote_support",
            "test_type",
            "search_text",
        ]
    ].rename(
        columns={
            "name_clean": "name",
            "description_clean": "description",
        }
    )

    df_out.insert(0, "item_id", range(len(df_out)))

    logger.info("Catalog normalization complete. Final rows: {}", len(df_out))

    return df_out


# ---------------------------
# IO helpers
# ---------------------------

def load_raw_catalog(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw catalog data from an Excel file.

    If no path is provided, we take the first .xlsx found under data/catalog_raw.
    """
    if path is None:
        candidates = sorted(CATALOG_RAW_DIR.glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError(
                f"No .xlsx files found under {CATALOG_RAW_DIR}. "
                f"Place the SHL catalog there and re-run."
            )
        path = candidates[0]

    logger.info("Loading raw catalog from {}", path)
    df = pd.read_excel(path)
    logger.info("Loaded {} rows from raw catalog", len(df))
    return df


def build_catalog_snapshot(
    raw_path: Optional[Path] = None,
    output_path: Path = CATALOG_SNAPSHOT_PATH,
) -> Path:
    """
    End-to-end: load raw catalog → normalize → write Parquet snapshot.

    Returns the output path.
    """
    df_raw = load_raw_catalog(raw_path)
    df_norm = normalize_catalog_df(df_raw)

    logger.info("Writing catalog snapshot to {}", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_parquet(output_path, index=False)
    logger.info("Catalog snapshot written with {} rows", len(df_norm))

    return output_path


def load_catalog_snapshot(path: Path = CATALOG_SNAPSHOT_PATH) -> pd.DataFrame:
    """
    Convenience helper to load the normalized catalog snapshot.
    """
    logger.info("Loading catalog snapshot from {}", path)
    df = pd.read_parquet(path)
    logger.info("Loaded catalog snapshot with {} rows", len(df))
    return df


# ---------------------------
# CLI entrypoint
# ---------------------------

if __name__ == "__main__":
    # Simple manual way to build the snapshot:
    # python -m src.catalog_build
    build_catalog_snapshot()
