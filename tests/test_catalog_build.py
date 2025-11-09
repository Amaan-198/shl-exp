import pandas as pd

from src.catalog_build import (
    build_search_text,
    normalize_catalog_df,
    parse_duration_to_minutes,
    parse_test_type_field,
)


def test_parse_duration_to_minutes():
    assert parse_duration_to_minutes("20-30 minutes") == 30
    assert parse_duration_to_minutes("45 min") == 45
    assert parse_duration_to_minutes(60) == 60
    assert parse_duration_to_minutes("no time") == 0


def test_parse_test_type_field():
    assert parse_test_type_field("K") == ["Knowledge & Skills"]
    assert parse_test_type_field("P") == ["Personality & Behavior"]

    labels = parse_test_type_field("K, P")
    assert "Knowledge & Skills" in labels
    assert "Personality & Behavior" in labels


def test_build_search_text_includes_flags_and_types():
    text = build_search_text(
        name="Assessment A",
        description="A great test.",
        test_type=["Knowledge & Skills", "Personality & Behavior"],
        adaptive_support="Yes",
        remote_support="No",
    )
    # All lowercased
    assert "assessment a" in text
    assert "a great test" in text
    assert "knowledge & skills" in text
    assert "personality & behavior" in text
    assert "adaptive" in text
    assert "remote" not in text  # because remote_support == "No"


def test_normalize_catalog_basic():
    raw = pd.DataFrame(
        {
            "name": ["Assessment A"],
            "url": ["https://example.com/a"],
            "description": ["A <b>great</b> test!"],
            "duration": ["20-30 minutes"],
            "adaptive": ["Yes"],
            "remote": ["no"],
            "type": ["K,P"],
        }
    )

    df = normalize_catalog_df(raw)

    # Schema check
    expected_cols = [
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
    assert list(df.columns) == expected_cols

    row = df.iloc[0]

    # Duration upper bound
    assert row["duration"] == 30

    # Flags normalized
    assert row["adaptive_support"] == "Yes"
    assert row["remote_support"] == "No"

    # Test type mapping
    assert "Knowledge & Skills" in row["test_type"]
    assert "Personality & Behavior" in row["test_type"]

    # HTML stripped in description
    assert row["description"] == "A great test!"

    # Search text contains adaptive but not remote
    assert "adaptive" in row["search_text"]
    assert "remote" not in row["search_text"]


