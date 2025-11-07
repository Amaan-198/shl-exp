import pandas as pd
from src.mapping import to_api_item, map_items_to_response
from src.config import RecommendResponse


def test_to_api_item_strict_schema():
    row = pd.Series(
        {
            "url": "https://example.com/test",
            "name": "Assessment A",
            "description": "Great test",
            "duration": "30",
            "adaptive_support": "Yes",
            "remote_support": "no",
            "test_type": ["Knowledge & Skills"],
        }
    )

    item = to_api_item(row)
    assert item.url.startswith("http")
    assert isinstance(item.duration, int)
    assert item.adaptive_support in {"Yes", "No"}
    assert isinstance(item.test_type, list)
    assert item.test_type[0] == "Knowledge & Skills"


def test_map_items_to_response_structure(tmp_path):
    df = pd.DataFrame(
        {
            "item_id": [1, 2],
            "url": ["https://a", "https://b"],
            "name": ["A", "B"],
            "description": ["desc A", "desc B"],
            "duration": [10, 20],
            "adaptive_support": ["Yes", "No"],
            "remote_support": ["No", "Yes"],
            "test_type": [["K"], ["P"]],
        }
    )
    resp = map_items_to_response([1, 2], catalog_df=df)
    assert isinstance(resp, RecommendResponse)
    assert len(resp.recommended_assessments) == 2
    assert all(item.url.startswith("https") for item in resp.recommended_assessments)
