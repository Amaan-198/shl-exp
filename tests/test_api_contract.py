from fastapi.testclient import TestClient

from src.api import app, run_full_pipeline
from src.config import AssessmentItem, RecommendResponse


client = TestClient(app)


def dummy_pipeline(query: str) -> RecommendResponse:
    # Minimal deterministic fake response: 5 items
    items = []
    for i in range(5):
        items.append(
            AssessmentItem(
                url=f"https://example.com/{i}",
                name=f"Test {i}",
                description="Desc",
                duration=30,
                adaptive_support="Yes",
                remote_support="No",
                test_type=["Knowledge & Skills"],
            )
        )
    return RecommendResponse(recommended_assessments=items)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"status": "healthy"}


def test_recommend_requires_non_empty_query(monkeypatch):
    # Monkeypatch pipeline so we don't hit real models
    monkeypatch.setattr("src.api.run_full_pipeline", dummy_pipeline)

    resp = client.post("/recommend", json={"query": " "})
    assert resp.status_code == 422


def test_recommend_returns_5_to_10_items(monkeypatch):
    monkeypatch.setattr("src.api.run_full_pipeline", dummy_pipeline)

    resp = client.post("/recommend", json={"query": "software engineer role"})
    assert resp.status_code == 200
    data = resp.json()
    assert "recommended_assessments" in data
    assert 5 <= len(data["recommended_assessments"]) <= 10
