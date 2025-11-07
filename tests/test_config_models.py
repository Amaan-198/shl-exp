from src.config import AssessmentItem, RecommendResponse, HealthResponse


def test_assessment_item_flags_normalization():
    item = AssessmentItem(
        url="https://example.com",
        name="Test",
        description="Desc",
        duration=30,
        adaptive_support="Maybe",
        remote_support="Yes",
        test_type=["Knowledge & Skills"],
    )
    item.ensure_flags_are_literal()
    assert item.adaptive_support in {"Yes", "No"}
    assert item.remote_support in {"Yes", "No"}


def test_recommend_response_structure():
    item = AssessmentItem(
        url="https://example.com",
        name="Test",
        description="Desc",
        duration=30,
        adaptive_support="Yes",
        remote_support="No",
        test_type=["Personality & Behavior"],
    )
    resp = RecommendResponse(recommended_assessments=[item])
    assert len(resp.recommended_assessments) == 1


def test_health_response():
    health = HealthResponse(status="healthy")
    assert health.status == "healthy"
