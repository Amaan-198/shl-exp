import numpy as np
import pandas as pd

from src.rerank import (
    Candidate,
    build_candidate_text,
    get_candidate_texts,
    rerank_candidates,
    score_with_model,
)


class DummyReranker:
    """
    Minimal stub with the same interface as CrossEncoder.predict.
    Scores are deterministic based on candidate text length.
    """

    def predict(self, pairs):
        # pairs: [(query, candidate_text), ...]
        scores = []
        for q, cand in pairs:
            # Simple, deterministic rule: longer candidate_text => higher score
            scores.append(len(cand))
        return np.array(scores, dtype="float32")


def test_build_candidate_text():
    row = pd.Series({"name": "Test A", "description": "A great assessment."})
    text = build_candidate_text(row)
    assert "Test A" in text
    assert "A great assessment." in text


def test_get_candidate_texts_from_custom_df():
    df = pd.DataFrame(
        {
            "item_id": [1, 2],
            "url": ["u1", "u2"],
            "name": ["A", "B"],
            "description": ["desc A", "desc B"],
        }
    ).set_index("item_id", drop=False)

    texts = get_candidate_texts([1, 2], catalog_df=df)
    assert texts[0].startswith("A")
    assert texts[1].startswith("B")


def test_rerank_candidates_with_dummy_model():
    # Fake catalog
    df = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "url": ["u1", "u2", "u3"],
            "name": ["Short", "Medium length", "Very very long name"],
            "description": ["d1", "d2", "d3"],
        }
    ).set_index("item_id", drop=False)

    # Fused candidates (ids already sorted but that shouldn't matter)
    fused = [(1, 0.1), (2, 0.2), (3, 0.3)]

    # Use dummy reranker so no real model is loaded
    model = DummyReranker()

    ranked = rerank_candidates(
        query_text="test query",
        fused_candidates=fused,
        cutoff=3,
        catalog_df=df,
        model=model,
    )

    # We expect item with the longest text to be ranked highest
    assert isinstance(ranked[0], Candidate)
    ranked_ids = [c.item_id for c in ranked]
    assert 3 in ranked_ids

    # Deterministic order (no ties in this setup, but list must be sorted by rerank desc)
    scores = [c.rerank_score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_score_with_model_uses_pairs_shape():
    model = DummyReranker()
    scores = score_with_model(model, "query", ["a", "abcd"])
    # Longer candidate text should have higher score
    assert scores[1] > scores[0]
