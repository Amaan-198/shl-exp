import numpy as np

from src.mmr import mmr_select


def test_mmr_balances_relevance_and_diversity():
    # 4 fake embeddings: 0/1 identical, 2/3 far apart
    emb = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.0],
            [0.0, 1.0],
            [0.0, 0.99],
        ],
        dtype="float32",
    )
    ids = [10, 11, 12, 13]

    # Relevance scores high for 0/1, lower for 2/3
    candidates = [(10, 1.0), (11, 0.9), (12, 0.6), (13, 0.5)]

    # Lambda=0.7 should pick one of 0/1, then a diverse one (2 or 3)
    selected = mmr_select(candidates, emb, ids, k=3, lambda_=0.7)

    assert len(selected) == 3
    # First must be top relevance (10)
    assert selected[0] == 10
    # The next chosen should not be its near-duplicate 11
    assert selected[1] in {12, 13}


