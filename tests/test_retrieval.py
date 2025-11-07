import numpy as np

from src.retrieval import fuse_scores, _winsorize


def test_winsorize_clips_extremes():
    arr = np.array([-10, -2, 0, 2, 10])
    clipped = _winsorize(arr, -3, 3)
    assert clipped.min() >= -3
    assert clipped.max() <= 3


def test_fuse_scores_weighted_zscore_behavior():
    bm25 = [(1, 1.0), (2, 2.0), (3, 3.0)]
    dense = [(2, 3.0), (3, 1.0), (4, 2.0)]

    fused = fuse_scores(bm25, dense, top_k=4)

    # Should produce all ids 1–4
    ids = [i for i, _ in fused]
    assert set(ids) == {1, 2, 3, 4}

    # Should be sorted by fused score descending
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
