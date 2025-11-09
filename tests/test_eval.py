from src.eval import recall_at_k, mean_recall_at_k


def test_recall_at_k_basic():
    gold = {"a", "b", "c"}
    preds = ["x", "b", "c", "y"]
    r = recall_at_k(gold, preds, k=3)
    # in top-3 preds we have b and c -> 2/3
    assert abs(r - (2 / 3)) < 1e-6


def test_mean_recall_at_k_multiple_queries():
    gold = {
        "q1": {"a", "b"},
        "q2": {"x"},
    }
    preds = {
        "q1": ["a", "z"],
        "q2": ["y", "x"],
    }
    # q1: 1/2, q2: 1/1 -> mean = 0.75
    mr = mean_recall_at_k(gold, preds, k=2)
    assert abs(mr - 0.75) < 1e-6


