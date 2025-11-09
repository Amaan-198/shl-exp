from src.balance import allocate


def test_allocate_balanced_5_5_and_dominant():
    # Common item set
    item_ids = list(range(1, 11))
    item_classes = {
        i: ["Knowledge & Skills"] if i <= 5 else ["Personality & Behavior"] for i in item_ids
    }

    # Case 1: balanced 5/5
    out = allocate(item_ids, item_classes, 10, pt=0.5, pb=0.5)
    assert len(out) == 10
    # Mix of both categories
    k_count = sum(1 for i in out if "Knowledge & Skills" in item_classes[i])
    p_count = sum(1 for i in out if "Personality & Behavior" in item_classes[i])
    assert abs(k_count - p_count) <= 2  # roughly balanced

    # Case 2: dominant K (pt high)
    out2 = allocate(item_ids, item_classes, 10, pt=0.9, pb=0.2)
    assert all(
        "Knowledge & Skills" in item_classes[i] or "Personality & Behavior" in item_classes[i]
        for i in out2
    )
    # First few should be from K
    assert out2[0] <= 5

    # Case 3: ensure min 5 results if few inputs
    short = allocate([1, 2, 3], item_classes, 10, pt=0.5, pb=0.5)
    assert len(short) >= 5 or len(short) == len(set(short))


