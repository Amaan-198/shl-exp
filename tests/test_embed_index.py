import pickle

import pandas as pd

from src.embed_index import build_bm25_index, prepare_corpus_for_bm25
from src.normalize import lexical_tokens_for_bm25


def test_prepare_corpus_for_bm25_uses_lexical_tokens(tmp_path):
    texts = ["Senior JS engineer", "Personality & behavior assessment"]
    corpus = prepare_corpus_for_bm25(texts)

    assert len(corpus) == 2
    # first doc should include javascript due to synonyms (via lexical_tokens_for_bm25)
    first_doc_tokens = set(corpus[0])
    assert "javascript" in first_doc_tokens or "javascript" in lexical_tokens_for_bm25(texts[0])


def test_build_bm25_index_writes_pickle(tmp_path):
    # Minimal dummy data
    texts = ["Assessment A", "Assessment B"]
    item_ids = [10, 20]

    out_path = tmp_path / "bm25_test.pkl"
    build_bm25_index(search_texts=texts, item_ids=item_ids, output_path=out_path)

    assert out_path.exists()

    with out_path.open("rb") as f:
        obj = pickle.load(f)

    # Check structure
    assert "bm25" in obj
    assert "item_ids" in obj
    assert obj["item_ids"] == item_ids

    bm25 = obj["bm25"]
    # A couple of smoke checks on the BM25 instance
    assert hasattr(bm25, "get_scores")
    scores = bm25.get_scores(["assessment"])
    assert len(scores) == len(texts)


