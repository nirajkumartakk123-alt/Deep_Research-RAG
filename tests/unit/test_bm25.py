import uuid

from app.retrieval.bm25 import _build_bm25, _rank, tokenize


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_build_bm25_empty_corpus_returns_none():
    bm25, ids = _build_bm25([])
    assert bm25 is None
    assert ids == []


def test_rank_returns_relevant_docs_before_irrelevant():
    id1, id2, id3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    corpus = [
        (id1, "The cat sat on the mat"),
        (id2, "Quarterly revenue exceeded analyst expectations"),
        (id3, "A cat and a dog played together"),
    ]
    bm25, ids = _build_bm25(corpus)
    results = _rank(bm25, ids, "cat", top_k=3)

    result_ids = [r[0] for r in results]
    assert id2 == result_ids[-1]  # the one document with zero relevant terms ranks last


def test_rank_respects_top_k_limit():
    corpus = [(uuid.uuid4(), f"document number {i} about cats") for i in range(10)]
    bm25, ids = _build_bm25(corpus)
    results = _rank(bm25, ids, "cats", top_k=3)
    assert len(results) == 3


def test_rank_with_no_index_returns_empty():
    assert _rank(None, [], "anything", top_k=5) == []