"""
Uses real chunk-like objects (a minimal stand-in, not the full ORM
model) so the reranker's logic can be tested without touching the
database - same testability principle as bm25.py's pure functions.
"""
from dataclasses import dataclass

from app.retrieval.reranker import rerank


@dataclass
class FakeChunk:
    content: str


def test_rerank_puts_relevant_content_first():
    candidates = [
        (FakeChunk("Bananas are a good source of potassium."), 0.5),
        (FakeChunk("PostgreSQL is a powerful open-source relational database."), 0.5),
    ]
    results = rerank("What is PostgreSQL?", candidates, top_k=2)

    assert "PostgreSQL" in results[0][0].content


def test_rerank_respects_top_k():
    candidates = [(FakeChunk(f"Document number {i}."), 0.1) for i in range(5)]
    results = rerank("document", candidates, top_k=2)
    assert len(results) == 2


def test_rerank_empty_candidates_returns_empty():
    assert rerank("anything", [], top_k=5) == []


def test_rerank_discards_incoming_scores_uses_own_scale():
    """Cross-encoder scores are NOT the same scale as the incoming
    retrieval scores - this test just confirms the returned scores
    are floats produced by the model, not the original input scores
    echoed back unchanged."""
    candidates = [(FakeChunk("Relevant text about cats."), 999.0)]
    results = rerank("cats", candidates, top_k=1)
    assert results[0][1] != 999.0