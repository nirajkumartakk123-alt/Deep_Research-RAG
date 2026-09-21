"""
Retrieval quality metrics - Recall@K, MRR, NDCG. All pure functions
over a ranked list of document names and a single expected/relevant
document name, matching this project's small eval set (one clearly
correct source document per question, not multi-relevant-document
ground truth). No LLM calls, no database, no network - these should
be the fastest, most reliable functions in the entire evaluation
framework.
"""
import math


def recall_at_k(ranked_document_names: list[str], expected_document_name: str, k: int) -> float:
    """1.0 if the expected document appears anywhere in the top K
    results, else 0.0. With a single relevant document per query,
    Recall@K collapses to a hit/miss indicator - this is a real,
    known simplification worth being upfront about: true Recall@K
    with multiple relevant documents per query would instead be
    (relevant docs found in top K) / (total relevant docs)."""
    return 1.0 if expected_document_name in ranked_document_names[:k] else 0.0


def mrr(ranked_document_names: list[str], expected_document_name: str) -> float:
    """Mean Reciprocal Rank for a SINGLE query - averaging across
    queries happens at the caller level. Returns 1/rank of the first
    occurrence of the expected document (1-indexed), or 0.0 if absent
    entirely."""
    for i, name in enumerate(ranked_document_names, start=1):
        if name == expected_document_name:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_document_names: list[str], expected_document_name: str, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K, for a single binary-
    relevance document. DCG = 1/log2(rank+1) if the expected document
    appears within the top K, else 0. IDCG (ideal DCG) is always
    1/log2(2)=1.0 for a single relevant document appearing at rank 1,
    so NDCG here simplifies to DCG/1.0 = DCG - again a known
    simplification appropriate for this dataset's single-relevant-
    document structure, not a general-purpose multi-relevance NDCG."""
    top_k = ranked_document_names[:k]
    for i, name in enumerate(top_k, start=1):
        if name == expected_document_name:
            return 1.0 / math.log2(i + 1)
    return 0.0