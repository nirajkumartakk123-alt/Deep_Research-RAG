"""
Cross-encoder reranking.

Unlike the bi-encoder embedding model (which encodes query and passage
SEPARATELY, then compares vectors via cosine similarity), a cross-
encoder feeds the (query, passage) pair through the model TOGETHER,
letting it directly attend to both at once. This is strictly more
accurate for relevance scoring - at the cost of being unable to
pre-compute passage representations, so it can only run at query time
over a small candidate set, not the whole corpus. That's exactly why
reranking sits AFTER hybrid retrieval narrows the field, not instead
of it: cross-encoding every chunk in a large corpus per query would be
far too slow.

Model is loaded once via lru_cache, same pattern as embeddings.py -
loading weights from disk is expensive relative to running inference.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.database.models import DocumentChunk


@lru_cache
def get_reranker_model() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.RERANKER_MODEL)


def rerank(
    query: str, candidates: list[tuple[DocumentChunk, float]], top_k: int
) -> list[tuple[DocumentChunk, float]]:
    """Re-scores (chunk, _) candidates against the query using the
    cross-encoder, discarding the incoming retrieval score entirely -
    reranking scores are on a different scale and semantics than
    RRF/cosine/BM25 scores, so mixing them would be meaningless.
    Returns the top_k candidates sorted by cross-encoder relevance,
    best first."""
    if not candidates:
        return []

    model = get_reranker_model()
    pairs = [(query, chunk.content) for chunk, _ in candidates]
    scores = model.predict(pairs)

    scored = list(zip([c for c, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [(chunk, float(score)) for chunk, score in scored[:top_k]]