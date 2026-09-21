"""
Hybrid retrieval: combines dense (vector) and sparse (BM25) retrieval
via Reciprocal Rank Fusion (RRF).

RRF chosen over normalized-score fusion because vector cosine
similarity and BM25 scores live on completely different, corpus-size-
dependent scales - averaging or min-max-normalizing them directly is
fragile and shifts as the corpus grows. RRF instead fuses purely on
RANK POSITION: score = sum(1 / (k + rank + 1)) across every retriever
a chunk appears in. This makes it scale-invariant and robust even when
one retriever's score distribution looks nothing like the other's.

reciprocal_rank_fusion is a pure function (plain IDs in, scored IDs
out) so it's testable without touching the database at all.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import DocumentChunk
from app.retrieval.bm25 import bm25_search
from app.retrieval.vector_search import vector_search

RRF_K = 60  # standard default from the original RRF paper - dampens the influence of any single top rank


def reciprocal_rank_fusion(rankings: list[list], k: int = RRF_K) -> list[tuple]:
    """Given multiple best-first ranked lists of IDs, returns a single
    fused ranking [(id, score)], best-first."""
    scores: dict = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def hybrid_search(
    db: AsyncSession, query: str, top_k: int, candidate_k: int | None = None
) -> list[tuple[DocumentChunk, float]]:
    """Returns [(chunk, fused_score)], best match first.

    candidate_k controls how many candidates each individual retriever
    contributes before fusion - defaults to top_k * 4 to give RRF a
    wider pool to re-rank from than the final result size."""
    candidate_k = candidate_k or top_k * 4

    vector_results = await vector_search(db, query, candidate_k)
    bm25_results = await bm25_search(db, query, candidate_k)

    vector_ranking = [chunk.id for chunk, _ in vector_results]
    bm25_ranking = [chunk_id for chunk_id, _ in bm25_results]

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])[:top_k]
    top_ids = [chunk_id for chunk_id, _ in fused]

    if not top_ids:
        return []

    result = await db.execute(
        select(DocumentChunk).options(selectinload(DocumentChunk.document)).where(DocumentChunk.id.in_(top_ids))
    )
    chunks_by_id = {c.id: c for c in result.scalars().all()}
    score_by_id = dict(fused)

    # top_ids is already in fused rank order; IN() does not guarantee
    # row order, so we rebuild the list from top_ids rather than from
    # the query result directly.
    return [(chunks_by_id[cid], score_by_id[cid]) for cid in top_ids if cid in chunks_by_id]