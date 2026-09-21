"""
Brute-force cosine-similarity search over document_chunks.

No ANN index yet (see migration file comment) - at MVP corpus sizes,
a sequential scan with pgvector's cosine_distance operator is fast
enough and avoids IVFFlat/HNSW tuning before there's data to justify it.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import DocumentChunk
from app.retrieval.embeddings import embed_query


async def vector_search(db: AsyncSession, query: str, top_k: int) -> list[tuple[DocumentChunk, float]]:
    """Returns [(chunk, similarity_score)], best match first.
    similarity_score = 1 - cosine_distance, so 1.0 = identical, 0.0 = orthogonal.

    Eagerly loads chunk.document via selectinload, since callers need
    document_name for the response - without this, accessing
    chunk.document later raises MissingGreenlet (see repository.py note)."""
    query_embedding = embed_query(query)

    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(DocumentChunk, distance.label("distance"))
        .options(selectinload(DocumentChunk.document))
        .order_by(distance)
        .limit(top_k)
    )

    result = await db.execute(stmt)
    rows = result.all()
    return [(row[0], 1 - row[1]) for row in rows]