"""
BM25 sparse retrieval.

Design: the corpus (tokenized chunk contents) is small enough at MVP
scale to hold entirely in memory and rebuild on demand, rather than
maintaining a persistent inverted index. The in-memory BM25Okapi
instance is cached at module level and invalidated explicitly via
invalidate_bm25_cache() whenever chunks are added or removed - callers
(the upload and delete routes, and the test fixture) are responsible
for calling it. This is a real gotcha: unlike the database, this cache
has no automatic transaction/rollback semantics, so any code path that
mutates document_chunks but forgets to invalidate will silently serve
stale search results.

Building the index is synchronous, CPU-bound work. For MVP corpus
sizes this completes fast enough not to matter, but it does briefly
block the event loop while running - a production system with a large
corpus would move index building to a background worker (Celery,
Phase 12) or a dedicated search engine (OpenSearch/Elasticsearch)
rather than rebuilding in-process on the request path.

_build_bm25 and _rank are deliberately pure (no DB access) and
underscore-prefixed as "internal", but are still imported directly by
tests - separating pure logic from the DB-fetching wrapper makes the
actual ranking algorithm testable without a database or event loop.
"""
import uuid
from dataclasses import dataclass, field
import re

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import DocumentChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _build_bm25(corpus: list[tuple[uuid.UUID, str]]):
    """Pure, DB-free: given [(id, text)] pairs, builds a BM25Okapi
    index. Returns (None, []) for an empty corpus rather than raising,
    since an empty document set is a valid (if unhelpful) state."""
    if not corpus:
        return None, []
    ids = [c[0] for c in corpus]
    tokenized_corpus = [tokenize(c[1]) for c in corpus]
    return BM25Okapi(tokenized_corpus), ids


def _rank(bm25, ids: list, query: str, top_k: int) -> list[tuple]:
    """Pure, DB-free: scores `query` against an already-built index."""
    if bm25 is None:
        return []
    scores = bm25.get_scores(tokenize(query))
    scored = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]


@dataclass
class _Bm25Cache:
    bm25: BM25Okapi | None = None
    chunk_ids: list = field(default_factory=list)
    built: bool = False


_cache = _Bm25Cache()


def invalidate_bm25_cache() -> None:
    """Call after any write to document_chunks (upload, delete) - and
    in test fixtures after truncating the DB - so the next query
    rebuilds against current data instead of serving a stale index."""
    _cache.bm25 = None
    _cache.chunk_ids = []
    _cache.built = False


async def _get_or_build_index(db: AsyncSession):
    if _cache.built:
        return _cache.bm25, _cache.chunk_ids

    result = await db.execute(select(DocumentChunk.id, DocumentChunk.content))
    rows = [(row[0], row[1]) for row in result.all()]

    bm25, ids = _build_bm25(rows)
    _cache.bm25 = bm25
    _cache.chunk_ids = ids
    _cache.built = True
    return bm25, ids


async def bm25_search(db: AsyncSession, query: str, top_k: int) -> list[tuple]:
    """Returns [(chunk_id, score)], best match first."""
    bm25, ids = await _get_or_build_index(db)
    return _rank(bm25, ids, query, top_k)


async def bm25_search_with_chunks(
    db: AsyncSession, query: str, top_k: int
) -> list[tuple[DocumentChunk, float]]:
    """Same as bm25_search, but fetches full DocumentChunk objects -
    used by the standalone /search/query?mode=bm25 endpoint so BM25
    can be tested/demoed in isolation, independent of hybrid fusion."""
    scored = await bm25_search(db, query, top_k)
    if not scored:
        return []

    ids = [chunk_id for chunk_id, _ in scored]
    result = await db.execute(
        select(DocumentChunk).options(selectinload(DocumentChunk.document)).where(DocumentChunk.id.in_(ids))
    )
    chunks_by_id = {c.id: c for c in result.scalars().all()}
    return [(chunks_by_id[chunk_id], score) for chunk_id, score in scored if chunk_id in chunks_by_id]