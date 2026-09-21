"""
Shared test fixtures. Truncates document tables before every test so
each test starts from a clean slate, and disposes the engine's
connection pool afterward (see NullPool note in connection.py for why
this matters on Windows with asyncpg + TestClient's per-call event loops).

invalidate_bm25_cache() is ALSO called here, and this matters for a
different reason: the BM25 index is a plain Python module-level cache,
completely independent of the database. Truncating document_chunks
does NOT clear it. Without this call, a test that builds the BM25
cache would leak stale chunk IDs into every subsequent test in the
same pytest session - the DB would be clean, but BM25 search would
silently return nothing (or fail to find chunks that were re-fetched
by ID and no longer exist). This is the same class of bug as the
event-loop/pooling issues from Phase 3: state that outlives the
boundary you assume resets it.
"""
import asyncio

import pytest
from sqlalchemy import text

from app.database.connection import engine
from app.retrieval.bm25 import invalidate_bm25_cache


@pytest.fixture(autouse=True)
def clean_database():
    async def _truncate_and_dispose():
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE document_chunks, documents RESTART IDENTITY CASCADE"))
        await engine.dispose()

    asyncio.run(_truncate_and_dispose())
    invalidate_bm25_cache()
    yield