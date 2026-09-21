"""
Celery tasks. Currently one task: async document ingestion, wrapping
the Phase 2 pipeline (loader -> cleaner -> chunker) plus Phase 3
embedding + persistence, exactly as the synchronous /documents/upload
route does it - the difference is this runs in a worker process, so
the HTTP request returns immediately with a job_id instead of blocking
on ingestion + embedding.

Since Celery tasks are synchronous by default and our DB/embedding
code is async, each task wraps its async logic in asyncio.run() -
this creates a fresh event loop per task invocation, which sidesteps
the cross-event-loop pooling issues from Phase 3 (NullPool means every
session is a fresh connection anyway).
"""
import asyncio
import logging
from pathlib import Path

from app.core.config import get_settings
from app.database.connection import AsyncSessionLocal
from app.database.repository import create_document_with_chunks, get_document_by_hash
from app.ingestion.pipeline import run_ingestion_pipeline
from app.retrieval.bm25 import invalidate_bm25_cache
from app.retrieval.embeddings import embed_texts
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _process_document_async(file_path: str, original_filename: str) -> dict:
    settings = get_settings()

    result = run_ingestion_pipeline(Path(file_path), original_filename)

    async with AsyncSessionLocal() as db:
        existing = await get_document_by_hash(db, result.content_hash)
        if existing is not None:
            return {
                "document_id": str(existing.id),
                "document_name": existing.document_name,
                "already_existed": True,
                "total_chunks": len(existing.chunks),
            }

        chunk_texts = [c.content for c in result.chunks]
        embeddings = embed_texts(chunk_texts)

        chunk_dicts = [
            {
                "chunk_index": c.chunk_index,
                "content": c.content,
                "page": c.page,
                "section": c.section,
                "token_count": c.token_count,
                "embedding": embeddings[i],
            }
            for i, c in enumerate(result.chunks)
        ]

        document = await create_document_with_chunks(
            db,
            document_name=result.document_name,
            source=original_filename,
            document_type=result.document_type,
            content_hash=result.content_hash,
            chunks=chunk_dicts,
        )

    invalidate_bm25_cache()

    return {
        "document_id": str(document.id),
        "document_name": document.document_name,
        "already_existed": False,
        "total_chunks": len(chunk_dicts),
    }


@celery_app.task(bind=True, name="process_document")
def process_document_task(self, file_path: str, original_filename: str) -> dict:
    """Synchronous Celery entrypoint - wraps the async pipeline in
    asyncio.run(). Raises on failure so Celery marks the task FAILURE
    and stores the exception; the job-status endpoint surfaces this."""
    try:
        return asyncio.run(_process_document_async(file_path, original_filename))
    except Exception as exc:
        logger.error(f"Document processing failed for {original_filename}: {exc}")
        raise