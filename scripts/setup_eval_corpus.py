"""
Truncates document_chunks/documents and uploads exactly the 4 eval
documents from app.evaluation.dataset.EVAL_DOCUMENTS - nothing else.

Run this BEFORE every evaluation run, not just once - Recall@K/MRR/
NDCG's ground truth assumes this exact, isolated corpus. Running
evaluation against a corpus with leftover manual-testing documents
(Antarctica, Zylobrex-9000, etc. from earlier phases) would produce
meaningless retrieval metrics, since a wrong/irrelevant document
could plausibly outrank the correct one purely by accident.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from sqlalchemy import text

from app.database.connection import AsyncSessionLocal, engine
from app.database.repository import create_document_with_chunks
from app.evaluation.dataset import EVAL_DOCUMENTS
from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_text
from app.ingestion.loader import PageContent
from app.retrieval.bm25 import invalidate_bm25_cache
from app.retrieval.embeddings import embed_texts
from app.core.config import get_settings


async def setup_eval_corpus():
    settings = get_settings()

    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE document_chunks, documents RESTART IDENTITY CASCADE"))
    print("Truncated existing corpus.")

    async with AsyncSessionLocal() as db:
        for document_name, content in EVAL_DOCUMENTS.items():
            cleaned = clean_text(content)
            page = PageContent(text=cleaned, page=None, section=None)
            chunks = chunk_pages([page], settings.MAX_CHUNK_SIZE, settings.CHUNK_OVERLAP)

            embeddings = embed_texts([c.content for c in chunks])
            chunk_dicts = [
                {
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "page": c.page,
                    "section": c.section,
                    "token_count": c.token_count,
                    "embedding": embeddings[i],
                }
                for i, c in enumerate(chunks)
            ]

            import hashlib

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            await create_document_with_chunks(
                db,
                document_name=document_name,
                source=document_name,
                document_type="txt",
                content_hash=content_hash,
                chunks=chunk_dicts,
            )
            print(f"  Uploaded: {document_name} ({len(chunks)} chunk(s))")

    invalidate_bm25_cache()
    print("Eval corpus ready: exactly 4 documents, BM25 cache invalidated.")


if __name__ == "__main__":
    asyncio.run(setup_eval_corpus())