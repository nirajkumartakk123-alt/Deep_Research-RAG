"""
Repository layer - the ONLY place raw DB queries live. Routes/services
call these functions rather than writing SQLAlchemy queries inline,
so query logic stays testable and centralized.

IMPORTANT (async SQLAlchemy specific): every query that returns a
Document or DocumentChunk whose related object will be accessed later
(document.chunks, chunk.document) MUST eagerly load that relationship
with selectinload() here. Lazy-loading a relationship after the async
session has moved past the point of the original query raises
MissingGreenlet - async SQLAlchemy cannot implicitly run a background
query the way sync SQLAlchemy can. This is not optional cleanup; it's
required correctness for every query in this file.
"""
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import Document, DocumentChunk


async def get_document_by_hash(db: AsyncSession, content_hash: str) -> Document | None:
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.content_hash == content_hash)
    )
    return result.scalar_one_or_none()


async def create_document_with_chunks(
    db: AsyncSession,
    document_name: str,
    source: str,
    document_type: str,
    content_hash: str,
    chunks: list[dict],
) -> Document:
    document = Document(
        document_name=document_name,
        source=source,
        document_type=document_type,
        content_hash=content_hash,
        status="completed",
    )
    db.add(document)
    await db.flush()  # populates document.id before we attach chunks to it

    for chunk in chunks:
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                page=chunk["page"],
                section=chunk["section"],
                token_count=chunk["token_count"],
                embedding=chunk["embedding"],
            )
        )

    await db.commit()
    await db.refresh(document, attribute_names=["chunks"])
    return document


async def list_documents(db: AsyncSession, limit: int = 50, offset: int = 0) -> list[Document]:
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def get_document_by_id(db: AsyncSession, document_id: uuid.UUID) -> Document | None:
    result = await db.execute(
        select(Document).options(selectinload(Document.chunks)).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()


async def delete_document(db: AsyncSession, document_id: uuid.UUID) -> bool:
    document = await get_document_by_id(db, document_id)
    if document is None:
        return False
    await db.execute(delete(Document).where(Document.id == document_id))
    await db.commit()
    return True