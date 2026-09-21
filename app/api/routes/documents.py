"""
Document endpoints: synchronous upload (Phases 2-9, unchanged),
async processing via Celery (Phase 11, new), list/get/delete, and
search with vector/bm25/hybrid modes + optional reranking.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import AppException, DocumentExtractionError, NotFoundError, ValidationAppError
from app.database.connection import get_db
from app.database.repository import (
    create_document_with_chunks,
    delete_document,
    get_document_by_hash,
    get_document_by_id,
    list_documents,
)
from app.ingestion.loader import ALLOWED_EXTENSIONS
from app.ingestion.pipeline import run_ingestion_pipeline
from app.retrieval.bm25 import bm25_search_with_chunks, invalidate_bm25_cache
from app.retrieval.embeddings import embed_texts
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.retrieval.vector_search import vector_search
from app.schemas.documents import (
    ChunkPreview,
    DocumentResponse,
    DocumentUploadResponse,
    SearchResponse,
    SearchResultItem,
)
from app.schemas.jobs import JobSubmittedResponse
from app.workers.tasks import process_document_task

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PREVIEW_LIMIT = 5


def _validate_and_save_upload(file: UploadFile, contents: bytes) -> Path:
    settings = get_settings()

    if not file.filename:
        raise ValidationAppError("Uploaded file is missing a filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValidationAppError(
            f"Unsupported file type '{suffix}'. Supported types: {sorted(ALLOWED_EXTENSIONS)}"
        )

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise ValidationAppError(
            f"File exceeds max upload size of {settings.MAX_UPLOAD_SIZE_MB}MB (got {size_mb:.2f}MB)."
        )

    unique_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{unique_id}_{file.filename}"
    saved_path.write_bytes(contents)
    return saved_path


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
) -> DocumentUploadResponse:
    """Synchronous upload - processes inline, blocks until done.
    Kept for small files / simple use; large files should use
    /documents/process for async handling."""
    contents = await file.read()
    saved_path = _validate_and_save_upload(file, contents)

    try:
        result = run_ingestion_pipeline(saved_path, file.filename)
    except AppException:
        raise
    except Exception as exc:
        raise DocumentExtractionError(f"Failed to process '{file.filename}': {exc}") from exc

    existing = await get_document_by_hash(db, result.content_hash)
    if existing is not None:
        preview = [
            ChunkPreview(
                chunk_index=c.chunk_index, content=c.content, page=c.page, section=c.section,
                token_count=c.token_count,
            )
            for c in existing.chunks[:PREVIEW_LIMIT]
        ]
        return DocumentUploadResponse(
            document_id=existing.id, document_name=existing.document_name,
            document_type=existing.document_type, content_hash=existing.content_hash,
            total_chunks=len(existing.chunks), already_existed=True, chunks_preview=preview,
        )

    chunk_texts = [c.content for c in result.chunks]
    embeddings = embed_texts(chunk_texts)
    chunk_dicts = [
        {
            "chunk_index": c.chunk_index, "content": c.content, "page": c.page,
            "section": c.section, "token_count": c.token_count, "embedding": embeddings[i],
        }
        for i, c in enumerate(result.chunks)
    ]

    document = await create_document_with_chunks(
        db, document_name=result.document_name, source=file.filename,
        document_type=result.document_type, content_hash=result.content_hash, chunks=chunk_dicts,
    )
    invalidate_bm25_cache()

    preview = [
        ChunkPreview(
            chunk_index=c["chunk_index"], content=c["content"], page=c["page"],
            section=c["section"], token_count=c["token_count"],
        )
        for c in chunk_dicts[:PREVIEW_LIMIT]
    ]

    return DocumentUploadResponse(
        document_id=document.id, document_name=document.document_name,
        document_type=document.document_type, content_hash=document.content_hash,
        total_chunks=len(chunk_dicts), already_existed=False, chunks_preview=preview,
    )


@router.post("/process", response_model=JobSubmittedResponse)
async def process_document_async(file: UploadFile = File(...)) -> JobSubmittedResponse:
    """Async upload - saves the file, submits a Celery task, returns
    immediately with a job_id. Poll GET /jobs/{job_id} for completion.
    Validation (type/size) happens here, synchronously, since a bad
    file should fail fast with a clear 422 rather than fail invisibly
    inside a background worker."""
    contents = await file.read()
    saved_path = _validate_and_save_upload(file, contents)

    task = process_document_task.delay(str(saved_path), file.filename)

    return JobSubmittedResponse(job_id=task.id, status="PENDING")


@router.get("", response_model=list[DocumentResponse])
async def get_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    documents = await list_documents(db, limit=limit, offset=offset)
    return [
        DocumentResponse(
            id=d.id, document_name=d.document_name, source=d.source, document_type=d.document_type,
            content_hash=d.content_hash, status=d.status, created_at=d.created_at,
            chunk_count=len(d.chunks),
        )
        for d in documents
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> DocumentResponse:
    document = await get_document_by_id(db, document_id)
    if document is None:
        raise NotFoundError(f"Document '{document_id}' not found.")
    return DocumentResponse(
        id=document.id, document_name=document.document_name, source=document.source,
        document_type=document.document_type, content_hash=document.content_hash,
        status=document.status, created_at=document.created_at, chunk_count=len(document.chunks),
    )


@router.delete("/{document_id}")
async def remove_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    deleted = await delete_document(db, document_id)
    if not deleted:
        raise NotFoundError(f"Document '{document_id}' not found.")
    invalidate_bm25_cache()
    return {"deleted": True, "document_id": str(document_id)}


@router.get("/search/query", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    mode: str = Query("hybrid", pattern="^(vector|bm25|hybrid)$"),
    use_reranker: bool = Query(False),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    settings = get_settings()
    retrieval_k = settings.TOP_K if use_reranker else top_k

    if mode == "vector":
        results = await vector_search(db, q, retrieval_k)
    elif mode == "bm25":
        results = await bm25_search_with_chunks(db, q, retrieval_k)
    else:
        results = await hybrid_search(db, q, retrieval_k)

    if use_reranker:
        results = rerank(q, results, top_k=min(top_k, settings.RERANK_TOP_K))

    return SearchResponse(
        query=q, mode=mode, reranked=use_reranker,
        results=[
            SearchResultItem(
                chunk_id=chunk.id, document_id=chunk.document_id,
                document_name=chunk.document.document_name, content=chunk.content,
                page=chunk.page, section=chunk.section, score=round(score, 4),
            )
            for chunk, score in results
        ],
    )