import uuid
from datetime import datetime

from pydantic import BaseModel


class ChunkPreview(BaseModel):
    chunk_index: int
    content: str
    page: int | None
    section: str | None
    token_count: int


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    document_name: str
    document_type: str
    content_hash: str
    total_chunks: int
    already_existed: bool
    chunks_preview: list[ChunkPreview]


class DocumentResponse(BaseModel):
    id: uuid.UUID
    document_name: str
    source: str
    document_type: str
    content_hash: str
    status: str
    created_at: datetime
    chunk_count: int


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    content: str
    page: int | None
    section: str | None
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    results: list[SearchResultItem]