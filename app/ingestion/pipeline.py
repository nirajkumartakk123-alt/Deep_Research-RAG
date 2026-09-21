"""
Orchestrates: validate -> load -> clean -> chunk.

Phase 2 stops here - it does NOT persist chunks to the database.
Storage lands in Phase 3 once the pgvector schema exists. The
original uploaded file IS saved to disk by the API route so it can
be re-run through this pipeline later without re-uploading.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import EmptyDocumentError, ValidationAppError
from app.ingestion.chunker import Chunk, chunk_pages
from app.ingestion.cleaner import clean_text
from app.ingestion.loader import PageContent, load_document


@dataclass
class IngestionResult:
    document_name: str
    document_type: str
    content_hash: str
    chunks: list[Chunk]


def compute_content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def run_ingestion_pipeline(file_path: Path, original_filename: str) -> IngestionResult:
    settings = get_settings()

    if settings.CHUNK_OVERLAP >= settings.MAX_CHUNK_SIZE:
        raise ValidationAppError(
            "CHUNK_OVERLAP must be smaller than MAX_CHUNK_SIZE. Check your .env configuration."
        )

    raw_bytes = file_path.read_bytes()
    if len(raw_bytes) == 0:
        raise EmptyDocumentError(f"'{original_filename}' is an empty file (0 bytes).")

    content_hash = compute_content_hash(raw_bytes)

    loaded = load_document(file_path, original_filename)

    cleaned_pages = [
        PageContent(text=clean_text(p.text), page=p.page, section=p.section) for p in loaded.pages
    ]
    cleaned_pages = [p for p in cleaned_pages if p.text]

    if not cleaned_pages:
        raise EmptyDocumentError(
            f"No extractable text content found in '{original_filename}'. "
            "The file may be empty, scanned/image-only, or corrupted."
        )

    chunks = chunk_pages(cleaned_pages, settings.MAX_CHUNK_SIZE, settings.CHUNK_OVERLAP)

    if not chunks:
        raise EmptyDocumentError(f"No chunks could be produced from '{original_filename}'.")

    return IngestionResult(
        document_name=original_filename,
        document_type=loaded.document_type,
        content_hash=content_hash,
        chunks=chunks,
    )