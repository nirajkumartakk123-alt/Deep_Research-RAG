"""
Structure-aware chunking.

Chunks are built per page/section (never merging content across a page
or section boundary), using a word-count sliding window with overlap.
Word count is used instead of a real tokenizer to avoid an extra
dependency at this stage - swap in a tokenizer matched to the embedding
model later if exact token budgets become necessary (Phase 3+).
"""
from dataclasses import dataclass

from app.ingestion.loader import PageContent


@dataclass
class Chunk:
    chunk_index: int
    content: str
    page: int | None
    section: str | None
    token_count: int


def chunk_pages(
    pages: list[PageContent],
    max_chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_overlap >= max_chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than max_chunk_size ({max_chunk_size})."
        )

    step = max_chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_content in pages:
        words = page_content.text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + max_chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(
                Chunk(
                    chunk_index=chunk_index,
                    content=" ".join(chunk_words),
                    page=page_content.page,
                    section=page_content.section,
                    token_count=len(chunk_words),
                )
            )
            chunk_index += 1

            if end == len(words):
                break
            start += step

    return chunks