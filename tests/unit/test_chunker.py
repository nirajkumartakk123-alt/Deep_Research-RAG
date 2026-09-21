import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.loader import PageContent


def test_chunk_pages_splits_long_text():
    pages = [PageContent(text="word " * 20, page=1, section=None)]
    chunks = chunk_pages(pages, max_chunk_size=5, chunk_overlap=1)
    assert len(chunks) > 1
    assert all(c.token_count <= 5 for c in chunks)


def test_chunk_pages_preserves_page_and_section_metadata():
    pages = [PageContent(text="a b c", page=3, section="Intro")]
    chunks = chunk_pages(pages, max_chunk_size=10, chunk_overlap=0)
    assert chunks[0].page == 3
    assert chunks[0].section == "Intro"


def test_chunk_pages_skips_empty_pages():
    pages = [PageContent(text="", page=1, section=None)]
    assert chunk_pages(pages, max_chunk_size=10, chunk_overlap=0) == []


def test_chunk_pages_indices_are_sequential():
    pages = [PageContent(text="word " * 20, page=1, section=None)]
    chunks = chunk_pages(pages, max_chunk_size=5, chunk_overlap=1)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_rejects_overlap_ge_chunk_size():
    pages = [PageContent(text="a b c", page=1, section=None)]
    with pytest.raises(ValueError):
        chunk_pages(pages, max_chunk_size=5, chunk_overlap=5)