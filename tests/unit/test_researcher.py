"""
Requires Postgres + Redis running (retrieval touches the real DB) AND
a real Groq call happens upstream if invoked via the full graph - but
these tests call private_retriever_node directly, bypassing the LLM
entirely, so they're pure retrieval tests, not agentic-reasoning tests.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.agents.researcher import private_retriever_node
from app.main import app

client = TestClient(app)


def _upload(filename: str, content: str):
    return client.post(
        "/documents/upload", files={"file": (filename, io.BytesIO(content.encode()), "text/plain")}
    )


@pytest.mark.asyncio
async def test_retriever_uses_query_when_no_subtasks_present():
    _upload("researcher_test1.txt", "The Great Wall of China is over 13,000 miles long. " * 5)

    result = await private_retriever_node({"query": "How long is the Great Wall of China?"})
    chunks = result["retrieved_chunks"]

    assert len(chunks) > 0
    assert all(c["task"] == "How long is the Great Wall of China?" for c in chunks)


@pytest.mark.asyncio
async def test_retriever_runs_once_per_subtask():
    _upload("researcher_test2.txt", "Mount Everest is the tallest mountain above sea level. " * 5)
    _upload("researcher_test3.txt", "The Mariana Trench is the deepest part of the ocean. " * 5)

    result = await private_retriever_node(
        {
            "query": "irrelevant - subtasks take priority",
            "subtasks": ["What is the tallest mountain?", "What is the deepest ocean trench?"],
        }
    )
    chunks = result["retrieved_chunks"]

    tasks_represented = {c["task"] for c in chunks}
    assert "What is the tallest mountain?" in tasks_represented
    assert "What is the deepest ocean trench?" in tasks_represented


@pytest.mark.asyncio
async def test_retrieved_chunks_include_citation_metadata():
    _upload("researcher_test4.txt", "Antarctica is the coldest continent on Earth. " * 5)

    result = await private_retriever_node({"query": "What is the coldest continent?"})
    chunks = result["retrieved_chunks"]

    assert len(chunks) > 0
    first = chunks[0]
    for field in ("chunk_id", "document_id", "document_name", "content", "score"):
        assert field in first