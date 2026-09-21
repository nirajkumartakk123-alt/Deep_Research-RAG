import pytest

from app.agents.query_rewriter import query_rewriter_node


@pytest.mark.asyncio
async def test_rewriter_produces_different_query_text():
    state = {
        "query": "irrelevant",
        "subtasks": ["What is the GPU pricing for cloud provider XYZ?"],
        "retrieved_chunks": [],
        "web_results": [],
        "verification_result": [
            {
                "task": "What is the GPU pricing for cloud provider XYZ?",
                "sufficient": False,
                "confidence": 0.2,
                "missing_evidence": "No relevant pricing information was found; 'XYZ' isn't a real provider name.",
            }
        ],
    }
    result = await query_rewriter_node(state)
    assert result["subtasks"][0] != "What is the GPU pricing for cloud provider XYZ?"
    assert len(result["pending_retrieval_tasks"]) == 1


@pytest.mark.asyncio
async def test_rewriter_increments_rewrite_count():
    state = {
        "query": "irrelevant",
        "subtasks": ["some task"],
        "retrieved_chunks": [],
        "web_results": [],
        "rewrite_count": 1,
        "verification_result": [
            {"task": "some task", "sufficient": False, "confidence": 0.1, "missing_evidence": "vague"}
        ],
    }
    result = await query_rewriter_node(state)
    assert result["rewrite_count"] == 2


@pytest.mark.asyncio
async def test_rewriter_removes_stale_evidence_for_rewritten_task():
    state = {
        "query": "irrelevant",
        "subtasks": ["old task"],
        "retrieved_chunks": [{"task": "old task", "content": "stale content"}],
        "web_results": [],
        "verification_result": [
            {"task": "old task", "sufficient": False, "confidence": 0.1, "missing_evidence": "weak"}
        ],
    }
    result = await query_rewriter_node(state)
    assert all(c["task"] != "old task" for c in result["retrieved_chunks"])