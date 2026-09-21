import pytest

from app.agents.verifier import verifier_node


@pytest.mark.asyncio
async def test_verifier_marks_relevant_evidence_sufficient():
    state = {
        "query": "What is the capital of France?",
        "retrieved_chunks": [
            {"task": "What is the capital of France?", "content": "Paris is the capital and largest city of France."}
        ],
        "web_results": [],
    }
    result = await verifier_node(state)
    verification = result["verification_result"]
    assert len(verification) == 1
    assert verification[0]["sufficient"] is True


@pytest.mark.asyncio
async def test_verifier_marks_missing_evidence_insufficient():
    state = {"query": "What is the exact population of a fictional city called Zylobrexville?", "retrieved_chunks": [], "web_results": []}
    result = await verifier_node(state)
    verification = result["verification_result"]
    assert verification[0]["sufficient"] is False
    assert len(verification[0]["missing_evidence"]) > 0


@pytest.mark.asyncio
async def test_verifier_skips_reverifying_already_sufficient_tasks():
    """If a prior verification_result already marks a task sufficient,
    verifier_node should reuse it rather than making a new LLM call -
    this test confirms the returned entry is literally the same dict,
    not a freshly re-judged one, by checking prior data survives
    unchanged even though no evidence is provided this time (which
    would otherwise force an 'insufficient: no evidence' result)."""
    state = {
        "query": "irrelevant",
        "retrieved_chunks": [],
        "web_results": [],
        "verification_result": [
            {"task": "irrelevant", "sufficient": True, "confidence": 0.99, "missing_evidence": ""}
        ],
    }
    result = await verifier_node(state)
    assert result["verification_result"][0]["sufficient"] is True
    assert result["verification_result"][0]["confidence"] == 0.99