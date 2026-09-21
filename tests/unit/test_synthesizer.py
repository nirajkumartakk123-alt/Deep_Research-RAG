import pytest

from app.agents.synthesizer import synthesizer_node


@pytest.mark.asyncio
async def test_synthesizer_produces_claims_from_sufficient_task():
    state = {
        "verification_result": [
            {"task": "What is the capital of France?", "sufficient": True, "confidence": 0.9, "missing_evidence": ""}
        ],
        "retrieved_chunks": [
            {
                "task": "What is the capital of France?",
                "content": "Paris has been the capital of France since 987 AD.",
                "document_name": "france.txt",
                "page": 1,
                "section": None,
                "chunk_id": "chunk-1",
            }
        ],
        "web_results": [],
    }
    result = await synthesizer_node(state)
    assert len(result["claims_with_evidence"]) > 0
    assert "final_report" in result
    assert "Insufficient" not in result["final_report"]


@pytest.mark.asyncio
async def test_synthesizer_marks_insufficient_tasks_as_gaps_not_answered():
    state = {
        "verification_result": [
            {"task": "What is the population of a fictional city?", "sufficient": False, "confidence": 0.9, "missing_evidence": "No data found"}
        ],
        "retrieved_chunks": [],
        "web_results": [],
    }
    result = await synthesizer_node(state)
    assert result["claims_with_evidence"] == []
    assert "Insufficient evidence" in result["final_report"]


@pytest.mark.asyncio
async def test_synthesizer_every_claim_has_traceable_evidence():
    """Confirms the evidence_index mechanism actually worked - every
    claim in claims_with_evidence must be paired with a real evidence
    dict that has actual content, not None or empty."""
    state = {
        "verification_result": [
            {"task": "What is the boiling point of water?", "sufficient": True, "confidence": 0.95, "missing_evidence": ""}
        ],
        "retrieved_chunks": [
            {
                "task": "What is the boiling point of water?",
                "content": "Water boils at 100 degrees Celsius at sea level atmospheric pressure.",
                "document_name": "physics.txt",
                "page": None,
                "section": None,
                "chunk_id": "chunk-2",
            }
        ],
        "web_results": [],
    }
    result = await synthesizer_node(state)
    for claim, evidence in result["claims_with_evidence"]:
        assert claim
        assert evidence["content"]
        assert evidence["document_name"] == "physics.txt"