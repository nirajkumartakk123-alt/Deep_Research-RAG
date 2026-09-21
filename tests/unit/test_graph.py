"""
Requires Postgres + Redis, real Groq calls, and a real Tavily call
for most tests. The termination-logic tests do NOT - they call
_route_after_verification directly with fabricated state, so they run
instantly and cannot be affected by LLM judgment variability. These
are the most important tests in this file, since a broken bound here
is the single worst failure mode in the whole corrective RAG loop.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.agents.graph import _route_after_verification, get_graph
from app.main import app

client = TestClient(app)


def _upload(filename: str, content: str):
    return client.post(
        "/documents/upload", files={"file": (filename, io.BytesIO(content.encode()), "text/plain")}
    )


# --- Pure, LLM-free, DB-free termination-logic tests ---

def test_route_after_verification_stops_when_all_sufficient():
    state = {"verification_result": [{"task": "a", "sufficient": True, "confidence": 0.9, "missing_evidence": ""}]}
    assert _route_after_verification(state) == "done"


def test_route_after_verification_continues_when_insufficient_and_under_limit():
    state = {
        "verification_result": [{"task": "a", "sufficient": False, "confidence": 0.5, "missing_evidence": "x"}],
        "rewrite_count": 0,
    }
    assert _route_after_verification(state) == "needs_rewrite"


def test_route_after_verification_stops_at_max_retries_even_if_insufficient():
    """The critical regression test for loop termination: no matter
    how bad the evidence is, once rewrite_count reaches MAX_RETRIES,
    the loop MUST end rather than continue indefinitely."""
    from app.core.config import get_settings

    max_retries = get_settings().MAX_RETRIES
    state = {
        "verification_result": [{"task": "a", "sufficient": False, "confidence": 0.1, "missing_evidence": "x"}],
        "rewrite_count": max_retries,
    }
    assert _route_after_verification(state) == "done"


def test_route_after_verification_handles_multiple_tasks_mixed_sufficiency():
    state = {
        "verification_result": [
            {"task": "a", "sufficient": True, "confidence": 0.9, "missing_evidence": ""},
            {"task": "b", "sufficient": False, "confidence": 0.3, "missing_evidence": "x"},
        ],
        "rewrite_count": 0,
    }
    assert _route_after_verification(state) == "needs_rewrite"


# --- Existing end-to-end graph tests ---

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graph_runs_end_to_end_and_populates_query_analysis():
    graph = get_graph()
    result = await graph.ainvoke({"query": "What is the boiling point of water at sea level?"})
    assert "query_analysis" in result

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graph_preserves_original_query_in_final_state():
    graph = get_graph()
    result = await graph.ainvoke({"query": "Explain photosynthesis briefly."})
    assert result["query"] == "Explain photosynthesis briefly."

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_simple_query_skips_planner_node():
    graph = get_graph()
    result = await graph.ainvoke({"query": "What is the capital of Japan?"})
    assert "subtasks" not in result

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complex_query_routes_through_planner_node():
    graph = get_graph()
    result = await graph.ainvoke(
        {
            "query": (
                "Compare PostgreSQL, MySQL, and MongoDB for a high-write-throughput "
                "analytics workload, considering performance, scalability, and cost."
            )
        }
    )
    assert "subtasks" in result
    assert len(result["subtasks"]) >= 3

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graph_populates_retrieved_chunks_for_simple_query():
    _upload("graph_e2e_test1.txt", "Jupiter is the largest planet in the solar system. " * 5)
    graph = get_graph()
    result = await graph.ainvoke({"query": "What is the largest planet, according to my uploaded document?"})
    assert "retrieved_chunks" in result

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graph_populates_web_results_when_web_search_needed():
    graph = get_graph()
    result = await graph.ainvoke({"query": "What is the current version of Python?"})
    assert "web_results" in result
    assert len(result["web_results"]) > 0

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_query_requiring_neither_source_skips_both_retrieval_nodes():
    graph = get_graph()
    result = await graph.ainvoke(
        {"query": "If a tree falls in a forest and no one is around, does it make a sound? Answer briefly with your own reasoning, no research needed."}
    )
    analysis = result["query_analysis"]
    if not analysis["requires_private_search"] and not analysis["requires_web_search"]:
        assert "retrieved_chunks" not in result
        assert "web_results" not in result
    else:
        pytest.skip("LLM classified this query as needing search - not a useful run for this case.")


# --- Corrective loop end-to-end tests ---
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graph_produces_verification_result_when_retrieval_happens():
    _upload("verify_test1.txt", "The speed of light in a vacuum is approximately 299,792 kilometers per second.")
    graph = get_graph()
    result = await graph.ainvoke({"query": "According to my document, what is the speed of light?"})
    assert "verification_result" in result
    assert len(result["verification_result"]) > 0

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_rewrite_count_never_exceeds_max_retries():
    """Uses a query about a document that was never uploaded, over a
    private-only search - guaranteed to find weak/no evidence and
    trigger the full rewrite loop up to its bound. This exercises the
    real end-to-end loop, not just the isolated routing function."""
    from app.core.config import get_settings

    graph = get_graph()
    result = await graph.ainvoke(
        {"query": "According to my private documents, what is the exact serial number of my grandmother's antique clock?"}
    )
    assert result.get("rewrite_count", 0) <= get_settings().MAX_RETRIES

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graph_produces_citations_for_sufficient_evidence():
    _upload("citation_test1.txt", "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.")
    graph = get_graph()
    result = await graph.ainvoke({"query": "According to my document, when was the Eiffel Tower completed?"})
    assert "final_report" in result
    assert "citations" in result

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graph_citations_reference_real_uploaded_document():
    _upload("citation_test2.txt", "Mount Kilimanjaro is the highest mountain in Africa, standing at 5,895 meters.")
    graph = get_graph()
    result = await graph.ainvoke({"query": "According to my document, how tall is Mount Kilimanjaro?"})
    private_citations = [c for c in result.get("citations", []) if c["source_type"] == "private"]
    if private_citations:
        assert any(c["document_name"] == "citation_test2.txt" for c in private_citations)
    else:
        pytest.skip("No private citations produced - evidence may not have been judged sufficient this run.")