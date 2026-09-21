"""
Marked e2e: every test here makes a real Groq API call via
query_analyzer_node. Excluded from the default `pytest` run due to
free-tier TPM limits - run explicitly with `pytest -m e2e`.
"""
import pytest

from app.agents.query_analyzer import query_analyzer_node

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_simple_factual_query_does_not_require_decomposition():
    result = await query_analyzer_node({"query": "What year was PostgreSQL first released?"})
    analysis = result["query_analysis"]
    assert analysis["requires_decomposition"] is False


@pytest.mark.asyncio
async def test_comparative_query_requires_decomposition():
    result = await query_analyzer_node(
        {
            "query": (
                "Compare AWS, GCP, and Azure for deploying a computer vision inference "
                "system, considering GPU availability, cost, and scalability."
            )
        }
    )
    analysis = result["query_analysis"]
    assert analysis["requires_decomposition"] is True
    assert analysis["query_type"] in ("comparative_research", "multi_part_research")


@pytest.mark.asyncio
async def test_query_analysis_includes_a_reason():
    result = await query_analyzer_node({"query": "What is the capital of France?"})
    assert len(result["query_analysis"]["reason"]) > 0