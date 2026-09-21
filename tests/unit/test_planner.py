"""
Real Groq API calls, same reasoning as test_query_analyzer.py -
structured-output reliability and decomposition quality are exactly
what we need confidence in, which a mocked test can't tell us.
"""
import pytest

from app.agents.planner import planner_node


@pytest.mark.asyncio
async def test_planner_produces_multiple_subtasks_for_comparison():
    result = await planner_node(
        {
            "query": (
                "Compare AWS, GCP, and Azure for deploying a computer vision inference "
                "system, considering GPU availability, cost, and scalability."
            )
        }
    )
    subtasks = result["subtasks"]
    assert 3 <= len(subtasks) <= 9


@pytest.mark.asyncio
async def test_planner_subtasks_are_non_empty_strings():
    result = await planner_node({"query": "Compare Python and JavaScript for web backend development."})
    for subtask in result["subtasks"]:
        assert isinstance(subtask, str)
        assert len(subtask) > 0


@pytest.mark.asyncio
async def test_planner_subtasks_cover_multiple_items_in_comparison():
    """Loose but meaningful check: for a 3-way comparison, at least 2
    of the 3 compared items should be explicitly named somewhere
    across the generated subtasks - otherwise the decomposition isn't
    actually covering the comparison it was asked to cover."""
    result = await planner_node(
        {"query": "Compare AWS, GCP, and Azure for GPU-based machine learning inference."}
    )
    subtasks_text = " ".join(result["subtasks"]).lower()
    mentioned = sum(1 for name in ["aws", "gcp", "azure"] if name in subtasks_text)
    assert mentioned >= 2