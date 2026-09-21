"""
Marked e2e: real Groq calls.
"""
import pytest

from app.evaluation.generation_metrics import evaluate_faithfulness, evaluate_relevancy

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_faithfulness_marks_supported_claim_as_faithful():
    result = await evaluate_faithfulness(
        answer="Paris is the capital of France.",
        evidence="Paris has been the capital of France since 987 AD.",
    )
    assert result.faithful is True


@pytest.mark.asyncio
async def test_faithfulness_marks_fabricated_claim_as_unfaithful():
    result = await evaluate_faithfulness(
        answer="Paris is the capital of France and has a population of exactly 50 million people.",
        evidence="Paris has been the capital of France since 987 AD.",
    )
    assert result.faithful is False
    assert len(result.unsupported_claims) > 0


@pytest.mark.asyncio
async def test_relevancy_marks_on_topic_answer_as_relevant():
    result = await evaluate_relevancy(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
    )
    assert result.relevant is True


@pytest.mark.asyncio
async def test_relevancy_marks_off_topic_answer_as_not_relevant():
    result = await evaluate_relevancy(
        question="What is the capital of France?",
        answer="Bananas are a good source of potassium.",
    )
    assert result.relevant is False