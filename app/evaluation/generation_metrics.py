"""
LLM-as-judge generation-quality metrics: Faithfulness and Answer
Relevancy. Implemented directly with our existing get_llm() +
with_structured_output() infrastructure - same pattern as verifier.py
and synthesizer.py - rather than RAGAS, after RAGAS proved to have
irreconcilable langchain-core version conflicts with the rest of this
project (see Phase 10 design notes / README limitations section).

This is a deliberate, explainable tradeoff: these custom metrics are
simpler than RAGAS's implementations and have not been validated
against RAGAS's own numbers on a shared benchmark - they measure the
same underlying concepts (does the answer fabricate anything beyond
its evidence; does the answer address the question) but via a
single, directly-inspectable LLM prompt rather than RAGAS's internal
(and, for this project, unusable) pipeline.
"""
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.models import get_llm
from app.llm.structured_outputs import FaithfulnessJudgment, RelevancyJudgment

logger = logging.getLogger(__name__)

FAITHFULNESS_SYSTEM_PROMPT = """You are evaluating whether a generated answer is faithful to its supporting evidence.

Given the evidence that was used to generate an answer, and the answer itself, judge whether every claim in the answer is actually supported by the evidence - with no fabricated, invented, or unsupported information.

Respond only via the required structured schema."""

RELEVANCY_SYSTEM_PROMPT = """You are evaluating whether a generated answer actually addresses the question it was meant to answer.

Given a question and a generated answer, judge whether the answer directly and adequately addresses what was asked - regardless of whether the answer is fully correct, just whether it's on-topic and responsive.

Respond only via the required structured schema."""


async def evaluate_faithfulness(answer: str, evidence: str) -> FaithfulnessJudgment:
    llm = get_llm().with_structured_output(FaithfulnessJudgment)
    return await llm.ainvoke(
        [
            SystemMessage(content=FAITHFULNESS_SYSTEM_PROMPT),
            HumanMessage(content=f"Evidence:\n{evidence}\n\nGenerated answer:\n{answer}"),
        ]
    )


async def evaluate_relevancy(question: str, answer: str) -> RelevancyJudgment:
    llm = get_llm().with_structured_output(RelevancyJudgment)
    return await llm.ainvoke(
        [
            SystemMessage(content=RELEVANCY_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nGenerated answer:\n{answer}"),
        ]
    )