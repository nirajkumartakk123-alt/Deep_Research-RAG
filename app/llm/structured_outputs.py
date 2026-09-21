"""
Pydantic schemas for every LLM call that needs structured output.
"""
from pydantic import BaseModel, Field


class QueryAnalysisResult(BaseModel):
    query_type: str = Field(
        description=(
            "One of: 'simple_factual', 'comparative_research', "
            "'multi_part_research', 'opinion_or_recommendation'."
        )
    )
    requires_private_search: bool = Field(
        description="True if the user's private uploaded documents are likely relevant to answering this query."
    )
    requires_web_search: bool = Field(
        description="True if current/external information from the web is likely needed to answer this query well."
    )
    requires_decomposition: bool = Field(
        description="True if this query has multiple distinct parts that should be researched separately before combining into one answer."
    )
    reason: str = Field(description="A one-sentence explanation of why these classifications were chosen.")


class PlanResult(BaseModel):
    subtasks: list[str] = Field(
        description=(
            "3 to 9 focused, self-contained research sub-questions that together "
            "cover every distinct part of the original query."
        )
    )


class EvidenceSufficiency(BaseModel):
    sufficient: bool = Field(
        description="True if the evidence snippets contain enough information to construct a well-supported answer to the task."
    )
    confidence: float = Field(description="Confidence 0.0-1.0 in this sufficiency judgment.")
    missing_evidence: str = Field(
        description="If insufficient, what specific information is missing. Empty string if sufficient."
    )


class QueryRewriteResult(BaseModel):
    rewritten_query: str = Field(
        description="A reformulated, natural-language research question likely to retrieve better evidence."
    )


class ClaimWithSource(BaseModel):
    claim: str = Field(description="A single factual claim, stated in your own words based on the evidence.")
    evidence_index: int = Field(
        description=(
            "The index (starting at 0) of the SPECIFIC evidence snippet from the "
            "provided numbered list that directly supports this claim. Must be an "
            "index that actually appears in the evidence list - never invent an "
            "index or cite evidence that wasn't provided."
        )
    )


class TaskSynthesis(BaseModel):
    claims: list[ClaimWithSource] = Field(
        description=(
            "2 to 6 distinct factual claims that answer the task, each traceable to "
            "one specific evidence snippet via evidence_index. Do not include any "
            "claim that isn't directly supported by one of the provided snippets."
        )
    )

class FaithfulnessJudgment(BaseModel):
    faithful: bool = Field(
        description="True if every claim in the generated answer is supported by the provided evidence/context, with no fabricated or unsupported information."
    )
    unsupported_claims: list[str] = Field(
        description="Specific claims from the generated answer that are NOT supported by the evidence. Empty list if faithful."
    )


class RelevancyJudgment(BaseModel):
    relevant: bool = Field(
        description="True if the generated answer actually addresses what the question asked, even if incomplete."
    )
    relevancy_score: float = Field(
        description="0.0-1.0 score for how directly and completely the answer addresses the question."
    )