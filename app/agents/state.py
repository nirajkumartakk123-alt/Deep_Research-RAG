"""
The single typed state threaded through every LangGraph node.
"""
from typing import TypedDict


class QueryAnalysis(TypedDict):
    query_type: str
    requires_private_search: bool
    requires_web_search: bool
    requires_decomposition: bool
    reason: str


class TaskVerification(TypedDict):
    task: str
    sufficient: bool
    confidence: float
    missing_evidence: str


class Citation(TypedDict):
    claim: str
    source_type: str  # "private" or "web"
    document_name: str | None
    page: int | None
    section: str | None
    url: str | None
    chunk_id: str | None


class ResearchState(TypedDict, total=False):
    # --- Input ---
    query: str

    # --- query_analyzer output ---
    query_analysis: QueryAnalysis

    # --- planner output ---
    subtasks: list[str]

    # --- retrieval outputs ---
    retrieved_chunks: list[dict]
    web_results: list[dict]

    # --- corrective RAG loop ---
    verification_result: list[TaskVerification]
    rewrite_count: int
    pending_retrieval_tasks: list[str]

    # --- Phase 9: synthesis + citations ---
    final_report: str
    # Intermediate handoff from synthesizer_node to citation_builder_node:
    # each entry is (claim_text, evidence_dict) where evidence_dict has
    # the real source metadata the claim was built from. citation_builder_node
    # consumes this to produce the final `citations` list below - kept
    # separate from `citations` itself so citation_builder_node's output
    # format can evolve independently of this internal handoff shape.
    claims_with_evidence: list[tuple[str, dict]]
    citations: list[Citation]

    # --- observability ---
    run_metadata: dict