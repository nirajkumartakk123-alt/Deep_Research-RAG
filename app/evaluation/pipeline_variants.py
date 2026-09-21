"""
Wraps each pipeline configuration behind a single async interface so
evaluate.py can iterate over all variants uniformly. Variants 1-3 are
retrieval-only (no LLM calls at all - see Phase 10 design notes on
free-tier cost). Variants 4-5 run the real LangGraph agentic pipeline
and DO incur real Groq token cost.
"""
from dataclasses import dataclass

from app.core.config import get_settings
from app.database.connection import AsyncSessionLocal
from app.retrieval.bm25 import bm25_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.retrieval.vector_search import vector_search


@dataclass
class VariantResult:
    ranked_document_names: list[str]
    generated_answer: str | None
    evidence_text: str | None


async def run_baseline_vector(question: str) -> VariantResult:
    """Variant 1: dense vector search only, no rerank."""
    settings = get_settings()
    async with AsyncSessionLocal() as db:
        results = await vector_search(db, question, top_k=settings.TOP_K)
    names = [chunk.document.document_name for chunk, _ in results]
    return VariantResult(ranked_document_names=names, generated_answer=None, evidence_text=None)


async def run_hybrid(question: str) -> VariantResult:
    """Variant 2: hybrid (dense + BM25 via RRF), no rerank."""
    settings = get_settings()
    async with AsyncSessionLocal() as db:
        results = await hybrid_search(db, question, top_k=settings.TOP_K)
    names = [chunk.document.document_name for chunk, _ in results]
    return VariantResult(ranked_document_names=names, generated_answer=None, evidence_text=None)


async def run_hybrid_reranked(question: str) -> VariantResult:
    """Variant 3: hybrid retrieval, then cross-encoder reranking."""
    settings = get_settings()
    async with AsyncSessionLocal() as db:
        candidates = await hybrid_search(db, question, top_k=settings.TOP_K)
    reranked = rerank(question, candidates, top_k=settings.RERANK_TOP_K)
    names = [chunk.document.document_name for chunk, _ in reranked]
    return VariantResult(ranked_document_names=names, generated_answer=None, evidence_text=None)


async def run_agentic_no_correction(question: str) -> VariantResult:
    """Variant 4: full agentic graph, but bypassing the corrective
    loop entirely - runs query_analyzer, planner (if needed),
    retrieval, and synthesizer directly on whatever evidence retrieval
    returns, WITHOUT verification or rewriting. Built as a direct
    sequence of node calls rather than a separate graph, since
    reusing the existing nodes (not duplicating their logic) is what
    actually matters here - a second parallel graph definition would
    risk drifting from the real one over time."""
    from app.agents.query_analyzer import query_analyzer_node
    from app.agents.planner import planner_node
    from app.agents.researcher import private_retriever_node, web_researcher_node
    from app.agents.synthesizer import synthesizer_node

    state: dict = {"query": question}
    state.update(await query_analyzer_node(state))

    if state["query_analysis"]["requires_decomposition"]:
        state.update(await planner_node(state))

    if state["query_analysis"]["requires_private_search"]:
        state.update(await private_retriever_node(state))
    if state["query_analysis"]["requires_web_search"]:
        state.update(await web_researcher_node(state))

    # No verifier - fabricate an all-sufficient verification_result so
    # synthesizer_node treats every task as ready to synthesize,
    # matching this variant's "no correction loop" definition.
    tasks = state.get("subtasks") or [question]
    state["verification_result"] = [
        {"task": t, "sufficient": True, "confidence": 1.0, "missing_evidence": ""} for t in tasks
    ]
    state.update(await synthesizer_node(state))

    names = [c["document_name"] for c in state.get("retrieved_chunks", []) if c.get("document_name")]
    evidence_text = "\n".join(c["content"] for c in state.get("retrieved_chunks", []))
    return VariantResult(
        ranked_document_names=names,
        generated_answer=state.get("final_report"),
        evidence_text=evidence_text or None,
    )


async def run_corrective_agentic(question: str) -> VariantResult:
    """Variant 5: the full production graph, corrective loop included."""
    from app.agents.graph import get_graph

    graph = get_graph()
    result = await graph.ainvoke({"query": question})

    names = [c["document_name"] for c in result.get("retrieved_chunks", []) if c.get("document_name")]
    evidence_text = "\n".join(c["content"] for c in result.get("retrieved_chunks", []))
    return VariantResult(
        ranked_document_names=names,
        generated_answer=result.get("final_report"),
        evidence_text=evidence_text or None,
    )


VARIANTS = {
    "baseline_vector": run_baseline_vector,
    "hybrid": run_hybrid,
    "hybrid_reranked": run_hybrid_reranked,
    "agentic_no_correction": run_agentic_no_correction,
    "corrective_agentic": run_corrective_agentic,
}