"""
LangGraph StateGraph construction.

Phase 9 adds: once the corrective loop's verifier reaches "done"
(either all-sufficient or MAX_RETRIES exhausted), the graph proceeds
to synthesizer_node (extract claims from verified evidence) then
citation_builder_node (deterministic claim -> source mapping) -> END.
"""
from langgraph.graph import END, StateGraph

from app.agents.citation_builder import citation_builder_node
from app.agents.planner import planner_node
from app.agents.query_analyzer import query_analyzer_node
from app.agents.query_rewriter import query_rewriter_node
from app.agents.researcher import private_retriever_node, targeted_retriever_node, web_researcher_node
from app.agents.state import ResearchState
from app.agents.synthesizer import synthesizer_node
from app.agents.verifier import verifier_node
from app.core.config import get_settings


def _retrieval_branch(analysis: dict) -> str:
    if analysis["requires_private_search"]:
        return "run_private"
    if analysis["requires_web_search"]:
        return "run_web_only"
    return "skip_all_retrieval"


def _route_after_analysis(state: ResearchState) -> str:
    analysis = state["query_analysis"]
    if analysis["requires_decomposition"]:
        return "needs_planning"
    return _retrieval_branch(analysis)


def _route_after_planning(state: ResearchState) -> str:
    return _retrieval_branch(state["query_analysis"])


def _route_web_research(state: ResearchState) -> str:
    if state["query_analysis"]["requires_web_search"]:
        return "run_web"
    return "skip_web"


def _route_after_verification(state: ResearchState) -> str:
    settings = get_settings()
    verification = state.get("verification_result", [])

    all_sufficient = all(v["sufficient"] for v in verification)
    if all_sufficient:
        return "done"

    rewrite_count = state.get("rewrite_count", 0)
    if rewrite_count >= settings.MAX_RETRIES:
        return "done"

    return "needs_rewrite"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("private_retriever", private_retriever_node)
    graph.add_node("web_researcher", web_researcher_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("targeted_retriever", targeted_retriever_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("citation_builder", citation_builder_node)

    graph.set_entry_point("query_analyzer")

    graph.add_conditional_edges(
        "query_analyzer",
        _route_after_analysis,
        {
            "needs_planning": "planner",
            "run_private": "private_retriever",
            "run_web_only": "web_researcher",
            "skip_all_retrieval": END,
        },
    )
    graph.add_conditional_edges(
        "planner",
        _route_after_planning,
        {
            "run_private": "private_retriever",
            "run_web_only": "web_researcher",
            "skip_all_retrieval": END,
        },
    )
    graph.add_conditional_edges(
        "private_retriever",
        _route_web_research,
        {
            "run_web": "web_researcher",
            "skip_web": "verifier",
        },
    )
    graph.add_edge("web_researcher", "verifier")

    graph.add_conditional_edges(
        "verifier",
        _route_after_verification,
        {
            "done": "synthesizer",
            "needs_rewrite": "query_rewriter",
        },
    )
    graph.add_edge("query_rewriter", "targeted_retriever")
    graph.add_edge("targeted_retriever", "verifier")

    graph.add_edge("synthesizer", "citation_builder")
    graph.add_edge("citation_builder", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph