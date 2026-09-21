import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import ResearchState
from app.core.observability import track_node
from app.llm.models import get_llm
from app.llm.prompts import QUERY_REWRITER_SYSTEM_PROMPT
from app.llm.structured_outputs import QueryRewriteResult

logger = logging.getLogger(__name__)


@track_node("query_rewriter")
async def query_rewriter_node(state: ResearchState) -> dict:
    verification = state.get("verification_result", [])
    insufficient = [v for v in verification if not v["sufficient"]]

    subtasks = list(state.get("subtasks") or [state["query"]])
    retrieved_chunks = list(state.get("retrieved_chunks", []))
    web_results = list(state.get("web_results", []))

    llm = get_llm().with_structured_output(QueryRewriteResult)
    pending: list[str] = []

    for v in insufficient:
        original_task = v["task"]

        result: QueryRewriteResult = await llm.ainvoke(
            [
                SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Original question: {original_task}\nWhy the previous search failed: {v['missing_evidence']}"
                ),
            ]
        )
        new_task = result.rewritten_query

        if original_task in subtasks:
            subtasks[subtasks.index(original_task)] = new_task

        retrieved_chunks = [c for c in retrieved_chunks if c["task"] != original_task]
        web_results = [r for r in web_results if r["task"] != original_task]

        pending.append(new_task)

    logger.info(f"Rewrote {len(insufficient)} task(s): {[v['task'] for v in insufficient]} -> {pending}")

    return {
        "subtasks": subtasks,
        "retrieved_chunks": retrieved_chunks,
        "web_results": web_results,
        "pending_retrieval_tasks": pending,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }