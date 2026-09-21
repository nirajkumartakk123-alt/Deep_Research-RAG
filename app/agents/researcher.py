import logging

from app.agents.state import ResearchState
from app.core.config import get_settings
from app.core.observability import track_node
from app.database.connection import AsyncSessionLocal
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.services.web_search import web_search

logger = logging.getLogger(__name__)


async def _research_single_task(task: str) -> list[dict]:
    settings = get_settings()

    async with AsyncSessionLocal() as db:
        candidates = await hybrid_search(db, task, top_k=settings.TOP_K)
        reranked = rerank(task, candidates, top_k=settings.RERANK_TOP_K)

        return [
            {
                "task": task,
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "document_name": chunk.document.document_name,
                "content": chunk.content,
                "page": chunk.page,
                "section": chunk.section,
                "score": float(score),
            }
            for chunk, score in reranked
        ]


def _web_results_for_task(task: str) -> list[dict]:
    results = web_search(task, max_results=5)
    return [
        {"task": task, "title": r.title, "url": r.url, "content": r.content, "score": r.score}
        for r in results
    ]


@track_node("private_retriever")
async def private_retriever_node(state: ResearchState) -> dict:
    tasks = state.get("subtasks") or [state["query"]]

    all_chunks: list[dict] = []
    for task in tasks:
        all_chunks.extend(await _research_single_task(task))

    logger.info(f"Private retrieval: {len(tasks)} task(s) -> {len(all_chunks)} total chunks retrieved")
    return {"retrieved_chunks": all_chunks}


@track_node("web_researcher")
async def web_researcher_node(state: ResearchState) -> dict:
    tasks = state.get("subtasks") or [state["query"]]

    all_results: list[dict] = []
    for task in tasks:
        all_results.extend(_web_results_for_task(task))

    logger.info(f"Web research: {len(tasks)} task(s) -> {len(all_results)} total results retrieved")
    return {"web_results": all_results}


@track_node("targeted_retriever")
async def targeted_retriever_node(state: ResearchState) -> dict:
    tasks = state.get("pending_retrieval_tasks", [])
    analysis = state["query_analysis"]

    retrieved_chunks = list(state.get("retrieved_chunks", []))
    web_results = list(state.get("web_results", []))

    for task in tasks:
        if analysis["requires_private_search"]:
            retrieved_chunks.extend(await _research_single_task(task))
        if analysis["requires_web_search"]:
            web_results.extend(_web_results_for_task(task))

    logger.info(f"Targeted re-retrieval for {len(tasks)} rewritten task(s) complete")

    return {
        "retrieved_chunks": retrieved_chunks,
        "web_results": web_results,
        "pending_retrieval_tasks": [],
    }