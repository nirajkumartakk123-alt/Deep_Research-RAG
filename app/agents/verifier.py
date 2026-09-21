import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import ResearchState, TaskVerification
from app.core.observability import track_node
from app.llm.models import get_llm
from app.llm.prompts import VERIFIER_SYSTEM_PROMPT
from app.llm.structured_outputs import EvidenceSufficiency

logger = logging.getLogger(__name__)

MAX_EVIDENCE_SNIPPETS_PER_TASK = 10


def _gather_evidence(task: str, retrieved_chunks: list[dict], web_results: list[dict]) -> list[str]:
    snippets = [c["content"] for c in retrieved_chunks if c["task"] == task]
    snippets += [r["content"] for r in web_results if r["task"] == task]
    return snippets[:MAX_EVIDENCE_SNIPPETS_PER_TASK]


@track_node("verifier")
async def verifier_node(state: ResearchState) -> dict:
    tasks = state.get("subtasks") or [state["query"]]
    retrieved_chunks = state.get("retrieved_chunks", [])
    web_results = state.get("web_results", [])
    prior_by_task = {v["task"]: v for v in state.get("verification_result", [])}

    llm = get_llm().with_structured_output(EvidenceSufficiency)
    results: list[TaskVerification] = []

    for task in tasks:
        prior = prior_by_task.get(task)
        if prior is not None and prior["sufficient"]:
            results.append(prior)
            continue

        snippets = _gather_evidence(task, retrieved_chunks, web_results)

        if not snippets:
            results.append(
                {
                    "task": task,
                    "sufficient": False,
                    "confidence": 1.0,
                    "missing_evidence": "No evidence was retrieved for this task at all.",
                }
            )
            continue

        evidence_text = "\n\n".join(f"- {s}" for s in snippets)
        result: EvidenceSufficiency = await llm.ainvoke(
            [
                SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
                HumanMessage(content=f"Task: {task}\n\nEvidence:\n{evidence_text}"),
            ]
        )
        results.append(
            {
                "task": task,
                "sufficient": result.sufficient,
                "confidence": result.confidence,
                "missing_evidence": result.missing_evidence,
            }
        )

    sufficient_count = sum(1 for r in results if r["sufficient"])
    logger.info(f"Verification: {sufficient_count}/{len(results)} tasks sufficient")

    return {"verification_result": results}