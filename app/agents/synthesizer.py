import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import ResearchState
from app.core.observability import track_node
from app.llm.models import get_llm
from app.llm.prompts import SYNTHESIZER_SYSTEM_PROMPT
from app.llm.structured_outputs import TaskSynthesis

logger = logging.getLogger(__name__)

MAX_EVIDENCE_SNIPPETS_PER_TASK = 10


def _gather_evidence_for_synthesis(task: str, retrieved_chunks: list[dict], web_results: list[dict]) -> list[dict]:
    evidence = []
    for c in retrieved_chunks:
        if c["task"] == task:
            evidence.append(
                {
                    "content": c["content"],
                    "source_type": "private",
                    "document_name": c["document_name"],
                    "page": c["page"],
                    "section": c["section"],
                    "url": None,
                    "chunk_id": c["chunk_id"],
                }
            )
    for r in web_results:
        if r["task"] == task:
            evidence.append(
                {
                    "content": r["content"],
                    "source_type": "web",
                    "document_name": None,
                    "page": None,
                    "section": None,
                    "url": r["url"],
                    "chunk_id": None,
                }
            )
    return evidence[:MAX_EVIDENCE_SNIPPETS_PER_TASK]


@track_node("synthesizer")
async def synthesizer_node(state: ResearchState) -> dict:
    verification = state.get("verification_result", [])
    sufficient_tasks = [v["task"] for v in verification if v["sufficient"]]

    retrieved_chunks = state.get("retrieved_chunks", [])
    web_results = state.get("web_results", [])

    llm = get_llm(temperature=0.3).with_structured_output(TaskSynthesis)

    claims_with_evidence: list[tuple[str, dict]] = []
    report_sections: list[str] = []

    for task in sufficient_tasks:
        evidence = _gather_evidence_for_synthesis(task, retrieved_chunks, web_results)
        if not evidence:
            continue

        numbered_evidence = "\n".join(f"[{i}] {e['content']}" for i, e in enumerate(evidence))
        result: TaskSynthesis = await llm.ainvoke(
            [
                SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
                HumanMessage(content=f"Task: {task}\n\nEvidence:\n{numbered_evidence}"),
            ]
        )

        task_claims = []
        for claim in result.claims:
            if 0 <= claim.evidence_index < len(evidence):
                claims_with_evidence.append((claim.claim, evidence[claim.evidence_index]))
                task_claims.append(claim.claim)
            else:
                logger.warning(
                    f"Dropped claim with out-of-range evidence_index "
                    f"{claim.evidence_index} (evidence list had {len(evidence)} items) for task: {task}"
                )

        if task_claims:
            report_sections.append(f"**{task}**\n" + "\n".join(f"- {c}" for c in task_claims))

    insufficient_tasks = [v["task"] for v in verification if not v["sufficient"]]
    for task in insufficient_tasks:
        report_sections.append(f"**{task}**\n- Insufficient evidence was found to answer this question.")

    final_report = "\n\n".join(report_sections) if report_sections else "No evidence was available to answer this query."

    logger.info(
        f"Synthesis: {len(claims_with_evidence)} claims produced across "
        f"{len(sufficient_tasks)} sufficient task(s), {len(insufficient_tasks)} task(s) marked insufficient"
    )

    return {"final_report": final_report, "claims_with_evidence": claims_with_evidence}