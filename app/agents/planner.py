import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import ResearchState
from app.core.observability import track_node
from app.llm.models import get_llm
from app.llm.prompts import PLANNER_SYSTEM_PROMPT
from app.llm.structured_outputs import PlanResult

logger = logging.getLogger(__name__)


@track_node("planner")
async def planner_node(state: ResearchState) -> dict:
    query = state["query"]

    llm = get_llm().with_structured_output(PlanResult)
    result: PlanResult = await llm.ainvoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )

    logger.info(f"Query decomposed into {len(result.subtasks)} subtasks: {result.subtasks}")

    return {"subtasks": result.subtasks}