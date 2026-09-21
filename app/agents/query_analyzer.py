"""
First node in the research graph. Classifies the incoming query so
downstream nodes (planner, retrievers) know how to route it.
"""
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import QueryAnalysis, ResearchState
from app.core.observability import track_node
from app.llm.models import get_llm
from app.llm.prompts import QUERY_ANALYZER_SYSTEM_PROMPT
from app.llm.structured_outputs import QueryAnalysisResult

logger = logging.getLogger(__name__)


@track_node("query_analyzer")
async def query_analyzer_node(state: ResearchState) -> dict:
    query = state["query"]

    llm = get_llm().with_structured_output(QueryAnalysisResult)
    result: QueryAnalysisResult = await llm.ainvoke(
        [
            SystemMessage(content=QUERY_ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )

    analysis: QueryAnalysis = {
        "query_type": result.query_type,
        "requires_private_search": result.requires_private_search,
        "requires_web_search": result.requires_web_search,
        "requires_decomposition": result.requires_decomposition,
        "reason": result.reason,
    }

    logger.info(
        f"Query analyzed: type={analysis['query_type']}, "
        f"private={analysis['requires_private_search']}, "
        f"web={analysis['requires_web_search']}, "
        f"decompose={analysis['requires_decomposition']}"
    )

    return {"query_analysis": analysis}