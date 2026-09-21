import uuid

from fastapi import APIRouter

from app.agents.graph import get_graph
from app.core.exceptions import AppException
from app.schemas.research import CitationResponse, ResearchRequest, ResearchResponse

router = APIRouter(prefix="/research", tags=["research"])


@router.post("", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    research_id = str(uuid.uuid4())
    graph = get_graph()

    try:
        result = await graph.ainvoke({"query": request.query})
    except Exception as exc:
        raise AppException(f"Research pipeline failed: {exc}") from exc

    citations = [
        CitationResponse(
            claim=c["claim"], source_type=c["source_type"], document_name=c["document_name"],
            page=c["page"], section=c["section"], url=c["url"], chunk_id=c["chunk_id"],
        )
        for c in result.get("citations", [])
    ]

    return ResearchResponse(
        research_id=research_id,
        query=request.query,
        final_report=result.get("final_report", "No report generated."),
        citations=citations,
        query_type=result.get("query_analysis", {}).get("query_type", "unknown"),
        rewrite_count=result.get("rewrite_count", 0),
        run_metadata=result.get("run_metadata"),
    )