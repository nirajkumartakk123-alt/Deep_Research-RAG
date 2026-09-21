import uuid
from datetime import datetime

from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str


class CitationResponse(BaseModel):
    claim: str
    source_type: str
    document_name: str | None
    page: int | None
    section: str | None
    url: str | None
    chunk_id: str | None


class ResearchResponse(BaseModel):
    research_id: str
    query: str
    final_report: str
    citations: list[CitationResponse]
    query_type: str
    rewrite_count: int
    run_metadata: dict | None = None