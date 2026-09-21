from app.agents.state import Citation, ResearchState
from app.core.observability import track_node_sync


@track_node_sync("citation_builder")
def citation_builder_node(state: ResearchState) -> dict:
    claims_with_evidence = state.get("claims_with_evidence", [])

    citations: list[Citation] = [
        {
            "claim": claim,
            "source_type": evidence["source_type"],
            "document_name": evidence["document_name"],
            "page": evidence["page"],
            "section": evidence["section"],
            "url": evidence["url"],
            "chunk_id": evidence["chunk_id"],
        }
        for claim, evidence in claims_with_evidence
    ]

    return {"citations": citations}