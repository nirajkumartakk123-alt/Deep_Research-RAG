"""
Pure, DB-free, LLM-free tests - citation_builder_node has zero I/O,
so these run instantly and test the deterministic mapping directly.
"""
from app.agents.citation_builder import citation_builder_node


def test_citation_builder_maps_private_evidence_correctly():
    state = {
        "claims_with_evidence": [
            (
                "Paris is the capital of France.",
                {
                    "source_type": "private",
                    "document_name": "geography.txt",
                    "page": 3,
                    "section": "Capitals",
                    "url": None,
                    "chunk_id": "abc-123",
                },
            )
        ]
    }
    result = citation_builder_node(state)
    citation = result["citations"][0]
    assert citation["claim"] == "Paris is the capital of France."
    assert citation["source_type"] == "private"
    assert citation["document_name"] == "geography.txt"
    assert citation["page"] == 3
    assert citation["url"] is None


def test_citation_builder_maps_web_evidence_correctly():
    state = {
        "claims_with_evidence": [
            (
                "Python 3.13 was released in 2024.",
                {
                    "source_type": "web",
                    "document_name": None,
                    "page": None,
                    "section": None,
                    "url": "https://python.org/downloads",
                    "chunk_id": None,
                },
            )
        ]
    }
    result = citation_builder_node(state)
    citation = result["citations"][0]
    assert citation["source_type"] == "web"
    assert citation["url"] == "https://python.org/downloads"
    assert citation["document_name"] is None


def test_citation_builder_empty_claims_returns_empty_citations():
    result = citation_builder_node({"claims_with_evidence": []})
    assert result["citations"] == []


def test_citation_builder_handles_missing_key_gracefully():
    """No claims_with_evidence key at all (e.g. synthesizer produced
    nothing) should not crash - just return empty citations."""
    result = citation_builder_node({})
    assert result["citations"] == []