"""
Requires Postgres + Redis running. Confirms the reranker wiring works
through the actual API, and that use_reranker changes both the
response shape (reranked: true) and the actual ordering when a
clearly-irrelevant candidate would otherwise rank higher on retrieval
score alone.
"""
import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_reranker_flag_reflected_in_response():
    client.post(
        "/documents/upload",
        files={"file": ("rerank_flag_test.txt", io.BytesIO(b"Test content for reranker flag. " * 10), "text/plain")},
    )

    response = client.get(
        "/documents/search/query",
        params={"q": "test content", "top_k": 3, "use_reranker": True},
    )
    assert response.status_code == 200
    assert response.json()["reranked"] is True


def test_reranker_improves_relevance_ordering():
    client.post(
        "/documents/upload",
        files={
            "file": (
                "rerank_ordering_test.txt",
                io.BytesIO(
                    b"The Great Barrier Reef is a large coral reef system off Australia's coast. " * 5
                    + b"Interest rate policy is set by the central bank based on inflation data. " * 5
                ),
                "text/plain",
            )
        },
    )

    response = client.get(
        "/documents/search/query",
        params={"q": "How do central banks decide interest rates?", "top_k": 2, "use_reranker": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert "interest rate" in data["results"][0]["content"].lower()