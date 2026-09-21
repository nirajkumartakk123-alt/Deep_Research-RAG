"""
Requires Postgres + Redis running - these tests upload real content
and exercise all three search modes through the actual API.
"""
import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _upload(filename: str, content: str):
    return client.post(
        "/documents/upload", files={"file": (filename, io.BytesIO(content.encode()), "text/plain")}
    )


def test_bm25_mode_finds_exact_keyword_match():
    _upload("bm25_test.txt", "The quokka is a small marsupial found in Western Australia. " * 5)

    response = client.get("/documents/search/query", params={"q": "quokka", "top_k": 3, "mode": "bm25"})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "bm25"
    assert len(data["results"]) > 0
    assert "quokka" in data["results"][0]["content"].lower()


def test_vector_mode_finds_semantic_match_without_exact_keyword():
    _upload(
        "vector_test.txt",
        "pgvector adds similarity search capabilities directly inside PostgreSQL. " * 5,
    )

    response = client.get(
        "/documents/search/query",
        params={"q": "How can I do embedding search in a SQL database?", "top_k": 3, "mode": "vector"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "vector"
    assert len(data["results"]) > 0


def test_hybrid_mode_returns_results_combining_both_signals():
    _upload("hybrid_test.txt", "Redis is an in-memory data store used for caching and message brokering. " * 5)

    response = client.get(
        "/documents/search/query", params={"q": "in-memory caching system", "top_k": 3, "mode": "hybrid"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert len(data["results"]) > 0


def test_default_mode_is_hybrid():
    _upload("default_mode_test.txt", "Kubernetes orchestrates containerized applications at scale. " * 5)

    response = client.get("/documents/search/query", params={"q": "container orchestration", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["mode"] == "hybrid"


def test_bm25_finds_rare_exact_term_that_vector_search_might_miss():
    """BM25's strength: exact rare-term matching. An invented, made-up
    product name has no semantic meaning for an embedding model to
    latch onto, but BM25 will match it exactly if present."""
    _upload("rare_term_test.txt", "The Zylobrex-9000 requires a firmware update before first use. " * 5)

    response = client.get(
        "/documents/search/query", params={"q": "Zylobrex-9000", "top_k": 3, "mode": "bm25"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert "zylobrex" in data["results"][0]["content"].lower()