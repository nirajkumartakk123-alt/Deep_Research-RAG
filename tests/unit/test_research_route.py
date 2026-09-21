"""
Marked e2e: exercises the full LangGraph pipeline via HTTP.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

pytestmark = pytest.mark.e2e


def test_research_endpoint_returns_report_and_citations():
    response = client.post("/research", json={"query": "What is the boiling point of water?"})
    assert response.status_code == 200
    data = response.json()
    assert "final_report" in data
    assert "citations" in data
    assert "research_id" in data