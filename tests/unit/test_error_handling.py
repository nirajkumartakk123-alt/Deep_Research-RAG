from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_malformed_request_body_returns_consistent_error_envelope():
    response = client.post("/research", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "validation_error"
    assert "message" in data


def test_nonexistent_document_returns_consistent_error_envelope():
    response = client.get("/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "not_found"