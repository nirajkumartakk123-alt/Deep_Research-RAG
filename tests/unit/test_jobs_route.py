"""
No real Celery worker needed for these - AsyncResult against a
job_id that was never submitted just returns PENDING (Celery's
default for unknown task IDs), which is enough to test the route
shape without running a full worker in the test suite.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_job_status_for_unknown_id_returns_pending():
    response = client.get("/jobs/nonexistent-job-id")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "nonexistent-job-id"
    assert data["status"] == "PENDING"


def test_job_status_response_shape():
    response = client.get("/jobs/some-id")
    data = response.json()
    assert "job_id" in data
    assert "status" in data