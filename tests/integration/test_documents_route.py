"""
Requires Postgres + Redis running (docker compose up -d postgres redis)
since uploads now persist to the real database.
"""
import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_txt_document_persists_and_returns_chunks():
    file_content = b"This is a simple test document for ingestion. " * 30
    response = client.post(
        "/documents/upload",
        files={"file": ("sample.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_type"] == "txt"
    assert data["total_chunks"] > 0
    assert data["already_existed"] is False
    assert "document_id" in data


def test_uploading_same_content_twice_is_detected_as_duplicate():
    file_content = b"Duplicate detection test content. " * 20

    first = client.post(
        "/documents/upload", files={"file": ("dup.txt", io.BytesIO(file_content), "text/plain")}
    )
    second = client.post(
        "/documents/upload", files={"file": ("dup_renamed.txt", io.BytesIO(file_content), "text/plain")}
    )

    assert first.json()["already_existed"] is False
    assert second.json()["already_existed"] is True
    assert first.json()["document_id"] == second.json()["document_id"]


def test_upload_unsupported_type_returns_422():
    response = client.post(
        "/documents/upload",
        files={"file": ("sample.exe", io.BytesIO(b"binary junk"), "application/octet-stream")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation_error"


def test_upload_empty_file_returns_422():
    response = client.post(
        "/documents/upload", files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "empty_document"


def test_list_documents_returns_uploaded_document():
    file_content = b"Listing test content for phase 3. " * 20
    upload = client.post(
        "/documents/upload", files={"file": ("list_test.txt", io.BytesIO(file_content), "text/plain")}
    )
    doc_id = upload.json()["document_id"]

    response = client.get("/documents")
    assert response.status_code == 200
    ids = [d["id"] for d in response.json()]
    assert doc_id in ids


def test_get_document_by_id():
    file_content = b"Get by id test content. " * 20
    upload = client.post(
        "/documents/upload", files={"file": ("get_test.txt", io.BytesIO(file_content), "text/plain")}
    )
    doc_id = upload.json()["document_id"]

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["id"] == doc_id


def test_get_nonexistent_document_returns_404():
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/documents/{fake_id}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "not_found"


def test_delete_document_removes_it():
    file_content = b"Delete test content. " * 20
    upload = client.post(
        "/documents/upload", files={"file": ("delete_test.txt", io.BytesIO(file_content), "text/plain")}
    )
    doc_id = upload.json()["document_id"]

    delete_response = client.delete(f"/documents/{doc_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    get_response = client.get(f"/documents/{doc_id}")
    assert get_response.status_code == 404


def test_search_returns_relevant_chunk():
    file_content = (
        b"The Eiffel Tower is located in Paris, France. " * 5
        + b"Quarterly financial results exceeded analyst expectations. " * 5
    )
    client.post("/documents/upload", files={"file": ("search_test.txt", io.BytesIO(file_content), "text/plain")})

    response = client.get("/documents/search/query", params={"q": "Where is the Eiffel Tower?", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert "Eiffel Tower" in data["results"][0]["content"]