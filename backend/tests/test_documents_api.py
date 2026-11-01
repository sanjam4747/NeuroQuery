"""Integration tests for Document API endpoints (/api/v1/documents)."""

import uuid
from fastapi.testclient import TestClient
from tests.conftest import make_docx, make_pdf, make_txt


def test_api_upload_pdf(client: TestClient):
    pdf_bytes = make_pdf(["NeuroQuery API upload test content on page 1."])
    response = client.post(
        "/api/v1/documents",
        files={"file": ("api_test.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()

    assert data["original_filename"] == "api_test.pdf"
    assert data["file_type"] == "pdf"
    assert data["status"] == "completed"
    assert data["total_pages"] == 1
    assert data["total_chunks"] >= 1
    # Security: internal absolute/relative storage path must never leak
    assert "storage_path" not in data


def test_api_upload_docx(client: TestClient):
    docx_bytes = make_docx(["Section 1 header.", "Detailed paragraph body."])
    response = client.post(
        "/api/v1/documents",
        files={"file": ("word_doc.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["file_type"] == "docx"
    assert data["status"] == "completed"
    assert data["total_pages"] is None  # Not fabricated


def test_api_upload_txt(client: TestClient):
    txt_bytes = make_txt("Plain text ingestion content for API endpoint test.")
    response = client.post(
        "/api/v1/documents",
        files={"file": ("test.txt", txt_bytes, "text/plain")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["file_type"] == "txt"
    assert data["status"] == "completed"


def test_api_upload_unsupported_format(client: TestClient):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("danger.sh", b"#!/bin/bash\necho bad", "application/x-sh")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_api_upload_empty_file(client: TestClient):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_list_and_get_document(client: TestClient):
    # Upload two documents
    client.post("/api/v1/documents", files={"file": ("doc1.txt", make_txt("Doc 1 content"), "text/plain")})
    res2 = client.post("/api/v1/documents", files={"file": ("doc2.txt", make_txt("Doc 2 content"), "text/plain")})
    doc2_id = res2.json()["id"]

    # List documents
    list_res = client.get("/api/v1/documents?skip=0&limit=10")
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["total"] >= 2
    assert len(data["items"]) >= 2

    # Get single document
    get_res = client.get(f"/api/v1/documents/{doc2_id}")
    assert get_res.status_code == 200
    assert get_res.json()["original_filename"] == "doc2.txt"


def test_api_get_document_chunks(client: TestClient):
    txt_content = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
    res = client.post("/api/v1/documents", files={"file": ("chunk_test.txt", make_txt(txt_content), "text/plain")})
    doc_id = res.json()["id"]

    chunks_res = client.get(f"/api/v1/documents/{doc_id}/chunks?skip=0&limit=50")
    assert chunks_res.status_code == 200
    chunk_data = chunks_res.json()
    assert chunk_data["total"] >= 1
    assert len(chunk_data["items"]) >= 1
    assert chunk_data["items"][0]["chunk_index"] == 0
    assert "text" in chunk_data["items"][0]


def test_api_delete_document(client: TestClient):
    res = client.post("/api/v1/documents", files={"file": ("to_delete.txt", make_txt("Content to be deleted"), "text/plain")})
    doc_id = res.json()["id"]

    # Delete
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200

    # Ensure document 404s after delete
    get_res = client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 404

    # Ensure chunks 404 after delete
    chunks_res = client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 404


def test_api_get_nonexistent_document(client: TestClient):
    random_id = uuid.uuid4()
    res = client.get(f"/api/v1/documents/{random_id}")
    assert res.status_code == 404
