"""Tests for application root and versioned API info endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """GET / should return root application info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "NeuroQuery"
    assert "version" in data
    assert data["docs_url"] == "/docs"
    assert data["api_v1_url"] == "/api/v1"


def test_api_v1_root_endpoint(client: TestClient):
    """GET /api/v1/ should return API v1 metadata."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "NeuroQuery"
    assert data["status"] == "online"
    assert data["docs_url"] == "/docs"
    assert "version" in data
