"""Tests for system health and pgvector telemetry."""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.schemas.health import DatabaseHealth


def test_health_endpoint_db_unavailable(client: TestClient):
    """When PostgreSQL is offline, health check must report honest degraded state without crashing."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert data["database"]["connected"] is False
    assert data["database"]["vector_extension"] is False
    assert data["database"]["error"] is not None
    # Verify no credentials leaked
    assert "password" not in data["database"]["error"].lower()
    assert "neuroquery_pass" not in data["database"]["error"]


def test_health_endpoint_db_healthy(client: TestClient):
    """When PostgreSQL and pgvector are both healthy, status should be 'healthy'."""
    with patch(
        "app.services.health.check_database_health",
        return_value=DatabaseHealth(
            connected=True,
            vector_extension=True,
            error=None,
        ),
    ):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["database"]["connected"] is True
        assert data["database"]["vector_extension"] is True
        assert data["database"]["error"] is None


def test_health_endpoint_pgvector_missing(client: TestClient):
    """When PostgreSQL is connected but pgvector extension is missing, status should be 'degraded'."""
    with patch(
        "app.services.health.check_database_health",
        return_value=DatabaseHealth(
            connected=True,
            vector_extension=False,
            error="pgvector extension is not enabled in database",
        ),
    ):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "degraded"
        assert data["database"]["connected"] is True
        assert data["database"]["vector_extension"] is False
        assert "pgvector extension is not enabled" in data["database"]["error"]
