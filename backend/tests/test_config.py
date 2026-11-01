"""Tests for centralized configuration and settings."""

from app.core.config import Settings


def test_default_settings():
    """Verify default application settings are properly initialized."""
    s = Settings()
    assert s.APP_NAME == "NeuroQuery"
    assert s.APP_VERSION == "0.1.0"
    assert s.API_V1_STR == "/api/v1"
    assert isinstance(s.BACKEND_CORS_ORIGINS, list)
    assert len(s.BACKEND_CORS_ORIGINS) > 0


def test_cors_origins_parsing():
    """Verify CORS origins string parsing."""
    # Comma-separated string parsing
    s = Settings(BACKEND_CORS_ORIGINS="http://localhost:3000,http://example.com")
    assert s.BACKEND_CORS_ORIGINS == ["http://localhost:3000", "http://example.com"]

    # List of strings
    s2 = Settings(BACKEND_CORS_ORIGINS=["http://test.local"])
    assert s2.BACKEND_CORS_ORIGINS == ["http://test.local"]


def test_database_url_masking():
    """Ensure database password is obfuscated in sanitized_database_url."""
    s = Settings(
        POSTGRES_USER="testuser",
        POSTGRES_PASSWORD="SuperSecretPassword123",
        POSTGRES_SERVER="db.internal",
        POSTGRES_PORT=5432,
        POSTGRES_DB="testdb",
    )
    assert "SuperSecretPassword123" in s.sync_database_url
    assert "SuperSecretPassword123" not in s.sanitized_database_url
    assert "testuser:***@db.internal:5432/testdb" in s.sanitized_database_url
