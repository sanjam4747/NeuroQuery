"""Pydantic schemas for health probes and API telemetry."""

from typing import Optional
from pydantic import BaseModel, Field


class DatabaseHealth(BaseModel):
    """Database connectivity and extension status."""
    connected: bool = Field(..., description="True if PostgreSQL database is reachable.")
    vector_extension: bool = Field(..., description="True if pgvector extension is installed and available.")
    error: Optional[str] = Field(None, description="Sanitized connection diagnostic message if unavailable.")


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall service status: 'healthy' or 'degraded'.")
    version: str = Field(..., description="Application semantic version.")
    environment: str = Field(..., description="Current environment mode (development/production).")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of check.")
    database: DatabaseHealth = Field(..., description="Database health metrics.")


class ApiRootInfo(BaseModel):
    """Basic API information returned from root endpoints."""
    name: str = Field(..., description="Application name.")
    version: str = Field(..., description="API Version.")
    docs_url: str = Field(..., description="Interactive Swagger documentation URL.")
    status: str = Field(..., description="API operational status.")
