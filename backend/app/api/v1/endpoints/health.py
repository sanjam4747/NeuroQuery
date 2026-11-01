"""System and database health telemetry endpoint."""

from fastapi import APIRouter
from app.schemas.health import HealthCheckResponse
from app.services.health import get_system_health

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System Health & Telemetry",
    description="Probes API operational state, PostgreSQL connection, and pgvector extension availability.",
)
def get_health() -> HealthCheckResponse:
    return get_system_health()
