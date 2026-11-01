"""API root endpoint providing basic service information."""

from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import ApiRootInfo

router = APIRouter()


@router.get(
    "/",
    response_model=ApiRootInfo,
    summary="API v1 Information",
    description="Returns metadata and available documentation links for NeuroQuery API v1.",
)
def get_api_root_info() -> ApiRootInfo:
    return ApiRootInfo(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        status="online",
    )
