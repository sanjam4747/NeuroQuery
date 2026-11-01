"""Centralized application configuration management via pydantic-settings."""

from typing import Any, List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application Metadata
    APP_NAME: str = "NeuroQuery"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v  # Handled by pydantic json decoder if string starts with [
        raise ValueError(v)

    # Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "neuroquery_user"
    POSTGRES_PASSWORD: str = "neuroquery_pass"
    POSTGRES_DB: str = "neuroquery"
    DATABASE_URL: Union[str, None] = None

    # Connection pooling and probe settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_PRE_PING: bool = True
    DB_CONNECT_TIMEOUT_SECONDS: float = 2.0

    @property
    def sync_database_url(self) -> str:
        """Return the SQLAlchemy connection URI."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def sanitized_database_url(self) -> str:
        """Return database URI with masked credentials for safe logging and diagnostics."""
        url = self.sync_database_url
        if "@" in url and "://" in url:
            scheme_and_auth, host_and_db = url.split("@", 1)
            scheme, auth = scheme_and_auth.split("://", 1)
            user = auth.split(":", 1)[0] if ":" in auth else auth
            return f"{scheme}://{user}:***@{host_and_db}"
        return url


settings = Settings()
