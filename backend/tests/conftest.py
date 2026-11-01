"""Pytest fixtures and configuration."""

import sys
import os
from typing import Generator
import pytest
from fastapi.testclient import TestClient

# Ensure backend and project root are in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(backend_dir)
for p in [backend_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Test client fixture using Starlette/FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client
