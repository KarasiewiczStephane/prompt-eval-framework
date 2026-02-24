"""Shared fixtures for the test suite."""

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.utils.database import Database
from tests.fixtures.mock_responses import MockAnthropicResponse, MockOpenAIResponse


@pytest.fixture()
def use_mock_api() -> bool:
    return os.getenv("USE_MOCK_API", "false").lower() == "true"


@pytest.fixture()
def mock_openai_client() -> AsyncMock:
    mock = AsyncMock()
    mock.chat.completions.create.return_value = MockOpenAIResponse()
    return mock


@pytest.fixture()
def mock_anthropic_client() -> AsyncMock:
    mock = AsyncMock()
    mock.messages.create.return_value = MockAnthropicResponse()
    return mock


@pytest.fixture()
def temp_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.duckdb")
