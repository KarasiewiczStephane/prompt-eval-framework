"""Tests for multi-model runner with mocked API clients."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.model_runner import (
    AnthropicRunner,
    ModelResponse,
    MultiModelRunner,
    OpenAIRunner,
    safe_run,
)
from src.prompts.template_manager import ModelConfig
from src.utils.config import Config

MESSAGES = [
    {"role": "system", "content": "Be helpful"},
    {"role": "user", "content": "Hello"},
]
CONFIG = ModelConfig(temperature=0.5, max_tokens=100)


def _make_openai_response() -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = "Hi there!"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model_dump.return_value = {"id": "mock"}
    return resp


def _make_anthropic_response() -> MagicMock:
    """Build a mock Anthropic Message response."""
    block = MagicMock()
    block.text = "Hello from Claude!"
    usage = MagicMock()
    usage.input_tokens = 12
    usage.output_tokens = 6
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    resp.model_dump.return_value = {"id": "mock-ant"}
    return resp


class TestOpenAIRunner:
    """OpenAIRunner with a mocked async client."""

    @pytest.mark.asyncio
    async def test_run_returns_response(self) -> None:
        runner = OpenAIRunner(api_key="fake-key")
        runner.client = MagicMock()
        runner.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )

        result = await runner.run(MESSAGES, CONFIG, "gpt-4")

        assert isinstance(result, ModelResponse)
        assert result.content == "Hi there!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model == "gpt-4-turbo-preview"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_model_alias_resolution(self) -> None:
        runner = OpenAIRunner(api_key="fake-key")
        runner.client = MagicMock()
        runner.client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )

        result = await runner.run(MESSAGES, CONFIG, "gpt-3.5")
        assert result.model == "gpt-3.5-turbo"


class TestAnthropicRunner:
    """AnthropicRunner with a mocked async client."""

    @pytest.mark.asyncio
    async def test_run_returns_response(self) -> None:
        runner = AnthropicRunner(api_key="fake-key")
        runner.client = MagicMock()
        runner.client.messages.create = AsyncMock(
            return_value=_make_anthropic_response()
        )

        result = await runner.run(MESSAGES, CONFIG, "claude-sonnet")

        assert isinstance(result, ModelResponse)
        assert result.content == "Hello from Claude!"
        assert result.input_tokens == 12
        assert result.output_tokens == 6
        assert result.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_extracts_system_message(self) -> None:
        runner = AnthropicRunner(api_key="fake-key")
        runner.client = MagicMock()
        create_mock = AsyncMock(return_value=_make_anthropic_response())
        runner.client.messages.create = create_mock

        await runner.run(MESSAGES, CONFIG, "haiku")

        _, kwargs = create_mock.call_args
        assert kwargs["system"] == "Be helpful"
        assert all(m["role"] != "system" for m in kwargs["messages"])


class TestMultiModelRunner:
    """MultiModelRunner registration and dispatch."""

    def test_registers_openai_models(self) -> None:
        cfg = Config(openai_api_key="k1")
        runner = MultiModelRunner(cfg)
        assert "gpt-4" in runner.available_models
        assert "gpt-3.5-turbo" in runner.available_models

    def test_registers_anthropic_models(self) -> None:
        cfg = Config(anthropic_api_key="k2")
        runner = MultiModelRunner(cfg)
        assert "claude-sonnet" in runner.available_models
        assert "haiku" in runner.available_models

    def test_no_keys_empty_models(self) -> None:
        cfg = Config()
        runner = MultiModelRunner(cfg)
        assert runner.available_models == []

    @pytest.mark.asyncio
    async def test_run_single_unknown_model_raises(self) -> None:
        cfg = Config()
        runner = MultiModelRunner(cfg)
        with pytest.raises(ValueError, match="not available"):
            await runner.run_single(MESSAGES, CONFIG, "unknown-model")

    @pytest.mark.asyncio
    async def test_run_parallel(self) -> None:
        cfg = Config(openai_api_key="k1")
        runner = MultiModelRunner(cfg)

        mock_resp = ModelResponse(
            content="ok",
            input_tokens=5,
            output_tokens=3,
            latency_ms=100,
            model="gpt-4-turbo-preview",
        )

        with patch.object(runner, "run_single", new=AsyncMock(return_value=mock_resp)):
            results = await runner.run_parallel(MESSAGES, CONFIG, ["gpt-4", "gpt-3.5"])

        assert "gpt-4" in results
        assert "gpt-3.5" in results
        assert results["gpt-4"].content == "ok"

    @pytest.mark.asyncio
    async def test_run_consistency_check(self) -> None:
        cfg = Config(openai_api_key="k1")
        runner = MultiModelRunner(cfg)

        mock_resp = ModelResponse(
            content="ok",
            input_tokens=5,
            output_tokens=3,
            latency_ms=100,
            model="gpt-4-turbo-preview",
        )

        with patch.object(runner, "run_single", new=AsyncMock(return_value=mock_resp)):
            results = await runner.run_consistency_check(MESSAGES, CONFIG, "gpt-4", n=3)

        assert len(results) == 3


class TestSafeRun:
    """safe_run should catch timeouts and exceptions."""

    @pytest.mark.asyncio
    async def test_returns_response_on_success(self) -> None:
        mock_runner = MagicMock()
        expected = ModelResponse(
            content="hi",
            input_tokens=1,
            output_tokens=1,
            latency_ms=10,
            model="test",
        )
        mock_runner.run = AsyncMock(return_value=expected)

        result = await safe_run(mock_runner, MESSAGES, CONFIG, "test")
        assert isinstance(result, ModelResponse)

    @pytest.mark.asyncio
    async def test_returns_exception_on_error(self) -> None:
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=RuntimeError("API error"))

        result = await safe_run(mock_runner, MESSAGES, CONFIG, "test")
        assert isinstance(result, RuntimeError)

    @pytest.mark.asyncio
    async def test_returns_timeout_error(self) -> None:
        mock_runner = MagicMock()

        async def slow_run(*args, **kwargs):
            import asyncio

            await asyncio.sleep(10)

        mock_runner.run = slow_run

        result = await safe_run(mock_runner, MESSAGES, CONFIG, "test", timeout=0.01)
        assert isinstance(result, TimeoutError)
