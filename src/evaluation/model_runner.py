"""Multi-model async runner supporting OpenAI and Anthropic APIs.

Provides base classes and concrete runners for parallel prompt
evaluation across multiple LLM providers.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import anthropic
import openai

from src.prompts.template_manager import ModelConfig
from src.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a single model invocation.

    Attributes:
        content: The generated text.
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.
        latency_ms: Wall-clock latency in milliseconds.
        model: Resolved model identifier.
        raw_response: Full provider response (for debugging).
    """

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    raw_response: dict[str, Any] | None = None


class BaseModelRunner(ABC):
    """Abstract interface for an LLM API runner."""

    @abstractmethod
    async def run(
        self, messages: list[dict[str, str]], config: ModelConfig, model: str
    ) -> ModelResponse:
        """Send messages to the model and return its response.

        Args:
            messages: Chat-style message list.
            config: Sampling parameters.
            model: Model alias or identifier.

        Returns:
            A :class:`ModelResponse`.
        """


class OpenAIRunner(BaseModelRunner):
    """Runner for OpenAI chat-completion models.

    Args:
        api_key: OpenAI API key.
    """

    MODEL_MAP: dict[str, str] = {
        "gpt-4": "gpt-4-turbo-preview",
        "gpt-4-turbo": "gpt-4-turbo-preview",
        "gpt-3.5": "gpt-3.5-turbo",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    }

    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def run(
        self, messages: list[dict[str, str]], config: ModelConfig, model: str
    ) -> ModelResponse:
        """Call OpenAI chat completions endpoint.

        Args:
            messages: Chat message list.
            config: Sampling parameters.
            model: Model alias.

        Returns:
            A :class:`ModelResponse` with token counts and latency.
        """
        model_id = self.MODEL_MAP.get(model, model)
        start = time.perf_counter()

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop_sequences or None,
        )

        latency = (time.perf_counter() - start) * 1000

        return ModelResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency,
            model=model_id,
            raw_response=response.model_dump(),
        )


class AnthropicRunner(BaseModelRunner):
    """Runner for Anthropic message models.

    Args:
        api_key: Anthropic API key.
    """

    MODEL_MAP: dict[str, str] = {
        "claude-sonnet": "claude-sonnet-4-20250514",
        "claude-haiku": "claude-haiku-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-4-20250514",
    }

    def __init__(self, api_key: str) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def run(
        self, messages: list[dict[str, str]], config: ModelConfig, model: str
    ) -> ModelResponse:
        """Call Anthropic messages endpoint.

        Args:
            messages: Chat message list (system role extracted separately).
            config: Sampling parameters.
            model: Model alias.

        Returns:
            A :class:`ModelResponse` with token counts and latency.
        """
        model_id = self.MODEL_MAP.get(model, model)
        start = time.perf_counter()

        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]

        response = await self.client.messages.create(
            model=model_id,
            max_tokens=config.max_tokens,
            system=system,
            messages=user_messages,
            temperature=config.temperature,
            top_p=config.top_p,
            stop_sequences=config.stop_sequences or None,
        )

        latency = (time.perf_counter() - start) * 1000

        return ModelResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
            model=model_id,
            raw_response=response.model_dump(),
        )


class MultiModelRunner:
    """Dispatch prompts across multiple providers in parallel.

    Automatically registers model aliases based on available API keys.

    Args:
        config: Application configuration with API keys.
    """

    def __init__(self, config: Config) -> None:
        self.runners: dict[str, BaseModelRunner] = {}

        if config.openai_api_key:
            openai_runner = OpenAIRunner(config.openai_api_key)
            for alias in OpenAIRunner.MODEL_MAP:
                self.runners[alias] = openai_runner

        if config.anthropic_api_key:
            anthropic_runner = AnthropicRunner(config.anthropic_api_key)
            for alias in AnthropicRunner.MODEL_MAP:
                self.runners[alias] = anthropic_runner

    @property
    def available_models(self) -> list[str]:
        """List all registered model aliases.

        Returns:
            Sorted list of model names.
        """
        return sorted(self.runners.keys())

    async def run_single(
        self, messages: list[dict[str, str]], config: ModelConfig, model: str
    ) -> ModelResponse:
        """Run a prompt on a single model.

        Args:
            messages: Chat message list.
            config: Sampling parameters.
            model: Model alias.

        Returns:
            The model's response.

        Raises:
            ValueError: If the model alias is not registered.
        """
        if model not in self.runners:
            raise ValueError(
                f"Model '{model}' not available. Available: {self.available_models}"
            )
        return await self.runners[model].run(messages, config, model)

    async def run_parallel(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        models: list[str],
    ) -> dict[str, ModelResponse | Exception]:
        """Run the same prompt across multiple models concurrently.

        Args:
            messages: Chat message list.
            config: Sampling parameters.
            models: List of model aliases.

        Returns:
            Dict mapping model name to response or exception.
        """
        tasks = {model: self.run_single(messages, config, model) for model in models}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))

    async def run_consistency_check(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        model: str,
        n: int = 5,
    ) -> list[ModelResponse | BaseException]:
        """Run the same prompt N times to measure output consistency.

        Args:
            messages: Chat message list.
            config: Sampling parameters.
            model: Model alias.
            n: Number of repetitions.

        Returns:
            List of responses (or exceptions on failure).
        """
        tasks = [self.run_single(messages, config, model) for _ in range(n)]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def safe_run(
    runner: BaseModelRunner,
    messages: list[dict[str, str]],
    config: ModelConfig,
    model: str,
    timeout: float = 60.0,
) -> ModelResponse | Exception:
    """Run a model call with timeout protection.

    Args:
        runner: The model runner to use.
        messages: Chat message list.
        config: Sampling parameters.
        model: Model alias.
        timeout: Timeout in seconds.

    Returns:
        The model response, or an exception on failure.
    """
    try:
        return await asyncio.wait_for(
            runner.run(messages, config, model),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return TimeoutError(f"Model {model} timed out after {timeout}s")
    except Exception as exc:
        return exc
