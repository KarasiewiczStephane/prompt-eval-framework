"""Mock API response objects for testing without real API calls."""

import json
from pathlib import Path
from types import SimpleNamespace

MOCK_RESPONSES_DIR = Path(__file__).parent / "mock_responses"


class MockOpenAIResponse:
    """Mimics openai ChatCompletion response structure."""

    def __init__(
        self,
        content: str = "Mock response",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        self.choices = [choice]
        self.usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )

    def model_dump(self) -> dict:
        return {"mock": True}


class MockAnthropicResponse:
    """Mimics anthropic Messages response structure."""

    def __init__(
        self,
        content: str = "Mock response",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        text_block = SimpleNamespace(text=content)
        self.content = [text_block]
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def model_dump(self) -> dict:
        return {"mock": True}


def load_mock_response(name: str) -> dict:
    """Load recorded API response from fixture."""
    path = MOCK_RESPONSES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def save_mock_response(name: str, response: dict) -> None:
    """Save API response for future mocking."""
    MOCK_RESPONSES_DIR.mkdir(exist_ok=True)
    path = MOCK_RESPONSES_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(response, f, indent=2)
