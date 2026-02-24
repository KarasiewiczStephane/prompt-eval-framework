"""Tests for CI pipeline configuration."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"


class TestCIWorkflow:
    """CI workflow should be valid and complete."""

    def test_workflow_exists(self) -> None:
        assert CI_PATH.exists()

    def test_valid_yaml(self) -> None:
        content = CI_PATH.read_text()
        data = yaml.safe_load(content)
        assert isinstance(data, dict)

    def test_has_name(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        assert "name" in data

    def test_triggers_on_push(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        # PyYAML parses bare `on:` as boolean True
        triggers = data.get("on") or data.get(True, {})
        assert "push" in triggers

    def test_triggers_on_pr(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        triggers = data.get("on") or data.get(True, {})
        assert "pull_request" in triggers

    def test_has_lint_job(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        assert "lint" in data["jobs"]

    def test_has_test_job(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        assert "test" in data["jobs"]

    def test_has_build_job(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        assert "build" in data["jobs"]

    def test_test_needs_lint(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        assert "lint" in data["jobs"]["test"]["needs"]

    def test_build_needs_test(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        assert "test" in data["jobs"]["build"]["needs"]

    def test_uses_python_311(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        lint_steps = data["jobs"]["lint"]["steps"]
        python_step = next(
            (s for s in lint_steps if s.get("name") == "Set up Python"), None
        )
        assert python_step is not None
        assert python_step["with"]["python-version"] == "3.11"

    def test_mock_api_env_set(self) -> None:
        data = yaml.safe_load(CI_PATH.read_text())
        test_steps = data["jobs"]["test"]["steps"]
        test_step = next(
            (s for s in test_steps if s.get("name") == "Run tests with coverage"),
            None,
        )
        assert test_step is not None
        assert test_step["env"]["USE_MOCK_API"] == "true"


class TestMockResponses:
    """Mock response fixtures should work correctly."""

    def test_openai_response_structure(self) -> None:
        from tests.fixtures.mock_responses import MockOpenAIResponse

        resp = MockOpenAIResponse("test content", 10, 5)
        assert resp.choices[0].message.content == "test content"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5

    def test_anthropic_response_structure(self) -> None:
        from tests.fixtures.mock_responses import MockAnthropicResponse

        resp = MockAnthropicResponse("test content", 10, 5)
        assert resp.content[0].text == "test content"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_load_mock_response(self) -> None:
        from tests.fixtures.mock_responses import load_mock_response

        data = load_mock_response("greeting")
        assert "content" in data
        assert "input_tokens" in data

    def test_save_and_load(self, tmp_path: Path) -> None:
        import json

        response_dir = tmp_path / "mock_responses"
        response_dir.mkdir()
        path = response_dir / "test.json"
        data = {"content": "hello", "tokens": 10}
        path.write_text(json.dumps(data))
        loaded = json.loads(path.read_text())
        assert loaded == data
