"""Tests for documentation and example files."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


class TestReadme:
    """README should document key features."""

    def test_exists(self) -> None:
        assert (ROOT / "README.md").exists()

    def test_has_title(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "Prompt Engineering Evaluation Framework" in content

    def test_has_quick_start(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "Quick Start" in content

    def test_has_usage_section(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "## Quick Start" in content or "## Usage" in content

    def test_has_docker_section(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "Docker" in content

    def test_has_project_structure(self) -> None:
        content = (ROOT / "README.md").read_text()
        assert "Project Structure" in content


class TestExamplePrompts:
    """Example prompts should be valid YAML with required fields."""

    PROMPTS_DIR = ROOT / "example_prompts"

    def test_prompts_dir_exists(self) -> None:
        assert self.PROMPTS_DIR.exists()

    def test_has_multiple_prompts(self) -> None:
        prompts = list(self.PROMPTS_DIR.glob("*.yaml"))
        assert len(prompts) >= 5

    def test_greeting_prompt(self) -> None:
        data = yaml.safe_load((self.PROMPTS_DIR / "greeting.yaml").read_text())
        assert data["name"] == "greeting"
        assert "system_prompt" in data
        assert "user_prompt" in data
        assert "variables" in data

    def test_summarization_prompt(self) -> None:
        data = yaml.safe_load((self.PROMPTS_DIR / "summarization.yaml").read_text())
        assert data["name"] == "summarization"
        assert "text" in data["variables"]

    def test_classification_prompt(self) -> None:
        data = yaml.safe_load((self.PROMPTS_DIR / "classification.yaml").read_text())
        assert data["name"] == "classification"

    def test_extraction_prompt(self) -> None:
        data = yaml.safe_load((self.PROMPTS_DIR / "extraction.yaml").read_text())
        assert data["name"] == "extraction"

    def test_code_review_prompt(self) -> None:
        data = yaml.safe_load((self.PROMPTS_DIR / "code_review.yaml").read_text())
        assert data["name"] == "code_review"
        assert "language" in data["variables"]

    def test_all_prompts_have_required_fields(self) -> None:
        required = {"name", "system_prompt", "user_prompt", "variables"}
        for path in self.PROMPTS_DIR.glob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            missing = required - set(data.keys())
            assert not missing, f"{path.name} missing: {missing}"


class TestExampleSuites:
    """Example test suites should be valid YAML with test cases."""

    SUITES_DIR = ROOT / "example_suites"

    def test_suites_dir_exists(self) -> None:
        assert self.SUITES_DIR.exists()

    def test_has_multiple_suites(self) -> None:
        suites = list(self.SUITES_DIR.glob("*.yaml"))
        assert len(suites) >= 4

    def test_greeting_suite(self) -> None:
        data = yaml.safe_load((self.SUITES_DIR / "greeting_suite.yaml").read_text())
        assert data["name"] == "greeting_test_suite"
        assert len(data["test_cases"]) >= 2

    def test_summarization_suite(self) -> None:
        data = yaml.safe_load(
            (self.SUITES_DIR / "summarization_suite.yaml").read_text()
        )
        assert len(data["test_cases"]) >= 5

    def test_classification_suite(self) -> None:
        data = yaml.safe_load(
            (self.SUITES_DIR / "classification_suite.yaml").read_text()
        )
        assert len(data["test_cases"]) >= 5

    def test_extraction_suite(self) -> None:
        data = yaml.safe_load((self.SUITES_DIR / "extraction_suite.yaml").read_text())
        assert len(data["test_cases"]) >= 5

    def test_all_suites_have_required_fields(self) -> None:
        required = {"name", "prompt", "test_cases"}
        for path in self.SUITES_DIR.glob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            missing = required - set(data.keys())
            assert not missing, f"{path.name} missing: {missing}"

    def test_all_test_cases_have_ids(self) -> None:
        for path in self.SUITES_DIR.glob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            for tc in data["test_cases"]:
                assert "id" in tc, f"{path.name}: test case missing id"
                assert "name" in tc, f"{path.name}: test case missing name"

    def test_total_test_cases(self) -> None:
        total = 0
        for path in self.SUITES_DIR.glob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            total += len(data["test_cases"])
        assert total >= 17


class TestConfig:
    """Config file should have required sections."""

    def test_config_exists(self) -> None:
        assert (ROOT / "configs" / "config.yaml").exists()

    def test_has_database_section(self) -> None:
        data = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
        assert "database" in data

    def test_has_defaults_section(self) -> None:
        data = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
        assert "defaults" in data

    def test_has_pricing_section(self) -> None:
        data = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
        assert "pricing" in data
