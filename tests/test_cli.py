"""Tests for the Click CLI commands."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def suite_file(tmp_path: Path) -> Path:
    """Create a minimal test suite YAML."""
    data = {
        "name": "cli_test_suite",
        "prompt": "greeting",
        "test_cases": [
            {
                "id": "tc1",
                "name": "Basic",
                "input": {"customer_name": "Alice", "topic": "returns"},
                "expected": "Hello",
                "assertion": "contains",
                "tags": ["basic"],
            },
        ],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(data))
    return path


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Create a minimal config YAML."""
    cfg = {
        "database": {"path": str(tmp_path / "test.duckdb")},
        "defaults": {"temperature": 0.5, "max_tokens": 100},
        "paths": {
            "prompts_dir": "example_prompts",
            "suites_dir": str(tmp_path),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


class TestCLIHelp:
    """CLI --help output."""

    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Prompt Engineering Evaluation Framework" in result.output

    def test_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--suite" in result.output

    def test_compare_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--suite" in result.output

    def test_estimate_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "--suite" in result.output

    def test_history_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["history", "--help"])
        assert result.exit_code == 0
        assert "--prompt" in result.output

    def test_report_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output


class TestEstimate:
    """estimate command should show cost table."""

    def test_estimate_runs(
        self, runner: CliRunner, suite_file: Path, config_file: Path
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "-c",
                str(config_file),
                "estimate",
                "-s",
                str(suite_file),
                "-m",
                "gpt-3.5-turbo",
            ],
        )
        assert result.exit_code == 0
        assert "Cost Estimate" in result.output
        assert "Test Cases" in result.output


class TestHistory:
    """history command should show version table or empty message."""

    def test_empty_history(self, runner: CliRunner, config_file: Path) -> None:
        result = runner.invoke(
            cli, ["-c", str(config_file), "history", "-p", "nonexistent"]
        )
        assert result.exit_code == 0
        assert "No history found" in result.output


class TestRunMissingFile:
    """run command should fail gracefully with missing suite file."""

    def test_missing_suite_errors(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "-s", "/nonexistent/suite.yaml"])
        assert result.exit_code != 0
