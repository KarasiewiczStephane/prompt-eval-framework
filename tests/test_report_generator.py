"""Tests for the HTML report generator."""

from pathlib import Path

import pytest

from src.evaluation.metrics import (
    AccuracyMetrics,
    CostMetrics,
    LatencyMetrics,
    ModelMetrics,
    TestResult,
    TokenMetrics,
)
from src.evaluation.model_runner import ModelResponse
from src.reporting.report_generator import ReportGenerator, RunInfo


def _make_metrics(model: str = "gpt-4") -> ModelMetrics:
    return ModelMetrics(
        model=model,
        accuracy=AccuracyMetrics(
            total=10,
            passed=8,
            failed=2,
            accuracy=0.8,
            by_tag={"basic": 0.9, "advanced": 0.7},
        ),
        latency=LatencyMetrics(p50=150, p95=300, p99=400, mean=180, std=50),
        tokens=TokenMetrics(
            total_input=500,
            total_output=200,
            total=700,
            avg_input=50,
            avg_output=20,
        ),
        cost=CostMetrics(
            total_cost_usd=0.05,
            cost_per_request=0.005,
            cost_per_correct=0.00625,
            input_cost=0.02,
            output_cost=0.03,
        ),
    )


def _make_run_info() -> RunInfo:
    return RunInfo(
        suite_name="test_suite", model="gpt-4", completed_at="2025-01-15T10:00:00"
    )


def _make_results() -> list[TestResult]:
    class FakeTC:
        def __init__(self, id, name, expected):
            self.id = id
            self.name = name
            self.expected_output = expected

    return [
        TestResult(
            passed=True,
            response=ModelResponse(
                content="ok",
                input_tokens=50,
                output_tokens=20,
                latency_ms=100,
                model="gpt-4",
            ),
            latency_ms=100,
            test_case=FakeTC("tc_1", "Basic test", "ok"),
        ),
        TestResult(
            passed=False,
            response=ModelResponse(
                content="wrong answer",
                input_tokens=50,
                output_tokens=20,
                latency_ms=200,
                model="gpt-4",
            ),
            latency_ms=200,
            test_case=FakeTC("tc_2", "Hard test", "correct answer"),
        ),
    ]


@pytest.fixture()
def generator() -> ReportGenerator:
    return ReportGenerator()


class TestReportGeneration:
    """Report template rendering."""

    def test_renders_html(self, generator: ReportGenerator) -> None:
        html = generator.generate(
            _make_run_info(),
            _make_metrics(),
            _make_results(),
        )
        assert "<!DOCTYPE html>" in html
        assert "Prompt Evaluation Report" in html

    def test_includes_metrics(self, generator: ReportGenerator) -> None:
        html = generator.generate(
            _make_run_info(),
            _make_metrics(),
            _make_results(),
        )
        assert "80.0%" in html
        assert "150ms" in html

    def test_includes_failed_tests(self, generator: ReportGenerator) -> None:
        html = generator.generate(
            _make_run_info(),
            _make_metrics(),
            _make_results(),
        )
        assert "Failed Tests" in html
        assert "Hard test" in html

    def test_no_failures(self, generator: ReportGenerator) -> None:
        results = [r for r in _make_results() if r.passed]
        html = generator.generate(
            _make_run_info(),
            _make_metrics(),
            results,
        )
        assert "Failed Tests" not in html

    def test_model_comparison(self, generator: ReportGenerator) -> None:
        model_metrics = {
            "gpt-4": _make_metrics("gpt-4"),
            "gpt-3.5": _make_metrics("gpt-3.5"),
        }
        html = generator.generate(
            _make_run_info(),
            _make_metrics(),
            _make_results(),
            model_metrics=model_metrics,
        )
        assert "Model Comparison" in html

    def test_single_model_no_comparison(self, generator: ReportGenerator) -> None:
        html = generator.generate(
            _make_run_info(),
            _make_metrics(),
            _make_results(),
            model_metrics={"gpt-4": _make_metrics()},
        )
        assert "Model Comparison" not in html


class TestGenerateToFile:
    """Writing report to a file."""

    def test_writes_file(self, generator: ReportGenerator, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        generator.generate_to_file(
            out,
            _make_run_info(),
            _make_metrics(),
            _make_results(),
        )
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content

    def test_creates_parent_dirs(
        self, generator: ReportGenerator, tmp_path: Path
    ) -> None:
        out = tmp_path / "nested" / "dir" / "report.html"
        generator.generate_to_file(
            out,
            _make_run_info(),
            _make_metrics(),
            _make_results(),
        )
        assert out.exists()


class TestExtractFailures:
    """_extract_failures should collect failed test details."""

    def test_extracts_correct_count(self) -> None:
        results = _make_results()
        failures = ReportGenerator._extract_failures(results)
        assert len(failures) == 1
        assert failures[0].id == "tc_2"

    def test_no_failures(self) -> None:
        results = [r for r in _make_results() if r.passed]
        failures = ReportGenerator._extract_failures(results)
        assert failures == []
