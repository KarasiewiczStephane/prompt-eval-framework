"""HTML report generation using Jinja2 and Plotly.

Renders evaluation metrics, latency distributions, cost breakdowns,
and failed-test details into a standalone HTML report.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.evaluation.metrics import ModelMetrics, TestResult

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class RunInfo:
    """Metadata about an evaluation run.

    Attributes:
        suite_name: Test suite name.
        model: Primary model identifier.
        completed_at: ISO timestamp string.
    """

    suite_name: str
    model: str
    completed_at: str


@dataclass
class FailedTest:
    """Details of a single failed test case.

    Attributes:
        id: Test case ID.
        name: Test case name.
        expected: Expected output text.
        actual: Actual model output text.
    """

    id: str
    name: str
    expected: str
    actual: str


class ReportGenerator:
    """Generate standalone HTML evaluation reports.

    Args:
        template_dir: Directory containing Jinja2 HTML templates.
    """

    def __init__(self, template_dir: Path | None = None) -> None:
        tdir = template_dir or _TEMPLATE_DIR
        self.env = Environment(
            loader=FileSystemLoader(str(tdir)),
            autoescape=True,
        )

    def generate(
        self,
        run_info: RunInfo,
        metrics: ModelMetrics,
        results: list[TestResult],
        model_metrics: dict[str, ModelMetrics] | None = None,
    ) -> str:
        """Render the evaluation report to an HTML string.

        Args:
            run_info: Run metadata.
            metrics: Primary model metrics.
            results: Individual test results.
            model_metrics: Optional cross-model metrics for comparison.

        Returns:
            Complete HTML string.
        """
        failed_tests = self._extract_failures(results)
        latency_data = [r.response.latency_ms for r in results if r.response]
        cost_data = {
            "labels": ["Input", "Output"],
            "values": [metrics.cost.input_cost, metrics.cost.output_cost],
        }
        tag_data = {
            "tags": list(metrics.accuracy.by_tag.keys()),
            "accuracies": list(metrics.accuracy.by_tag.values()),
        }

        template = self.env.get_template("report.html")
        html = template.render(
            run=run_info,
            metrics=metrics,
            failed_tests=failed_tests,
            latency_data=json.dumps(latency_data),
            cost_data=json.dumps(cost_data),
            tag_data=json.dumps(tag_data),
            model_comparison=model_metrics is not None and len(model_metrics) > 1,
            model_metrics=model_metrics or {},
        )
        return html

    def generate_to_file(
        self,
        output_path: Path,
        run_info: RunInfo,
        metrics: ModelMetrics,
        results: list[TestResult],
        model_metrics: dict[str, ModelMetrics] | None = None,
    ) -> Path:
        """Render the report and write it to a file.

        Args:
            output_path: Destination file path.
            run_info: Run metadata.
            metrics: Primary model metrics.
            results: Individual test results.
            model_metrics: Optional cross-model metrics.

        Returns:
            The output path.
        """
        html = self.generate(run_info, metrics, results, model_metrics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        logger.info("Report written to %s", output_path)
        return output_path

    @staticmethod
    def _extract_failures(results: list[TestResult]) -> list[FailedTest]:
        """Collect details for all failed tests.

        Args:
            results: List of test results.

        Returns:
            List of :class:`FailedTest` with expected/actual output.
        """
        failures: list[FailedTest] = []
        for r in results:
            if not r.passed and r.test_case:
                failures.append(
                    FailedTest(
                        id=getattr(r.test_case, "id", "unknown"),
                        name=getattr(r.test_case, "name", "unknown"),
                        expected=str(getattr(r.test_case, "expected_output", "")),
                        actual=r.response.content if r.response else "(no response)",
                    )
                )
        return failures
