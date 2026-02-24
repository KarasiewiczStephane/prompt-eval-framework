"""Metrics calculation engine for evaluation results.

Computes accuracy, latency percentiles, token usage, cost, and output
consistency.  Also provides cross-model comparison rankings.
"""

import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.evaluation.model_runner import ModelResponse

logger = logging.getLogger(__name__)

PRICING: dict[str, dict[str, float]] = {
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-20250514": {"input": 0.25, "output": 1.25},
}


@dataclass
class LatencyMetrics:
    """Latency distribution statistics.

    Attributes:
        p50: Median latency (ms).
        p95: 95th-percentile latency (ms).
        p99: 99th-percentile latency (ms).
        mean: Mean latency (ms).
        std: Standard deviation (ms).
    """

    p50: float
    p95: float
    p99: float
    mean: float
    std: float


@dataclass
class TokenMetrics:
    """Aggregate token usage.

    Attributes:
        total_input: Sum of input tokens.
        total_output: Sum of output tokens.
        total: Combined token count.
        avg_input: Mean input tokens per request.
        avg_output: Mean output tokens per request.
    """

    total_input: int
    total_output: int
    total: int
    avg_input: float
    avg_output: float


@dataclass
class CostMetrics:
    """Cost breakdown.

    Attributes:
        total_cost_usd: Total spend.
        cost_per_request: Average cost per API call.
        cost_per_correct: Cost divided by number of passing tests.
        input_cost: Spend on input tokens.
        output_cost: Spend on output tokens.
    """

    total_cost_usd: float
    cost_per_request: float
    cost_per_correct: float | None
    input_cost: float
    output_cost: float


@dataclass
class ConsistencyMetrics:
    """Output consistency statistics across repeated runs.

    Attributes:
        unique_outputs: Number of distinct outputs.
        variance_score: 0 = identical, 1 = all different.
        most_common_output: The most frequent output text.
        most_common_frequency: Frequency of the most common output.
    """

    unique_outputs: int
    variance_score: float
    most_common_output: str
    most_common_frequency: float


@dataclass
class AccuracyMetrics:
    """Pass/fail accuracy summary.

    Attributes:
        total: Total number of test cases.
        passed: Number that passed.
        failed: Number that failed.
        accuracy: Pass rate (0.0 – 1.0).
        by_tag: Accuracy broken down by tag.
    """

    total: int
    passed: int
    failed: int
    accuracy: float
    by_tag: dict[str, float] = field(default_factory=dict)


@dataclass
class TestResult:
    """A single test-case evaluation outcome.

    Attributes:
        passed: Whether the assertion passed.
        response: The model response (if successful).
        latency_ms: Latency in ms.
        test_case: Reference back to the test-case definition.
    """

    passed: bool
    response: ModelResponse | None = None
    latency_ms: float = 0.0
    test_case: Any = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for one model.

    Attributes:
        model: Model identifier.
        accuracy: Accuracy metrics.
        latency: Latency distribution.
        tokens: Token usage.
        cost: Cost breakdown.
        consistency: Optional consistency metrics.
    """

    model: str
    accuracy: AccuracyMetrics
    latency: LatencyMetrics
    tokens: TokenMetrics
    cost: CostMetrics
    consistency: ConsistencyMetrics | None = None


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list.

    Args:
        sorted_values: Sorted numeric sequence.
        pct: Percentile as a fraction (e.g. 0.95).

    Returns:
        The value at the given percentile.
    """
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


class MetricsCalculator:
    """Calculate evaluation metrics from test results.

    Args:
        pricing: Per-model token pricing dict (per 1 M tokens).
    """

    def __init__(self, pricing: dict[str, dict[str, float]] | None = None) -> None:
        self.pricing = pricing or PRICING

    def calculate_latency(self, latencies: Sequence[float]) -> LatencyMetrics:
        """Compute latency distribution.

        Args:
            latencies: Per-request latency values (ms).

        Returns:
            Latency percentiles and stats.
        """
        if not latencies:
            return LatencyMetrics(p50=0, p95=0, p99=0, mean=0, std=0)

        sorted_lat = sorted(latencies)
        return LatencyMetrics(
            p50=_percentile(sorted_lat, 0.50),
            p95=_percentile(sorted_lat, 0.95),
            p99=_percentile(sorted_lat, 0.99),
            mean=statistics.mean(latencies),
            std=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        )

    def calculate_tokens(self, responses: Sequence[ModelResponse]) -> TokenMetrics:
        """Aggregate token counts.

        Args:
            responses: List of model responses.

        Returns:
            Token usage summary.
        """
        if not responses:
            return TokenMetrics(
                total_input=0, total_output=0, total=0, avg_input=0, avg_output=0
            )

        input_tokens = [r.input_tokens for r in responses]
        output_tokens = [r.output_tokens for r in responses]
        return TokenMetrics(
            total_input=sum(input_tokens),
            total_output=sum(output_tokens),
            total=sum(input_tokens) + sum(output_tokens),
            avg_input=statistics.mean(input_tokens),
            avg_output=statistics.mean(output_tokens),
        )

    def calculate_cost(
        self, model: str, tokens: TokenMetrics, passed: int
    ) -> CostMetrics:
        """Compute cost from token usage and pricing.

        Args:
            model: Model identifier for pricing lookup.
            tokens: Aggregated token metrics.
            passed: Number of passing test cases.

        Returns:
            Cost breakdown.
        """
        pricing = self.pricing.get(model, {"input": 0, "output": 0})
        input_cost = (tokens.total_input / 1_000_000) * pricing["input"]
        output_cost = (tokens.total_output / 1_000_000) * pricing["output"]
        total = input_cost + output_cost
        n_requests = max(
            tokens.total_input / tokens.avg_input if tokens.avg_input > 0 else 1, 1
        )

        return CostMetrics(
            total_cost_usd=total,
            cost_per_request=total / n_requests,
            cost_per_correct=total / passed if passed > 0 else None,
            input_cost=input_cost,
            output_cost=output_cost,
        )

    def calculate_accuracy(
        self,
        results: list[TestResult],
        tags: list[str] | None = None,
    ) -> AccuracyMetrics:
        """Compute pass/fail accuracy, optionally broken down by tag.

        Args:
            results: List of test outcomes.
            tags: Optional tags to compute per-tag accuracy for.

        Returns:
            Accuracy summary.
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        by_tag: dict[str, float] = {}
        if tags:
            for tag in tags:
                tag_results = [
                    r
                    for r in results
                    if r.test_case
                    and hasattr(r.test_case, "tags")
                    and tag in r.test_case.tags
                ]
                if tag_results:
                    by_tag[tag] = sum(1 for r in tag_results if r.passed) / len(
                        tag_results
                    )

        return AccuracyMetrics(
            total=total,
            passed=passed,
            failed=total - passed,
            accuracy=passed / total if total > 0 else 0,
            by_tag=by_tag,
        )

    def calculate_consistency(self, outputs: list[str]) -> ConsistencyMetrics:
        """Measure output consistency across repeated runs.

        Args:
            outputs: List of model output strings.

        Returns:
            Consistency statistics.
        """
        if not outputs:
            return ConsistencyMetrics(
                unique_outputs=0,
                variance_score=1.0,
                most_common_output="",
                most_common_frequency=0.0,
            )

        normalized = [o.strip().lower() for o in outputs]
        unique = set(normalized)
        counts = Counter(normalized)
        most_common_norm, freq = counts.most_common(1)[0]

        variance = (len(unique) - 1) / (len(outputs) - 1) if len(outputs) > 1 else 0
        original = next(o for o in outputs if o.strip().lower() == most_common_norm)

        return ConsistencyMetrics(
            unique_outputs=len(unique),
            variance_score=variance,
            most_common_output=original,
            most_common_frequency=freq / len(outputs),
        )

    def calculate_all(
        self,
        model: str,
        results: list[TestResult],
        consistency_outputs: list[str] | None = None,
    ) -> ModelMetrics:
        """Compute all metrics for a single model's results.

        Args:
            model: Model identifier.
            results: Test outcomes.
            consistency_outputs: Optional repeated-run outputs.

        Returns:
            Complete :class:`ModelMetrics`.
        """
        responses = [r.response for r in results if r.response]

        accuracy = self.calculate_accuracy(results)
        latency = self.calculate_latency([r.latency_ms for r in responses])
        tokens = self.calculate_tokens(responses)
        cost = self.calculate_cost(model, tokens, accuracy.passed)
        consistency = (
            self.calculate_consistency(consistency_outputs)
            if consistency_outputs
            else None
        )

        return ModelMetrics(
            model=model,
            accuracy=accuracy,
            latency=latency,
            tokens=tokens,
            cost=cost,
            consistency=consistency,
        )


def compare_models(metrics: dict[str, ModelMetrics]) -> dict[str, Any]:
    """Generate cross-model comparison rankings.

    Args:
        metrics: Model name to metrics mapping.

    Returns:
        Dict with accuracy, cost, latency rankings and efficiency scores.
    """
    return {
        "accuracy_ranking": sorted(
            metrics.keys(),
            key=lambda m: metrics[m].accuracy.accuracy,
            reverse=True,
        ),
        "cost_ranking": sorted(
            metrics.keys(), key=lambda m: metrics[m].cost.total_cost_usd
        ),
        "latency_ranking": sorted(metrics.keys(), key=lambda m: metrics[m].latency.p50),
        "efficiency_score": {
            m: (
                metrics[m].accuracy.accuracy / metrics[m].cost.total_cost_usd
                if metrics[m].cost.total_cost_usd > 0
                else float("inf")
            )
            for m in metrics
        },
    }
