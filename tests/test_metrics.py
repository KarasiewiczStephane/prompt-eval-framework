"""Tests for the metrics calculation engine."""

import pytest

from src.evaluation.metrics import (
    MetricsCalculator,
    ModelMetrics,
    TestResult,
    compare_models,
)
from src.evaluation.model_runner import ModelResponse


def _resp(
    content: str = "ok",
    input_tokens: int = 10,
    output_tokens: int = 5,
    latency_ms: float = 100.0,
    model: str = "gpt-4-turbo-preview",
) -> ModelResponse:
    return ModelResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model=model,
    )


@pytest.fixture()
def calc() -> MetricsCalculator:
    return MetricsCalculator()


class TestLatency:
    """Latency distribution calculations."""

    def test_basic_percentiles(self, calc: MetricsCalculator) -> None:
        lat = calc.calculate_latency([100, 200, 300, 400, 500])
        assert lat.p50 == 300
        assert lat.mean == pytest.approx(300)
        assert lat.std > 0

    def test_single_value(self, calc: MetricsCalculator) -> None:
        lat = calc.calculate_latency([42.0])
        assert lat.p50 == 42.0
        assert lat.std == 0

    def test_empty(self, calc: MetricsCalculator) -> None:
        lat = calc.calculate_latency([])
        assert lat.mean == 0


class TestTokens:
    """Token aggregation."""

    def test_sums_and_averages(self, calc: MetricsCalculator) -> None:
        responses = [
            _resp(input_tokens=10, output_tokens=5),
            _resp(input_tokens=20, output_tokens=10),
        ]
        tok = calc.calculate_tokens(responses)
        assert tok.total_input == 30
        assert tok.total_output == 15
        assert tok.total == 45
        assert tok.avg_input == 15.0

    def test_empty(self, calc: MetricsCalculator) -> None:
        tok = calc.calculate_tokens([])
        assert tok.total == 0


class TestCost:
    """Cost computation from token usage."""

    def test_known_pricing(self, calc: MetricsCalculator) -> None:
        from src.evaluation.metrics import TokenMetrics

        tok = TokenMetrics(
            total_input=1_000_000,
            total_output=1_000_000,
            total=2_000_000,
            avg_input=500_000,
            avg_output=500_000,
        )
        cost = calc.calculate_cost("gpt-4-turbo-preview", tok, passed=2)
        assert cost.input_cost == pytest.approx(10.0)
        assert cost.output_cost == pytest.approx(30.0)
        assert cost.total_cost_usd == pytest.approx(40.0)
        assert cost.cost_per_correct == pytest.approx(20.0)

    def test_zero_passed(self, calc: MetricsCalculator) -> None:
        from src.evaluation.metrics import TokenMetrics

        tok = TokenMetrics(
            total_input=100,
            total_output=50,
            total=150,
            avg_input=100,
            avg_output=50,
        )
        cost = calc.calculate_cost("gpt-4-turbo-preview", tok, passed=0)
        assert cost.cost_per_correct is None

    def test_unknown_model(self, calc: MetricsCalculator) -> None:
        from src.evaluation.metrics import TokenMetrics

        tok = TokenMetrics(
            total_input=100,
            total_output=50,
            total=150,
            avg_input=100,
            avg_output=50,
        )
        cost = calc.calculate_cost("unknown-model", tok, passed=1)
        assert cost.total_cost_usd == 0


class TestAccuracy:
    """Accuracy calculations."""

    def test_basic_accuracy(self, calc: MetricsCalculator) -> None:
        results = [
            TestResult(passed=True),
            TestResult(passed=True),
            TestResult(passed=False),
        ]
        acc = calc.calculate_accuracy(results)
        assert acc.total == 3
        assert acc.passed == 2
        assert acc.accuracy == pytest.approx(2 / 3)

    def test_empty_results(self, calc: MetricsCalculator) -> None:
        acc = calc.calculate_accuracy([])
        assert acc.accuracy == 0

    def test_by_tag(self, calc: MetricsCalculator) -> None:
        class FakeTC:
            def __init__(self, tags):
                self.tags = tags

        results = [
            TestResult(passed=True, test_case=FakeTC(["basic"])),
            TestResult(passed=False, test_case=FakeTC(["basic", "advanced"])),
            TestResult(passed=True, test_case=FakeTC(["advanced"])),
        ]
        acc = calc.calculate_accuracy(results, tags=["basic", "advanced"])
        assert acc.by_tag["basic"] == pytest.approx(0.5)
        assert acc.by_tag["advanced"] == pytest.approx(0.5)


class TestConsistency:
    """Output consistency metrics."""

    def test_identical_outputs(self, calc: MetricsCalculator) -> None:
        c = calc.calculate_consistency(["hello", "hello", "hello"])
        assert c.unique_outputs == 1
        assert c.variance_score == 0

    def test_all_different(self, calc: MetricsCalculator) -> None:
        c = calc.calculate_consistency(["a", "b", "c"])
        assert c.unique_outputs == 3
        assert c.variance_score == 1.0

    def test_most_common(self, calc: MetricsCalculator) -> None:
        c = calc.calculate_consistency(["a", "b", "a", "a"])
        assert c.most_common_output == "a"
        assert c.most_common_frequency == pytest.approx(0.75)

    def test_empty(self, calc: MetricsCalculator) -> None:
        c = calc.calculate_consistency([])
        assert c.unique_outputs == 0


class TestCalculateAll:
    """End-to-end ModelMetrics calculation."""

    def test_produces_all_metric_types(self, calc: MetricsCalculator) -> None:
        results = [
            TestResult(passed=True, response=_resp(latency_ms=100)),
            TestResult(passed=False, response=_resp(latency_ms=200)),
        ]
        mm = calc.calculate_all("gpt-4-turbo-preview", results)
        assert isinstance(mm, ModelMetrics)
        assert mm.accuracy.total == 2
        assert mm.latency.mean > 0
        assert mm.tokens.total > 0

    def test_with_consistency(self, calc: MetricsCalculator) -> None:
        results = [TestResult(passed=True, response=_resp())]
        mm = calc.calculate_all(
            "gpt-4-turbo-preview",
            results,
            consistency_outputs=["ok", "ok", "different"],
        )
        assert mm.consistency is not None
        assert mm.consistency.unique_outputs == 2


class TestCompareModels:
    """Cross-model comparison rankings."""

    def test_rankings(self, calc: MetricsCalculator) -> None:
        results_a = [
            TestResult(
                passed=True, response=_resp(latency_ms=50, model="gpt-3.5-turbo")
            )
        ]
        results_b = [
            TestResult(
                passed=False,
                response=_resp(latency_ms=200, model="gpt-4-turbo-preview"),
            )
        ]

        metrics_a = calc.calculate_all("gpt-3.5-turbo", results_a)
        metrics_b = calc.calculate_all("gpt-4-turbo-preview", results_b)

        comparison = compare_models(
            {"gpt-3.5-turbo": metrics_a, "gpt-4-turbo-preview": metrics_b}
        )

        assert comparison["accuracy_ranking"][0] == "gpt-3.5-turbo"
        assert comparison["latency_ranking"][0] == "gpt-3.5-turbo"
        assert "efficiency_score" in comparison
