"""Tests for cost estimation, budget enforcement, and recommendations."""

import pytest

from src.evaluation.cost_optimizer import BudgetEnforcer, CostOptimizer
from src.evaluation.metrics import (
    AccuracyMetrics,
    CostMetrics,
    LatencyMetrics,
    ModelMetrics,
    TokenMetrics,
)


@pytest.fixture()
def optimizer() -> CostOptimizer:
    return CostOptimizer()


class TestEstimateTokens:
    """Token estimation heuristics."""

    def test_returns_positive(self, optimizer: CostOptimizer) -> None:
        tokens = optimizer.estimate_tokens("Hello world, this is a test prompt.")
        assert tokens > 0

    def test_longer_text_more_tokens(self, optimizer: CostOptimizer) -> None:
        short = optimizer.estimate_tokens("Hi")
        long = optimizer.estimate_tokens("Hi " * 100)
        assert long > short

    def test_empty_string(self, optimizer: CostOptimizer) -> None:
        tokens = optimizer.estimate_tokens("")
        assert tokens >= 0


class TestEstimateCost:
    """Pre-run cost estimation."""

    def test_single_model(self, optimizer: CostOptimizer) -> None:
        est = optimizer.estimate_cost(
            prompts=["Hello world"] * 10,
            models=["gpt-3.5-turbo"],
            task_type="classification",
        )
        assert est.estimated_input_tokens > 0
        assert est.estimated_cost_usd > 0
        assert est.within_budget is True

    def test_multiple_models(self, optimizer: CostOptimizer) -> None:
        est = optimizer.estimate_cost(
            prompts=["Test"] * 5,
            models=["gpt-3.5-turbo", "gpt-4-turbo-preview"],
        )
        assert len(est.breakdown_by_model) == 2
        assert est.estimated_cost_usd == pytest.approx(
            sum(est.breakdown_by_model.values())
        )

    def test_within_budget(self, optimizer: CostOptimizer) -> None:
        est = optimizer.estimate_cost(
            prompts=["Test"], models=["gpt-3.5-turbo"], budget=100.0
        )
        assert est.within_budget is True

    def test_over_budget(self, optimizer: CostOptimizer) -> None:
        est = optimizer.estimate_cost(
            prompts=["Test " * 1000] * 1000,
            models=["gpt-4-turbo-preview"],
            budget=0.0001,
        )
        assert est.within_budget is False
        assert est.budget_utilization > 100


class TestRecommendCheaperModel:
    """Model optimization recommendations."""

    def _make_metrics(self, model: str, accuracy: float, cost: float) -> ModelMetrics:
        return ModelMetrics(
            model=model,
            accuracy=AccuracyMetrics(
                total=100,
                passed=int(accuracy * 100),
                failed=100 - int(accuracy * 100),
                accuracy=accuracy,
            ),
            latency=LatencyMetrics(p50=100, p95=200, p99=300, mean=150, std=50),
            tokens=TokenMetrics(
                total_input=1000,
                total_output=500,
                total=1500,
                avg_input=10,
                avg_output=5,
            ),
            cost=CostMetrics(
                total_cost_usd=cost,
                cost_per_request=cost / 100,
                cost_per_correct=cost / max(int(accuracy * 100), 1),
                input_cost=cost * 0.3,
                output_cost=cost * 0.7,
            ),
        )

    def test_recommends_cheaper(self, optimizer: CostOptimizer) -> None:
        metrics = {
            "expensive": self._make_metrics("expensive", 0.95, 10.0),
            "cheap": self._make_metrics("cheap", 0.93, 2.0),
        }
        rec = optimizer.recommend_cheaper_model(metrics)
        assert rec is not None
        assert rec.recommended_model == "cheap"
        assert rec.savings_percent > 0

    def test_best_is_cheapest(self, optimizer: CostOptimizer) -> None:
        metrics = {
            "best": self._make_metrics("best", 0.99, 1.0),
            "other": self._make_metrics("other", 0.50, 5.0),
        }
        rec = optimizer.recommend_cheaper_model(metrics)
        assert rec is not None
        assert rec.recommended_model == "best"
        assert rec.savings_percent == 0

    def test_empty_metrics(self, optimizer: CostOptimizer) -> None:
        assert optimizer.recommend_cheaper_model({}) is None


class TestSuggestPromptCompression:
    """Prompt compression suggestions."""

    def test_detects_politeness(self, optimizer: CostOptimizer) -> None:
        result = optimizer.suggest_prompt_compression("Please kindly do this")
        assert any("politeness" in s.lower() for s in result["suggestions"])

    def test_detects_whitespace(self, optimizer: CostOptimizer) -> None:
        prompt = "Line1\n\nLine2\n\nLine3\n\nLine4\n\nLine5"
        result = optimizer.suggest_prompt_compression(prompt)
        assert any("whitespace" in s.lower() for s in result["suggestions"])

    def test_detects_repeated_instructions(self, optimizer: CostOptimizer) -> None:
        prompt = "Make sure A. Make sure B. Ensure C. Ensure D."
        result = optimizer.suggest_prompt_compression(prompt)
        assert any("consolidate" in s.lower() for s in result["suggestions"])

    def test_clean_prompt_no_suggestions(self, optimizer: CostOptimizer) -> None:
        result = optimizer.suggest_prompt_compression("Summarize the text.")
        assert result["suggestions"] == []


class TestBudgetEnforcer:
    """Budget tracking and enforcement."""

    def test_can_spend_within_budget(self) -> None:
        be = BudgetEnforcer(10.0)
        assert be.can_spend(5.0) is True

    def test_cannot_exceed_budget(self) -> None:
        be = BudgetEnforcer(10.0)
        be.record_spend(8.0)
        assert be.can_spend(5.0) is False

    def test_remaining(self) -> None:
        be = BudgetEnforcer(10.0)
        be.record_spend(3.0)
        assert be.remaining == pytest.approx(7.0)

    def test_exact_budget(self) -> None:
        be = BudgetEnforcer(10.0)
        assert be.can_spend(10.0) is True
        be.record_spend(10.0)
        assert be.remaining == pytest.approx(0.0)
