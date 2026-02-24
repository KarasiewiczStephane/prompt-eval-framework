"""Tests for A/B testing with statistical comparison."""

import pytest

from src.evaluation.ab_tester import ABTester
from src.evaluation.metrics import CostMetrics


@pytest.fixture()
def tester() -> ABTester:
    return ABTester()


class TestMcNemarTest:
    """McNemar's test with known contingency tables."""

    def test_no_discordant_pairs(self, tester: ABTester) -> None:
        pairs = [(True, True), (False, False), (True, True)]
        p = tester._mcnemar_test(pairs)
        assert p == 1.0

    def test_significant_difference(self, tester: ABTester) -> None:
        # 20 cases where A wrong but B right, 2 opposite
        pairs = [(False, True)] * 20 + [(True, False)] * 2
        p = tester._mcnemar_test(pairs)
        assert p < 0.05

    def test_balanced_discordant(self, tester: ABTester) -> None:
        pairs = [(True, False)] * 10 + [(False, True)] * 10
        p = tester._mcnemar_test(pairs)
        assert p > 0.5  # No significant difference


class TestBootstrapCI:
    """Bootstrap confidence intervals."""

    def test_identical_results_narrow_ci(self, tester: ABTester) -> None:
        a = [True] * 50
        b = [True] * 50
        lo, hi = tester._bootstrap_ci(a, b, n_bootstrap=500)
        assert lo == pytest.approx(0.0, abs=0.05)
        assert hi == pytest.approx(0.0, abs=0.05)

    def test_b_better_positive_ci(self, tester: ABTester) -> None:
        a = [True] * 30 + [False] * 70
        b = [True] * 70 + [False] * 30
        lo, hi = tester._bootstrap_ci(a, b, n_bootstrap=1000)
        assert lo > 0  # B is consistently better
        assert hi > lo

    def test_empty_results(self, tester: ABTester) -> None:
        lo, hi = tester._bootstrap_ci([], [], n_bootstrap=100)
        assert lo == 0.0
        assert hi == 0.0


class TestCompare:
    """Full A/B comparison."""

    def test_b_wins(self, tester: ABTester) -> None:
        a = [True] * 30 + [False] * 70
        b = [True] * 80 + [False] * 20
        result = tester.compare(a, b, n_bootstrap=500)
        assert result.winner == "B"
        assert result.effect_size > 0
        assert result.p_value < 0.05
        assert result.confidence > 0

    def test_a_wins(self, tester: ABTester) -> None:
        a = [True] * 80 + [False] * 20
        b = [True] * 30 + [False] * 70
        result = tester.compare(a, b, n_bootstrap=500)
        assert result.winner == "A"
        assert result.effect_size < 0

    def test_tie(self, tester: ABTester) -> None:
        a = [True] * 50 + [False] * 50
        b = [True] * 50 + [False] * 50
        result = tester.compare(a, b, n_bootstrap=500)
        assert result.winner == "tie"
        assert result.confidence == 0.0

    def test_mismatched_lengths_raises(self, tester: ABTester) -> None:
        with pytest.raises(ValueError, match="same length"):
            tester.compare([True], [True, False])


class TestRecommendation:
    """Recommendation generation."""

    def test_tie_lower_cost_a(self, tester: ABTester) -> None:
        cost_a = CostMetrics(
            total_cost_usd=1.0,
            cost_per_request=0.1,
            cost_per_correct=0.2,
            input_cost=0.5,
            output_cost=0.5,
        )
        cost_b = CostMetrics(
            total_cost_usd=5.0,
            cost_per_request=0.5,
            cost_per_correct=1.0,
            input_cost=2.5,
            output_cost=2.5,
        )
        rec = tester._generate_recommendation("tie", 0.0, 0.5, cost_a, cost_b)
        assert "Variant A" in rec

    def test_tie_lower_cost_b(self, tester: ABTester) -> None:
        cost_a = CostMetrics(
            total_cost_usd=5.0,
            cost_per_request=0.5,
            cost_per_correct=1.0,
            input_cost=2.5,
            output_cost=2.5,
        )
        cost_b = CostMetrics(
            total_cost_usd=1.0,
            cost_per_request=0.1,
            cost_per_correct=0.2,
            input_cost=0.5,
            output_cost=0.5,
        )
        rec = tester._generate_recommendation("tie", 0.0, 0.5, cost_a, cost_b)
        assert "Variant B" in rec

    def test_tie_no_cost(self, tester: ABTester) -> None:
        rec = tester._generate_recommendation("tie", 0.0, 0.5, None, None)
        assert "No significant difference" in rec

    def test_winner_recommendation(self, tester: ABTester) -> None:
        rec = tester._generate_recommendation("B", 0.15, 0.01, None, None)
        assert "Variant B" in rec
        assert "15.0%" in rec
