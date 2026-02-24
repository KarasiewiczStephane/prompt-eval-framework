"""A/B testing framework for prompt variant comparison.

Uses McNemar's test for paired categorical data and bootstrap
confidence intervals for robust statistical comparison of two
prompt variants.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from src.evaluation.metrics import CostMetrics, MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class ABResult:
    """Statistical comparison outcome of two prompt variants.

    Attributes:
        variant_a_accuracy: Pass rate for variant A.
        variant_b_accuracy: Pass rate for variant B.
        winner: Which variant won, or ``"tie"``.
        confidence: Confidence level (1 - p_value when significant).
        p_value: McNemar's test p-value.
        effect_size: Accuracy difference (B - A).
        ci_lower: Lower bound of bootstrap CI for the difference.
        ci_upper: Upper bound of bootstrap CI for the difference.
        recommendation: Human-readable recommendation string.
    """

    variant_a_accuracy: float
    variant_b_accuracy: float
    winner: Literal["A", "B", "tie"]
    confidence: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    recommendation: str


class ABTester:
    """Run paired A/B tests between two prompt variants.

    Args:
        metrics_calc: Calculator for cost/token metrics.
    """

    def __init__(self, metrics_calc: MetricsCalculator | None = None) -> None:
        self.metrics = metrics_calc or MetricsCalculator()

    def compare(
        self,
        results_a: list[bool],
        results_b: list[bool],
        n_bootstrap: int = 1000,
        cost_a: CostMetrics | None = None,
        cost_b: CostMetrics | None = None,
    ) -> ABResult:
        """Compare two sets of pass/fail results statistically.

        Args:
            results_a: Per-test-case pass/fail for variant A.
            results_b: Per-test-case pass/fail for variant B.
            n_bootstrap: Bootstrap iterations for CI.
            cost_a: Optional cost metrics for variant A.
            cost_b: Optional cost metrics for variant B.

        Returns:
            An :class:`ABResult` with statistical comparison.

        Raises:
            ValueError: If the result lists differ in length.
        """
        if len(results_a) != len(results_b):
            raise ValueError(
                f"Result lists must have same length: {len(results_a)} vs {len(results_b)}"
            )

        pairs = list(zip(results_a, results_b))

        p_value = self._mcnemar_test(pairs)

        acc_a = sum(results_a) / len(results_a) if results_a else 0
        acc_b = sum(results_b) / len(results_b) if results_b else 0
        ci_lower, ci_upper = self._bootstrap_ci(results_a, results_b, n_bootstrap)

        effect_size = acc_b - acc_a

        if p_value < 0.05:
            winner: Literal["A", "B", "tie"] = "B" if effect_size > 0 else "A"
            confidence = 1 - p_value
        else:
            winner = "tie"
            confidence = 0.0

        recommendation = self._generate_recommendation(
            winner, effect_size, p_value, cost_a, cost_b
        )

        return ABResult(
            variant_a_accuracy=acc_a,
            variant_b_accuracy=acc_b,
            winner=winner,
            confidence=confidence,
            p_value=p_value,
            effect_size=effect_size,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            recommendation=recommendation,
        )

    @staticmethod
    def _mcnemar_test(pairs: list[tuple[bool, bool]]) -> float:
        """McNemar's test for paired nominal data.

        Args:
            pairs: List of (a_passed, b_passed) tuples.

        Returns:
            The p-value (1.0 if no discordant pairs).
        """
        b = sum(1 for a, bp in pairs if a and not bp)
        c = sum(1 for a, bp in pairs if not a and bp)

        if b + c == 0:
            return 1.0

        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        return float(1 - stats.chi2.cdf(chi2, df=1))

    @staticmethod
    def _bootstrap_ci(
        results_a: list[bool],
        results_b: list[bool],
        n_bootstrap: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for accuracy difference (B - A).

        Args:
            results_a: Pass/fail for variant A.
            results_b: Pass/fail for variant B.
            n_bootstrap: Number of bootstrap samples.
            confidence: Confidence level.

        Returns:
            (lower, upper) bounds.
        """
        n = len(results_a)
        if n == 0:
            return 0.0, 0.0

        rng = np.random.default_rng(seed=42)
        arr_a = np.array(results_a, dtype=float)
        arr_b = np.array(results_b, dtype=float)
        diffs = []

        for _ in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            diff = np.mean(arr_b[indices]) - np.mean(arr_a[indices])
            diffs.append(diff)

        alpha = 1 - confidence
        return (
            float(np.percentile(diffs, alpha / 2 * 100)),
            float(np.percentile(diffs, (1 - alpha / 2) * 100)),
        )

    @staticmethod
    def _generate_recommendation(
        winner: str,
        effect: float,
        p_value: float,
        cost_a: CostMetrics | None,
        cost_b: CostMetrics | None,
    ) -> str:
        """Build a human-readable recommendation string.

        Args:
            winner: ``"A"``, ``"B"``, or ``"tie"``.
            effect: Accuracy difference (B - A).
            p_value: Statistical significance.
            cost_a: Variant A cost metrics.
            cost_b: Variant B cost metrics.

        Returns:
            Recommendation string.
        """
        if winner == "tie":
            if cost_a and cost_b:
                if cost_a.total_cost_usd < cost_b.total_cost_usd:
                    return "No significant accuracy difference. Recommend Variant A (lower cost)."
                if cost_b.total_cost_usd < cost_a.total_cost_usd:
                    return "No significant accuracy difference. Recommend Variant B (lower cost)."
            return "No significant difference between variants."

        effect_pct = abs(effect) * 100
        return (
            f"Recommend Variant {winner} with {effect_pct:.1f}% "
            f"accuracy improvement (p={p_value:.4f})."
        )
