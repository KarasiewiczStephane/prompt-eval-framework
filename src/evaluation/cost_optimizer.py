"""Cost estimation, budget enforcement, and optimization recommendations.

Estimates token counts and costs before running evaluations, enforces
budget limits, and suggests cheaper models with similar accuracy.
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.evaluation.metrics import PRICING, ModelMetrics

logger = logging.getLogger(__name__)

# Average output tokens by task type (calibrate to your workload)
OUTPUT_ESTIMATES: dict[str, int] = {
    "classification": 10,
    "extraction": 100,
    "generation": 300,
    "summarization": 200,
    "default": 150,
}


@dataclass
class CostEstimate:
    """Pre-run cost estimation result.

    Attributes:
        estimated_input_tokens: Total estimated input tokens.
        estimated_output_tokens: Total estimated output tokens.
        estimated_cost_usd: Estimated total cost.
        breakdown_by_model: Per-model cost breakdown.
        within_budget: Whether the estimate fits the budget.
        budget_utilization: Percentage of budget the estimate uses.
    """

    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    breakdown_by_model: dict[str, float]
    within_budget: bool
    budget_utilization: float


@dataclass
class OptimizationRecommendation:
    """Model optimization recommendation.

    Attributes:
        current_model: The currently best-accuracy model.
        current_cost: Cost of the current model.
        current_accuracy: Accuracy of the current model.
        recommended_model: The suggested cheaper model.
        recommended_cost: Cost of the recommendation.
        expected_accuracy: Accuracy of the recommendation.
        savings_percent: Percentage cost reduction.
        accuracy_tradeoff: Percentage accuracy loss.
        recommendation: Human-readable recommendation.
    """

    current_model: str
    current_cost: float
    current_accuracy: float
    recommended_model: str
    recommended_cost: float
    expected_accuracy: float
    savings_percent: float
    accuracy_tradeoff: float
    recommendation: str


class CostOptimizer:
    """Estimate costs, enforce budgets, and recommend optimisations.

    Args:
        pricing: Per-model token pricing (per 1 M tokens).
    """

    def __init__(self, pricing: dict[str, dict[str, float]] | None = None) -> None:
        self.pricing = pricing or PRICING

    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Estimate token count for a text string.

        Uses a character-based heuristic (~4 chars per token) to avoid
        requiring tiktoken as a hard dependency.

        Args:
            text: Input text.
            model: Model name (unused in heuristic mode, reserved).

        Returns:
            Estimated token count.
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            return max(len(text) // 4, 1)

    def estimate_cost(
        self,
        prompts: list[str],
        models: list[str],
        task_type: str = "default",
        budget: float | None = None,
    ) -> CostEstimate:
        """Estimate the cost of running prompts across models.

        Args:
            prompts: List of prompt texts to estimate.
            models: List of model names.
            task_type: Task category for output token estimation.
            budget: Optional budget cap in USD.

        Returns:
            A :class:`CostEstimate`.
        """
        total_input = sum(self.estimate_tokens(p, models[0]) for p in prompts)
        avg_output = OUTPUT_ESTIMATES.get(task_type, OUTPUT_ESTIMATES["default"])
        total_output = avg_output * len(prompts)

        breakdown: dict[str, float] = {}
        for model in models:
            pricing = self.pricing.get(model, {"input": 10, "output": 30})
            cost = (total_input / 1_000_000) * pricing["input"] + (
                total_output / 1_000_000
            ) * pricing["output"]
            breakdown[model] = cost

        total_cost = sum(breakdown.values())

        return CostEstimate(
            estimated_input_tokens=total_input,
            estimated_output_tokens=total_output * len(models),
            estimated_cost_usd=total_cost,
            breakdown_by_model=breakdown,
            within_budget=budget is None or total_cost <= budget,
            budget_utilization=(total_cost / budget * 100) if budget else 0,
        )

    def recommend_cheaper_model(
        self,
        model_metrics: dict[str, ModelMetrics],
        accuracy_threshold: float = 0.95,
    ) -> OptimizationRecommendation | None:
        """Recommend a cheaper model maintaining similar accuracy.

        Args:
            model_metrics: Model name to metrics mapping.
            accuracy_threshold: Minimum fraction of best accuracy to accept.

        Returns:
            A recommendation, or ``None`` if no metrics available.
        """
        if not model_metrics:
            return None

        best_model = max(
            model_metrics.keys(),
            key=lambda m: model_metrics[m].accuracy.accuracy,
        )
        best_accuracy = model_metrics[best_model].accuracy.accuracy
        best_cost = model_metrics[best_model].cost.total_cost_usd

        candidates = [
            (m, metrics)
            for m, metrics in model_metrics.items()
            if metrics.accuracy.accuracy >= best_accuracy * accuracy_threshold
        ]

        if not candidates:
            return None

        cheapest_model, cheapest_metrics = min(
            candidates, key=lambda x: x[1].cost.total_cost_usd
        )

        if cheapest_model == best_model:
            return OptimizationRecommendation(
                current_model=best_model,
                current_cost=best_cost,
                current_accuracy=best_accuracy,
                recommended_model=best_model,
                recommended_cost=best_cost,
                expected_accuracy=best_accuracy,
                savings_percent=0,
                accuracy_tradeoff=0,
                recommendation=f"{best_model} is already the most cost-effective option.",
            )

        savings = (
            (best_cost - cheapest_metrics.cost.total_cost_usd) / best_cost * 100
            if best_cost > 0
            else 0
        )
        accuracy_loss = (
            (best_accuracy - cheapest_metrics.accuracy.accuracy) / best_accuracy * 100
            if best_accuracy > 0
            else 0
        )

        return OptimizationRecommendation(
            current_model=best_model,
            current_cost=best_cost,
            current_accuracy=best_accuracy,
            recommended_model=cheapest_model,
            recommended_cost=cheapest_metrics.cost.total_cost_usd,
            expected_accuracy=cheapest_metrics.accuracy.accuracy,
            savings_percent=savings,
            accuracy_tradeoff=accuracy_loss,
            recommendation=(
                f"Switch to {cheapest_model}: Save {savings:.1f}% cost "
                f"with only {accuracy_loss:.1f}% accuracy loss."
            ),
        )

    def suggest_prompt_compression(
        self, prompt: str, target_reduction: float = 0.2
    ) -> dict[str, Any]:
        """Suggest ways to shorten a prompt.

        Args:
            prompt: The prompt text to analyze.
            target_reduction: Target fraction to reduce by.

        Returns:
            Dict with original/target tokens, suggestions, and savings.
        """
        original_tokens = self.estimate_tokens(prompt, "gpt-4")
        target_tokens = int(original_tokens * (1 - target_reduction))

        suggestions: list[str] = []

        if "please" in prompt.lower() or "kindly" in prompt.lower():
            suggestions.append(
                "Remove politeness words (please, kindly) — LLMs don't need them"
            )

        if prompt.count("\n\n") > 3:
            suggestions.append("Reduce excessive whitespace and blank lines")

        if len(prompt.split()) > 500:
            suggestions.append("Consider using bullet points instead of paragraphs")

        if prompt.lower().count("make sure") > 1 or prompt.lower().count("ensure") > 1:
            suggestions.append("Consolidate repeated instructions")

        return {
            "original_tokens": original_tokens,
            "target_tokens": target_tokens,
            "suggestions": suggestions,
            "estimated_savings_usd": (original_tokens - target_tokens) / 1_000_000 * 10,
        }


class BudgetEnforcer:
    """Track spend against a fixed budget.

    Args:
        budget_usd: Maximum allowed spend in USD.
    """

    def __init__(self, budget_usd: float) -> None:
        self.budget = budget_usd
        self.spent = 0.0

    def can_spend(self, amount: float) -> bool:
        """Check if an expenditure fits within the remaining budget.

        Args:
            amount: Proposed spend in USD.

        Returns:
            ``True`` if within budget.
        """
        return self.spent + amount <= self.budget

    def record_spend(self, amount: float) -> None:
        """Record an expenditure.

        Args:
            amount: Amount spent in USD.
        """
        self.spent += amount

    @property
    def remaining(self) -> float:
        """USD remaining in the budget."""
        return self.budget - self.spent
