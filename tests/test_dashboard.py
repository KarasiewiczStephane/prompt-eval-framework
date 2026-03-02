"""Tests for the prompt evaluation framework dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    generate_ab_test_results,
    generate_accuracy_by_tag,
    generate_eval_results,
    generate_token_usage,
)


class TestEvalResults:
    def test_returns_dataframe(self) -> None:
        df = generate_eval_results()
        assert isinstance(df, pd.DataFrame)

    def test_has_four_models(self) -> None:
        df = generate_eval_results()
        assert len(df) == 4

    def test_has_required_columns(self) -> None:
        df = generate_eval_results()
        for col in ["model", "total_tests", "passed", "failed", "accuracy"]:
            assert col in df.columns

    def test_passed_plus_failed_equals_total(self) -> None:
        df = generate_eval_results()
        assert (df["passed"] + df["failed"] == df["total_tests"]).all()

    def test_accuracy_bounded(self) -> None:
        df = generate_eval_results()
        assert (df["accuracy"] >= 0).all()
        assert (df["accuracy"] <= 1).all()

    def test_reproducible(self) -> None:
        df1 = generate_eval_results(seed=99)
        df2 = generate_eval_results(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestTokenUsage:
    def test_returns_dataframe(self) -> None:
        df = generate_token_usage()
        assert isinstance(df, pd.DataFrame)

    def test_has_four_models(self) -> None:
        df = generate_token_usage()
        assert len(df) == 4

    def test_costs_positive(self) -> None:
        df = generate_token_usage()
        assert (df["total_cost"] > 0).all()

    def test_tokens_positive(self) -> None:
        df = generate_token_usage()
        assert (df["input_tokens"] > 0).all()
        assert (df["output_tokens"] > 0).all()


class TestAccuracyByTag:
    def test_returns_dataframe(self) -> None:
        df = generate_accuracy_by_tag()
        assert isinstance(df, pd.DataFrame)

    def test_has_six_tags(self) -> None:
        df = generate_accuracy_by_tag()
        assert len(df) == 6

    def test_accuracy_bounded(self) -> None:
        df = generate_accuracy_by_tag()
        assert (df["accuracy"] >= 0).all()
        assert (df["accuracy"] <= 1).all()

    def test_test_count_positive(self) -> None:
        df = generate_accuracy_by_tag()
        assert (df["test_count"] > 0).all()


class TestAbTestResults:
    def test_returns_dataframe(self) -> None:
        df = generate_ab_test_results()
        assert isinstance(df, pd.DataFrame)

    def test_has_four_metrics(self) -> None:
        df = generate_ab_test_results()
        assert len(df) == 4

    def test_p_values_bounded(self) -> None:
        df = generate_ab_test_results()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1).all()

    def test_has_both_variants(self) -> None:
        df = generate_ab_test_results()
        assert "variant_a" in df.columns
        assert "variant_b" in df.columns
