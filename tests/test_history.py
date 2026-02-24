"""Tests for DuckDB logging and history management."""

from pathlib import Path

import pytest

from src.utils.database import Database
from src.utils.history import HistoryManager


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.duckdb")


@pytest.fixture()
def hm(db: Database) -> HistoryManager:
    return HistoryManager(db)


class TestCreateRun:
    """Run creation should return a valid ID."""

    def test_returns_positive_id(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("suite_a", "gpt-4")
        assert run_id > 0

    def test_sequential_ids(self, hm: HistoryManager) -> None:
        id1 = hm.create_run("s", "m")
        id2 = hm.create_run("s", "m")
        assert id2 > id1

    def test_stores_config(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("s", "m", config={"key": "val"})
        run = hm.get_run(run_id)
        assert run is not None


class TestCompleteRun:
    """complete_run should set completed_at."""

    def test_sets_completed_at(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("s", "m")
        hm.complete_run(run_id)
        run = hm.get_run(run_id)
        assert run["completed_at"] is not None


class TestLogResult:
    """log_result should persist test results."""

    def test_logs_single_result(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("s", "m")
        hm.log_result(
            run_id=run_id,
            test_case_id="tc_1",
            prompt_version="v1",
            input_vars={"name": "Alice"},
            expected_output="hello",
            actual_output="hello!",
            passed=True,
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        )
        results = hm.get_run_results(run_id)
        assert len(results) == 1
        assert results[0]["test_case_id"] == "tc_1"
        assert results[0]["passed"] is True


class TestLogResultsBatch:
    """Batch logging should store multiple results."""

    def test_logs_batch(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("s", "m")
        batch = [
            {
                "test_case_id": f"tc_{i}",
                "prompt_version": "v1",
                "input_vars": {},
                "expected_output": "x",
                "actual_output": "y",
                "passed": i % 2 == 0,
                "latency_ms": 50.0,
                "input_tokens": 5,
                "output_tokens": 3,
                "cost_usd": 0.001,
            }
            for i in range(5)
        ]
        hm.log_results_batch(run_id, batch)
        results = hm.get_run_results(run_id)
        assert len(results) == 5


class TestGetRun:
    """get_run should return run metadata or None."""

    def test_existing_run(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("my_suite", "gpt-4")
        run = hm.get_run(run_id)
        assert run is not None
        assert run["suite_name"] == "my_suite"
        assert run["model"] == "gpt-4"

    def test_missing_run(self, hm: HistoryManager) -> None:
        assert hm.get_run(9999) is None


class TestListRuns:
    """list_runs should return recent runs."""

    def test_lists_all(self, hm: HistoryManager) -> None:
        hm.create_run("a", "m")
        hm.create_run("b", "m")
        runs = hm.list_runs()
        assert len(runs) >= 2

    def test_filters_by_suite(self, hm: HistoryManager) -> None:
        hm.create_run("target", "m")
        hm.create_run("other", "m")
        runs = hm.list_runs(suite_name="target")
        assert all(r["suite_name"] == "target" for r in runs)

    def test_respects_limit(self, hm: HistoryManager) -> None:
        for _ in range(5):
            hm.create_run("s", "m")
        runs = hm.list_runs(limit=3)
        assert len(runs) <= 3


class TestGetRunSummary:
    """get_run_summary should aggregate results."""

    def test_computes_summary(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("s", "m")
        for i in range(4):
            hm.log_result(
                run_id=run_id,
                test_case_id=f"tc_{i}",
                prompt_version="v1",
                input_vars={},
                expected_output="x",
                actual_output="y",
                passed=i < 3,
                latency_ms=100.0 + i * 10,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.01,
            )
        summary = hm.get_run_summary(run_id)
        assert summary is not None
        assert summary["total"] == 4
        assert summary["passed"] == 3
        assert summary["failed"] == 1
        assert summary["accuracy"] == pytest.approx(0.75)
        assert summary["total_cost_usd"] == pytest.approx(0.04)

    def test_empty_run(self, hm: HistoryManager) -> None:
        run_id = hm.create_run("s", "m")
        assert hm.get_run_summary(run_id) is None
