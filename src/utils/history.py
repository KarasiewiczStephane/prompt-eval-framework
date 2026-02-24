"""DuckDB-based logging and history management for evaluation runs.

Provides a high-level interface for creating runs, logging results,
and querying historical evaluation data with filtering and aggregation.
"""

import json
import logging
from datetime import datetime
from typing import Any

from src.utils.database import Database

logger = logging.getLogger(__name__)


class HistoryManager:
    """Log evaluation runs and query historical data.

    Args:
        db: A :class:`Database` instance.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def create_run(
        self,
        suite_name: str,
        model: str,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Start a new evaluation run.

        Args:
            suite_name: Name of the test suite.
            model: Model identifier.
            config: Optional configuration snapshot.

        Returns:
            The new run ID.
        """
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO runs (suite_name, model, started_at, config) "
                "VALUES (?, ?, ?, ?)",
                [suite_name, model, datetime.now(), json.dumps(config or {})],
            )
            row = conn.execute("SELECT MAX(id) FROM runs").fetchone()
            run_id = row[0]

        logger.info("Created run %d for suite=%s model=%s", run_id, suite_name, model)
        return run_id

    def complete_run(self, run_id: int) -> None:
        """Mark a run as completed.

        Args:
            run_id: The run to complete.
        """
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE runs SET completed_at = ? WHERE id = ?",
                [datetime.now(), run_id],
            )
        logger.info("Completed run %d", run_id)

    def log_result(
        self,
        run_id: int,
        test_case_id: str,
        prompt_version: str,
        input_vars: dict[str, Any],
        expected_output: str,
        actual_output: str,
        passed: bool,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        error: str | None = None,
    ) -> None:
        """Log a single test-case result.

        Args:
            run_id: Parent run ID.
            test_case_id: Test case identifier.
            prompt_version: Prompt template version string.
            input_vars: Variable mapping used.
            expected_output: Expected output text.
            actual_output: Model's actual output.
            passed: Whether the assertion passed.
            latency_ms: Request latency in ms.
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost_usd: Cost of this request.
            error: Optional error message.
        """
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO results "
                "(run_id, test_case_id, prompt_version, input_vars, "
                "expected_output, actual_output, passed, latency_ms, "
                "input_tokens, output_tokens, cost_usd, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    test_case_id,
                    prompt_version,
                    json.dumps(input_vars),
                    expected_output,
                    actual_output,
                    passed,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    error,
                ],
            )

    def log_results_batch(self, run_id: int, results: list[dict[str, Any]]) -> None:
        """Log multiple results in a single transaction.

        Args:
            run_id: Parent run ID.
            results: List of result dicts with the same keys as
                :meth:`log_result`.
        """
        for r in results:
            self.log_result(run_id=run_id, **r)
        logger.info("Logged %d results for run %d", len(results), run_id)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Fetch a single run's metadata.

        Args:
            run_id: The run ID.

        Returns:
            Dict with run fields, or ``None`` if not found.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT id, suite_name, model, started_at, completed_at, config "
                "FROM runs WHERE id = ?",
                [run_id],
            ).fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "suite_name": row[1],
            "model": row[2],
            "started_at": row[3],
            "completed_at": row[4],
            "config": row[5],
        }

    def get_run_results(self, run_id: int) -> list[dict[str, Any]]:
        """Fetch all results for a run.

        Args:
            run_id: The run ID.

        Returns:
            List of result dicts.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT test_case_id, passed, latency_ms, "
                "input_tokens, output_tokens, cost_usd, error "
                "FROM results WHERE run_id = ?",
                [run_id],
            ).fetchall()

        return [
            {
                "test_case_id": r[0],
                "passed": r[1],
                "latency_ms": r[2],
                "input_tokens": r[3],
                "output_tokens": r[4],
                "cost_usd": r[5],
                "error": r[6],
            }
            for r in rows
        ]

    def list_runs(
        self, suite_name: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List recent runs, optionally filtered by suite.

        Args:
            suite_name: Optional filter.
            limit: Max number of runs to return.

        Returns:
            List of run summary dicts, newest first.
        """
        with self.db.connection() as conn:
            if suite_name:
                rows = conn.execute(
                    "SELECT id, suite_name, model, started_at, completed_at "
                    "FROM runs WHERE suite_name = ? "
                    "ORDER BY id DESC LIMIT ?",
                    [suite_name, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, suite_name, model, started_at, completed_at "
                    "FROM runs ORDER BY id DESC LIMIT ?",
                    [limit],
                ).fetchall()

        return [
            {
                "id": r[0],
                "suite_name": r[1],
                "model": r[2],
                "started_at": r[3],
                "completed_at": r[4],
            }
            for r in rows
        ]

    def get_run_summary(self, run_id: int) -> dict[str, Any] | None:
        """Compute aggregate statistics for a run.

        Args:
            run_id: The run ID.

        Returns:
            Summary dict with counts and averages, or ``None``.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed, "
                "AVG(latency_ms) as avg_latency, "
                "SUM(cost_usd) as total_cost, "
                "SUM(input_tokens) as total_input_tokens, "
                "SUM(output_tokens) as total_output_tokens "
                "FROM results WHERE run_id = ?",
                [run_id],
            ).fetchone()

        if row is None or row[0] == 0:
            return None

        return {
            "total": row[0],
            "passed": row[1],
            "failed": row[0] - row[1],
            "accuracy": row[1] / row[0] if row[0] > 0 else 0,
            "avg_latency_ms": row[2],
            "total_cost_usd": row[3],
            "total_input_tokens": row[4],
            "total_output_tokens": row[5],
        }
