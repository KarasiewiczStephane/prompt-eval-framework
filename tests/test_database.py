"""Tests for DuckDB database setup and CRUD operations."""

from datetime import datetime
from pathlib import Path

import pytest

from src.utils.database import Database


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Create a temporary database for testing."""
    return Database(tmp_path / "test.duckdb")


class TestDatabaseInit:
    """Schema should be created on Database instantiation."""

    def test_creates_file(self, db: Database) -> None:
        assert db.db_path.exists()

    def test_tables_exist(self, db: Database) -> None:
        with db.connection() as conn:
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
            table_names = {row[0] for row in tables}
        assert "runs" in table_names
        assert "results" in table_names
        assert "prompt_versions" in table_names

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "test.duckdb"
        Database(nested)
        assert nested.parent.exists()


class TestDatabaseCRUD:
    """Basic insert / select operations on the evaluation tables."""

    def test_insert_and_select_run(self, db: Database) -> None:
        now = datetime.now()
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO runs (suite_name, model, started_at) VALUES (?, ?, ?)",
                ["my_suite", "gpt-4", now],
            )
            rows = conn.execute("SELECT suite_name, model FROM runs").fetchall()
        assert len(rows) == 1
        assert rows[0] == ("my_suite", "gpt-4")

    def test_insert_and_select_result(self, db: Database) -> None:
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO runs (suite_name, model, started_at) VALUES (?, ?, ?)",
                ["s", "m", datetime.now()],
            )
            run_id = conn.execute("SELECT id FROM runs").fetchone()[0]
            conn.execute(
                "INSERT INTO results (run_id, test_case_id, passed, latency_ms) "
                "VALUES (?, ?, ?, ?)",
                [run_id, "tc_1", True, 123.4],
            )
            rows = conn.execute(
                "SELECT test_case_id, passed, latency_ms FROM results"
            ).fetchall()
        assert rows[0][0] == "tc_1"
        assert rows[0][1] is True
        assert rows[0][2] == pytest.approx(123.4, rel=1e-4)

    def test_insert_prompt_version(self, db: Database) -> None:
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO prompt_versions "
                "(prompt_name, version, content, created_at, hash) "
                "VALUES (?, ?, ?, ?, ?)",
                ["greeting", 1, '{"system":"hi"}', datetime.now(), "abc123"],
            )
            row = conn.execute(
                "SELECT prompt_name, version, hash FROM prompt_versions"
            ).fetchone()
        assert row == ("greeting", 1, "abc123")

    def test_unique_hash_constraint(self, db: Database) -> None:
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO prompt_versions "
                "(prompt_name, version, content, created_at, hash) "
                "VALUES (?, ?, ?, ?, ?)",
                ["p", 1, "{}", datetime.now(), "same_hash"],
            )
            with pytest.raises(Exception):
                conn.execute(
                    "INSERT INTO prompt_versions "
                    "(prompt_name, version, content, created_at, hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ["p", 2, "{}", datetime.now(), "same_hash"],
                )


class TestDatabaseConnection:
    """Connection context manager should properly close."""

    def test_connection_closes(self, db: Database) -> None:
        with db.connection() as conn:
            conn.execute("SELECT 1")
        # After exit, connection should be closed — calling execute raises
        with pytest.raises(Exception):
            conn.execute("SELECT 1")
