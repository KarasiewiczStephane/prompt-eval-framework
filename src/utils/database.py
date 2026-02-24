"""DuckDB database setup and connection management.

Handles schema creation, connection pooling via context manager,
and provides the single source of truth for the evaluation schema.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
CREATE SEQUENCE IF NOT EXISTS runs_id_seq START 1;
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER DEFAULT nextval('runs_id_seq') PRIMARY KEY,
    suite_name VARCHAR,
    model VARCHAR,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    config JSON
);

CREATE SEQUENCE IF NOT EXISTS results_id_seq START 1;
CREATE TABLE IF NOT EXISTS results (
    id INTEGER DEFAULT nextval('results_id_seq') PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    test_case_id VARCHAR,
    prompt_version VARCHAR,
    input_vars JSON,
    expected_output TEXT,
    actual_output TEXT,
    passed BOOLEAN,
    latency_ms FLOAT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd FLOAT,
    error TEXT
);

CREATE SEQUENCE IF NOT EXISTS prompt_versions_id_seq START 1;
CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER DEFAULT nextval('prompt_versions_id_seq') PRIMARY KEY,
    prompt_name VARCHAR,
    version INTEGER,
    content JSON,
    created_at TIMESTAMP,
    hash VARCHAR UNIQUE
);
"""


class Database:
    """Thin wrapper around a DuckDB database file.

    On construction the evaluation schema is created if it does not
    already exist.  Use :meth:`connection` as a context manager to
    obtain a short-lived connection.

    Args:
        db_path: Filesystem path for the DuckDB file.  Parent
            directories are created automatically.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and sequences if they don't exist."""
        with self.connection() as conn:
            conn.execute(_SCHEMA_SQL)
        logger.info("Database schema initialised at %s", self.db_path)

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Yield a DuckDB connection, closing it on exit.

        Yields:
            A live DuckDB connection.
        """
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
