"""Version history queries for prompt templates.

Provides read-only access to the ``prompt_versions`` table so callers
can inspect, compare, and roll back template revisions.
"""

import json
import logging
from typing import Any

from src.utils.database import Database

logger = logging.getLogger(__name__)


class VersionControl:
    """Query version history stored in the database.

    Args:
        db: A :class:`Database` instance.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_history(self, prompt_name: str) -> list[dict[str, Any]]:
        """Retrieve the full version history for a prompt.

        Args:
            prompt_name: The prompt identifier.

        Returns:
            List of dicts with *version*, *created_at*, and *hash* keys,
            ordered newest-first.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT version, created_at, hash "
                "FROM prompt_versions "
                "WHERE prompt_name = ? ORDER BY version DESC",
                [prompt_name],
            ).fetchall()

        return [
            {"version": row[0], "created_at": row[1], "hash": row[2]} for row in rows
        ]

    def get_version(self, prompt_name: str, version: int) -> dict[str, Any] | None:
        """Retrieve a specific prompt version's stored content.

        Args:
            prompt_name: The prompt identifier.
            version: The version number to fetch.

        Returns:
            Parsed content dict, or ``None`` if not found.
        """
        with self.db.connection() as conn:
            result = conn.execute(
                "SELECT content FROM prompt_versions "
                "WHERE prompt_name = ? AND version = ?",
                [prompt_name, version],
            ).fetchone()

        if result is None:
            return None
        content = result[0]
        if isinstance(content, str):
            return json.loads(content)
        return content

    def diff(
        self, prompt_name: str, v1: int, v2: int
    ) -> dict[str, dict[str, Any] | None]:
        """Return side-by-side content of two versions.

        Args:
            prompt_name: The prompt identifier.
            v1: First (older) version number.
            v2: Second (newer) version number.

        Returns:
            Dict with ``old`` and ``new`` keys mapping to the respective
            content dicts (or ``None`` if a version doesn't exist).
        """
        old = self.get_version(prompt_name, v1)
        new = self.get_version(prompt_name, v2)
        return {"old": old, "new": new}
