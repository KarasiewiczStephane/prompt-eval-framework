"""Tests for prompt version history queries."""

from pathlib import Path

import pytest
import yaml

from src.prompts.template_manager import TemplateManager
from src.prompts.version_control import VersionControl
from src.utils.database import Database


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Temporary DuckDB."""
    return Database(tmp_path / "test.duckdb")


@pytest.fixture()
def prompts_dir(tmp_path: Path) -> Path:
    """Sample prompts directory."""
    d = tmp_path / "prompts"
    d.mkdir()
    data = {
        "system_prompt": "Sys",
        "user_prompt": "Hello {{ name }}",
        "variables": ["name"],
    }
    (d / "test_prompt.yaml").write_text(yaml.dump(data))
    return d


@pytest.fixture()
def vc_with_versions(
    db: Database, prompts_dir: Path
) -> tuple[VersionControl, TemplateManager]:
    """Create two versions and return VC + manager."""
    mgr = TemplateManager(db, prompts_dir)
    mgr.load("test_prompt")

    # Mutate and reload to create v2
    data = yaml.safe_load((prompts_dir / "test_prompt.yaml").read_text())
    data["user_prompt"] = "Goodbye {{ name }}"
    (prompts_dir / "test_prompt.yaml").write_text(yaml.dump(data))
    mgr._cache.clear()
    mgr.load("test_prompt")

    return VersionControl(db), mgr


class TestGetHistory:
    """get_history should list all versions newest-first."""

    def test_returns_all_versions(
        self, vc_with_versions: tuple[VersionControl, TemplateManager]
    ) -> None:
        vc, _ = vc_with_versions
        history = vc.get_history("test_prompt")
        assert len(history) == 2
        assert history[0]["version"] == 2
        assert history[1]["version"] == 1

    def test_empty_for_unknown_prompt(self, db: Database) -> None:
        vc = VersionControl(db)
        assert vc.get_history("unknown") == []


class TestGetVersion:
    """get_version should retrieve specific version content."""

    def test_returns_content_dict(
        self, vc_with_versions: tuple[VersionControl, TemplateManager]
    ) -> None:
        vc, _ = vc_with_versions
        v1 = vc.get_version("test_prompt", 1)
        assert v1 is not None
        assert "system_prompt" in v1

    def test_returns_none_for_missing(self, db: Database) -> None:
        vc = VersionControl(db)
        assert vc.get_version("unknown", 99) is None


class TestDiff:
    """diff should return side-by-side version content."""

    def test_diff_two_versions(
        self, vc_with_versions: tuple[VersionControl, TemplateManager]
    ) -> None:
        vc, _ = vc_with_versions
        result = vc.diff("test_prompt", 1, 2)
        assert result["old"] is not None
        assert result["new"] is not None
        assert result["old"] != result["new"]

    def test_diff_missing_version(self, db: Database) -> None:
        vc = VersionControl(db)
        result = vc.diff("unknown", 1, 2)
        assert result["old"] is None
        assert result["new"] is None
