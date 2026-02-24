"""Tests for YAML prompt template loading, versioning, and management."""

from pathlib import Path

import pytest
import yaml

from src.prompts.template_manager import (
    ModelConfig,
    PromptTemplate,
    TemplateManager,
)
from src.utils.database import Database


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Temporary DuckDB for versioning tests."""
    return Database(tmp_path / "test.duckdb")


@pytest.fixture()
def prompts_dir(tmp_path: Path) -> Path:
    """Create a prompts directory with a sample YAML file."""
    d = tmp_path / "prompts"
    d.mkdir()
    data = {
        "system_prompt": "You are helpful.",
        "user_prompt": "Hello {{ name }}",
        "variables": ["name"],
        "category": "test-cat",
        "model_config": {"temperature": 0.5, "max_tokens": 128},
        "few_shot_examples": [{"user": "Hi Alice", "assistant": "Hello Alice!"}],
        "metadata": {"author": "tester"},
    }
    (d / "sample.yaml").write_text(yaml.dump(data))
    return d


@pytest.fixture()
def manager(db: Database, prompts_dir: Path) -> TemplateManager:
    """A TemplateManager wired to the temp db and prompts dir."""
    return TemplateManager(db, prompts_dir)


class TestLoad:
    """TemplateManager.load should parse YAML into a PromptTemplate."""

    def test_loads_basic_fields(self, manager: TemplateManager) -> None:
        tpl = manager.load("sample")
        assert tpl.name == "sample"
        assert tpl.system_prompt == "You are helpful."
        assert tpl.user_prompt == "Hello {{ name }}"
        assert tpl.variables == ["name"]
        assert tpl.category == "test-cat"

    def test_loads_model_config(self, manager: TemplateManager) -> None:
        tpl = manager.load("sample")
        assert tpl.model_config.temperature == 0.5
        assert tpl.model_config.max_tokens == 128

    def test_loads_few_shot(self, manager: TemplateManager) -> None:
        tpl = manager.load("sample")
        assert len(tpl.few_shot_examples) == 1
        assert tpl.few_shot_examples[0].user == "Hi Alice"

    def test_loads_metadata(self, manager: TemplateManager) -> None:
        tpl = manager.load("sample")
        assert tpl.metadata["author"] == "tester"

    def test_missing_file_raises(self, manager: TemplateManager) -> None:
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent")

    def test_invalid_yaml_raises(self, db: Database, tmp_path: Path) -> None:
        d = tmp_path / "bad_prompts"
        d.mkdir()
        (d / "bad.yaml").write_text(": invalid: yaml: {{")
        mgr = TemplateManager(db, d)
        with pytest.raises(Exception):
            mgr.load("bad")


class TestVersioning:
    """Version auto-increment on content changes."""

    def test_first_load_is_version_1(self, manager: TemplateManager) -> None:
        tpl = manager.load("sample")
        assert tpl.version == 1

    def test_same_content_same_version(self, manager: TemplateManager) -> None:
        tpl1 = manager.load("sample")
        tpl2 = manager.load("sample")
        assert tpl1.version == tpl2.version

    def test_changed_content_bumps_version(
        self, db: Database, prompts_dir: Path
    ) -> None:
        mgr = TemplateManager(db, prompts_dir)
        tpl1 = mgr.load("sample")
        assert tpl1.version == 1

        # Modify the YAML file
        data = yaml.safe_load((prompts_dir / "sample.yaml").read_text())
        data["user_prompt"] = "Goodbye {{ name }}"
        (prompts_dir / "sample.yaml").write_text(yaml.dump(data))

        mgr._cache.clear()
        tpl2 = mgr.load("sample")
        assert tpl2.version == 2


class TestContentHash:
    """PromptTemplate.content_hash consistency."""

    def test_same_input_same_hash(self) -> None:
        tpl = PromptTemplate(
            name="x",
            system_prompt="sys",
            user_prompt="user",
            variables=[],
            model_config=ModelConfig(),
        )
        assert tpl.content_hash() == tpl.content_hash()

    def test_different_user_prompt_different_hash(self) -> None:
        tpl_a = PromptTemplate(
            name="x",
            system_prompt="sys",
            user_prompt="a",
            variables=[],
            model_config=ModelConfig(),
        )
        tpl_b = PromptTemplate(
            name="x",
            system_prompt="sys",
            user_prompt="b",
            variables=[],
            model_config=ModelConfig(),
        )
        assert tpl_a.content_hash() != tpl_b.content_hash()


class TestListing:
    """Listing and category filtering."""

    def test_list_prompts(self, manager: TemplateManager) -> None:
        names = manager.list_prompts()
        assert "sample" in names

    def test_list_by_category(self, manager: TemplateManager) -> None:
        names = manager.list_by_category("test-cat")
        assert "sample" in names

    def test_list_by_category_empty(self, manager: TemplateManager) -> None:
        names = manager.list_by_category("nonexistent")
        assert names == []


class TestExportImport:
    """Round-trip export and import of prompt collections."""

    def test_export_and_import(
        self, db: Database, prompts_dir: Path, tmp_path: Path
    ) -> None:
        mgr = TemplateManager(db, prompts_dir)
        mgr.load("sample")

        export_path = tmp_path / "collection.yaml"
        mgr.export_collection(["sample"], export_path)
        assert export_path.exists()

        # Import into a fresh directory
        new_dir = tmp_path / "imported"
        new_dir.mkdir()
        mgr2 = TemplateManager(db, new_dir)
        imported = mgr2.import_collection(export_path)
        assert "sample" in imported
        assert (new_dir / "sample.yaml").exists()

        tpl = mgr2.load("sample")
        assert tpl.system_prompt == "You are helpful."
