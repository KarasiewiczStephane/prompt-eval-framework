"""Tests for configuration loading from env and YAML."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import Config


@pytest.fixture()
def config_yaml(tmp_path: Path) -> Path:
    """Write a minimal config YAML and return its path."""
    cfg = {
        "database": {"path": str(tmp_path / "test.duckdb")},
        "defaults": {
            "temperature": 0.5,
            "max_tokens": 512,
            "top_p": 0.9,
            "timeout": 30.0,
        },
        "paths": {
            "prompts_dir": "my_prompts",
            "suites_dir": "my_suites",
            "reports_dir": "my_reports",
        },
        "logging": {"level": "DEBUG"},
        "pricing": {
            "gpt-4": {"input": 30.0, "output": 60.0},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


class TestConfigFromEnv:
    """Config.from_env should read API keys and DB_PATH from the env."""

    def test_reads_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
        cfg = Config.from_env()
        assert cfg.openai_api_key == "sk-test-openai"
        assert cfg.anthropic_api_key == "sk-test-anthropic"

    def test_missing_keys_are_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = Config.from_env()
        assert cfg.openai_api_key is None
        assert cfg.anthropic_api_key is None

    def test_custom_db_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_PATH", "/tmp/custom.duckdb")
        cfg = Config.from_env()
        assert cfg.db_path == Path("/tmp/custom.duckdb")


class TestConfigFromYaml:
    """Config.from_yaml should parse values from a YAML file."""

    def test_loads_all_fields(
        self, config_yaml: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = Config.from_yaml(config_yaml)
        assert cfg.default_temperature == 0.5
        assert cfg.default_max_tokens == 512
        assert cfg.default_top_p == 0.9
        assert cfg.default_timeout == 30.0
        assert cfg.prompts_dir == Path("my_prompts")
        assert cfg.log_level == "DEBUG"
        assert "gpt-4" in cfg.pricing

    def test_env_keys_override(
        self, config_yaml: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        cfg = Config.from_yaml(config_yaml)
        assert cfg.openai_api_key == "from-env"

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(Path("/nonexistent/config.yaml"))


class TestRequireKeys:
    """require_*_key helpers should raise when keys are missing."""

    def test_require_openai_key_raises(self) -> None:
        cfg = Config(openai_api_key=None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            cfg.require_openai_key()

    def test_require_openai_key_returns(self) -> None:
        cfg = Config(openai_api_key="sk-ok")
        assert cfg.require_openai_key() == "sk-ok"

    def test_require_anthropic_key_raises(self) -> None:
        cfg = Config(anthropic_api_key=None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            cfg.require_anthropic_key()

    def test_require_anthropic_key_returns(self) -> None:
        cfg = Config(anthropic_api_key="sk-ok")
        assert cfg.require_anthropic_key() == "sk-ok"
