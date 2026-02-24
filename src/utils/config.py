"""Configuration management for the prompt evaluation framework.

Loads settings from environment variables, YAML config files, or both.
API keys come from the environment; all other settings from config.yaml.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration container.

    Attributes:
        openai_api_key: OpenAI API key (from environment).
        anthropic_api_key: Anthropic API key (from environment).
        db_path: Path to the DuckDB database file.
        default_temperature: Default sampling temperature for model calls.
        default_max_tokens: Default maximum output tokens.
        default_top_p: Default nucleus-sampling parameter.
        default_timeout: Default request timeout in seconds.
        prompts_dir: Directory containing prompt YAML templates.
        suites_dir: Directory containing test-suite definitions.
        reports_dir: Directory for generated HTML reports.
        log_level: Python logging level name.
        pricing: Per-model token pricing (per 1 M tokens).
    """

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    db_path: Path = Path("data/prompteval.duckdb")
    default_temperature: float = 0.7
    default_max_tokens: int = 1024
    default_top_p: float = 1.0
    default_timeout: float = 60.0
    prompts_dir: Path = Path("example_prompts")
    suites_dir: Path = Path("example_suites")
    reports_dir: Path = Path("reports")
    log_level: str = "INFO"
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables alone.

        Returns:
            A Config instance populated from ``os.environ``.
        """
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            db_path=Path(os.getenv("DB_PATH", "data/prompteval.duckdb")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Build a Config by merging a YAML file with environment variables.

        Args:
            path: Filesystem path to a YAML config file.

        Returns:
            A Config instance with YAML values overridden by env-var API keys.

        Raises:
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        with open(path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        database_cfg = data.get("database", {})
        defaults_cfg = data.get("defaults", {})
        paths_cfg = data.get("paths", {})
        logging_cfg = data.get("logging", {})

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            db_path=Path(database_cfg.get("path", "data/prompteval.duckdb")),
            default_temperature=defaults_cfg.get("temperature", 0.7),
            default_max_tokens=defaults_cfg.get("max_tokens", 1024),
            default_top_p=defaults_cfg.get("top_p", 1.0),
            default_timeout=defaults_cfg.get("timeout", 60.0),
            prompts_dir=Path(paths_cfg.get("prompts_dir", "example_prompts")),
            suites_dir=Path(paths_cfg.get("suites_dir", "example_suites")),
            reports_dir=Path(paths_cfg.get("reports_dir", "reports")),
            log_level=logging_cfg.get("level", "INFO"),
            pricing=data.get("pricing", {}),
        )

    def require_openai_key(self) -> str:
        """Return the OpenAI key or raise.

        Returns:
            The API key string.

        Raises:
            ValueError: If the key is not set.
        """
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required but not set. "
                "Add it to your .env file or environment."
            )
        return self.openai_api_key

    def require_anthropic_key(self) -> str:
        """Return the Anthropic key or raise.

        Returns:
            The API key string.

        Raises:
            ValueError: If the key is not set.
        """
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required but not set. "
                "Add it to your .env file or environment."
            )
        return self.anthropic_api_key
