"""YAML prompt template loading, caching, and automatic versioning.

Templates are stored as YAML files on disk.  Each time a template's
content changes, a new version row is written to the database so the
full history is preserved.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils.database import Database

logger = logging.getLogger(__name__)


@dataclass
class FewShotExample:
    """A single few-shot demonstration pair.

    Attributes:
        user: The user turn of the example.
        assistant: The assistant response of the example.
    """

    user: str
    assistant: str


@dataclass
class ModelConfig:
    """Per-prompt model sampling parameters.

    Attributes:
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        top_p: Nucleus-sampling parameter.
        stop_sequences: Optional stop strings.
    """

    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class PromptTemplate:
    """A single prompt definition loaded from YAML.

    Attributes:
        name: Unique prompt identifier (matches file stem).
        system_prompt: The system-message string (may contain Jinja2 vars).
        user_prompt: The user-message Jinja2 template.
        variables: Declared variable names expected by the template.
        model_config: Default sampling parameters.
        few_shot_examples: Optional in-context demonstrations.
        category: Logical grouping label.
        version: Auto-resolved version number.
        metadata: Arbitrary key-value metadata.
    """

    name: str
    system_prompt: str
    user_prompt: str
    variables: list[str]
    model_config: ModelConfig
    few_shot_examples: list[FewShotExample] = field(default_factory=list)
    category: str = "default"
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Compute a short SHA-256 hash of the mutable prompt content.

        Returns:
            A 16-character hex digest.
        """
        content = f"{self.system_prompt}|{self.user_prompt}|{self.few_shot_examples}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class TemplateManager:
    """Load, cache, and version-control YAML prompt templates.

    Args:
        db: A :class:`Database` instance for version tracking.
        prompts_dir: Directory containing ``*.yaml`` prompt files.
    """

    def __init__(self, db: Database, prompts_dir: Path) -> None:
        self.db = db
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, PromptTemplate] = {}

    def load(self, name: str) -> PromptTemplate:
        """Load a prompt template from YAML and resolve its version.

        Args:
            name: The prompt file stem (without ``.yaml`` extension).

        Returns:
            A fully-populated :class:`PromptTemplate`.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        path = self.prompts_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        with open(path) as fh:
            data = yaml.safe_load(fh)

        template = PromptTemplate(
            name=name,
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"],
            variables=data.get("variables", []),
            model_config=ModelConfig(**data.get("model_config", {})),
            few_shot_examples=[
                FewShotExample(**ex) for ex in data.get("few_shot_examples", [])
            ],
            category=data.get("category", "default"),
            metadata=data.get("metadata", {}),
        )

        template.version = self._resolve_version(template)
        self._cache[name] = template
        logger.info("Loaded prompt '%s' v%d", name, template.version)
        return template

    def _resolve_version(self, template: PromptTemplate) -> int:
        """Check the DB for existing versions; bump if content changed.

        Args:
            template: The template whose version to resolve.

        Returns:
            The resolved version number.
        """
        content_hash = template.content_hash()
        with self.db.connection() as conn:
            result = conn.execute(
                "SELECT version FROM prompt_versions "
                "WHERE prompt_name = ? ORDER BY version DESC LIMIT 1",
                [template.name],
            ).fetchone()

            if result is None:
                new_version = 1
            else:
                existing = conn.execute(
                    "SELECT hash FROM prompt_versions "
                    "WHERE prompt_name = ? AND version = ?",
                    [template.name, result[0]],
                ).fetchone()
                if existing and existing[0] == content_hash:
                    return result[0]
                new_version = result[0] + 1

            conn.execute(
                "INSERT INTO prompt_versions "
                "(prompt_name, version, content, created_at, hash) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    template.name,
                    new_version,
                    json.dumps(self._template_to_dict(template)),
                    datetime.now(),
                    content_hash,
                ],
            )
            return new_version

    def list_prompts(self) -> list[str]:
        """Return the stems of all YAML files in the prompts directory.

        Returns:
            Sorted list of prompt names.
        """
        return sorted(p.stem for p in self.prompts_dir.glob("*.yaml"))

    def list_by_category(self, category: str) -> list[str]:
        """List prompt names belonging to a specific category.

        Args:
            category: The category string to filter on.

        Returns:
            List of matching prompt names.
        """
        return [
            name for name in self.list_prompts() if self.load(name).category == category
        ]

    def export_collection(self, names: list[str], output_path: Path) -> None:
        """Export multiple prompts to a single YAML file.

        Args:
            names: Prompt names to export.
            output_path: Destination file path.
        """
        collection = {name: self._template_to_dict(self.load(name)) for name in names}
        with open(output_path, "w") as fh:
            yaml.dump(collection, fh, default_flow_style=False)
        logger.info("Exported %d prompts to %s", len(names), output_path)

    def import_collection(self, path: Path) -> list[str]:
        """Import prompts from a collection YAML file.

        Args:
            path: Path to the collection file.

        Returns:
            List of imported prompt names.
        """
        with open(path) as fh:
            collection = yaml.safe_load(fh)

        imported: list[str] = []
        for name, data in collection.items():
            output_path = self.prompts_dir / f"{name}.yaml"
            with open(output_path, "w") as fh:
                yaml.dump(data, fh, default_flow_style=False)
            imported.append(name)
        logger.info("Imported %d prompts from %s", len(imported), path)
        return imported

    @staticmethod
    def _template_to_dict(template: PromptTemplate) -> dict[str, Any]:
        """Serialise a template to a plain dict for YAML export.

        Args:
            template: The template to convert.

        Returns:
            A dict suitable for ``yaml.dump``.
        """
        return {
            "name": template.name,
            "system_prompt": template.system_prompt,
            "user_prompt": template.user_prompt,
            "variables": template.variables,
            "model_config": {
                "temperature": template.model_config.temperature,
                "max_tokens": template.model_config.max_tokens,
                "top_p": template.model_config.top_p,
                "stop_sequences": template.model_config.stop_sequences,
            },
            "few_shot_examples": [
                {"user": ex.user, "assistant": ex.assistant}
                for ex in template.few_shot_examples
            ],
            "category": template.category,
            "metadata": template.metadata,
        }
