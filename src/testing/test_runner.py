"""Test case and test suite definitions.

Defines the YAML-loadable data structures for organising evaluation
test cases, including tag-based filtering.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case.

    Attributes:
        id: Unique identifier within the suite.
        name: Human-readable name.
        input_variables: Variable mapping for prompt rendering.
        expected_output: Value to assert against.
        assertion_type: Which assertion to use (e.g. ``contains``).
        assertion_params: Extra parameters for the assertion.
        tags: Categorisation labels for filtering.
        timeout: Per-test timeout in seconds.
    """

    id: str
    name: str
    input_variables: dict[str, Any]
    expected_output: Any
    assertion_type: str = "contains"
    assertion_params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    timeout: float = 30.0


@dataclass
class TestSuite:
    """A collection of test cases tied to a prompt template.

    Attributes:
        name: Suite identifier.
        prompt_name: Name of the prompt template to evaluate.
        test_cases: Ordered list of test cases.
        tags: Suite-level tags.
        description: Optional human-readable description.
    """

    name: str
    prompt_name: str
    test_cases: list[TestCase]
    tags: list[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "TestSuite":
        """Load a test suite from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A populated :class:`TestSuite`.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        with open(path) as fh:
            data = yaml.safe_load(fh)

        test_cases = [
            TestCase(
                id=tc.get("id", f"tc_{i}"),
                name=tc["name"],
                input_variables=tc["input"],
                expected_output=tc["expected"],
                assertion_type=tc.get("assertion", "contains"),
                assertion_params=tc.get("assertion_params", {}),
                tags=tc.get("tags", []),
                timeout=tc.get("timeout", 30.0),
            )
            for i, tc in enumerate(data["test_cases"])
        ]

        return cls(
            name=data["name"],
            prompt_name=data["prompt"],
            test_cases=test_cases,
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )

    def filter_by_tags(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[TestCase]:
        """Filter test cases by tag inclusion/exclusion.

        Args:
            include: Keep only cases with at least one of these tags.
            exclude: Drop cases that have any of these tags.

        Returns:
            Filtered list of :class:`TestCase` instances.
        """
        filtered = self.test_cases
        if include:
            filtered = [tc for tc in filtered if any(t in tc.tags for t in include)]
        if exclude:
            filtered = [tc for tc in filtered if not any(t in tc.tags for t in exclude)]
        return filtered
