"""Assertion types for evaluating LLM outputs.

Provides pluggable assertion classes (exact match, contains, regex,
semantic similarity, JSON schema, custom) and a factory function to
build them from config dicts.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class AssertionResult:
    """Outcome of a single assertion check.

    Attributes:
        passed: Whether the assertion succeeded.
        message: Human-readable summary.
        details: Optional extra data (e.g. similarity score).
    """

    passed: bool
    message: str
    details: dict[str, Any] | None = None


class Assertion(ABC):
    """Base class for all assertion types."""

    @abstractmethod
    def evaluate(self, actual: str, expected: Any) -> AssertionResult:
        """Evaluate the assertion.

        Args:
            actual: The model's actual output text.
            expected: The expected value to compare against.

        Returns:
            An :class:`AssertionResult`.
        """


class ExactMatch(Assertion):
    """Assert that actual output exactly matches the expected string.

    Args:
        case_sensitive: Whether comparison is case-sensitive.
    """

    def __init__(self, case_sensitive: bool = True) -> None:
        self.case_sensitive = case_sensitive

    def evaluate(self, actual: str, expected: str) -> AssertionResult:
        """Check for exact string match (after stripping whitespace).

        Args:
            actual: Model output.
            expected: Expected string.

        Returns:
            Pass/fail result.
        """
        a, e = actual.strip(), expected.strip()
        if not self.case_sensitive:
            a, e = a.lower(), e.lower()
        passed = a == e
        msg = "Exact match" if passed else f"Expected '{expected}', got '{actual}'"
        return AssertionResult(passed=passed, message=msg)


class Contains(Assertion):
    """Assert that the expected substring appears in the output.

    Args:
        case_sensitive: Whether the search is case-sensitive.
    """

    def __init__(self, case_sensitive: bool = True) -> None:
        self.case_sensitive = case_sensitive

    def evaluate(self, actual: str, expected: str) -> AssertionResult:
        """Check if expected is contained within actual.

        Args:
            actual: Model output.
            expected: Substring to look for.

        Returns:
            Pass/fail result.
        """
        a = actual if self.case_sensitive else actual.lower()
        e = expected if self.case_sensitive else expected.lower()
        passed = e in a
        msg = "Contains expected" if passed else f"'{expected}' not found in output"
        return AssertionResult(passed=passed, message=msg)


class RegexMatch(Assertion):
    """Assert that the output matches a regular expression pattern."""

    def evaluate(self, actual: str, expected: str) -> AssertionResult:
        """Search for a regex pattern in the output.

        Args:
            actual: Model output.
            expected: Regex pattern string.

        Returns:
            Pass/fail result with match details.
        """
        pattern = re.compile(expected, re.MULTILINE)
        match = pattern.search(actual)
        return AssertionResult(
            passed=match is not None,
            message="Regex matched" if match else f"Pattern '{expected}' not found",
            details={"match": match.group() if match else None},
        )


class SemanticSimilarity(Assertion):
    """Assert that output is semantically similar to the expected text.

    Uses cosine similarity on embeddings from a user-supplied function.

    Args:
        threshold: Minimum similarity score to pass.
        embedding_fn: Callable mapping text to a numpy vector.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.threshold = threshold
        self.embedding_fn = embedding_fn

    def evaluate(self, actual: str, expected: str) -> AssertionResult:
        """Compute cosine similarity between embeddings.

        Args:
            actual: Model output.
            expected: Reference text.

        Returns:
            Pass/fail result with similarity score.

        Raises:
            ValueError: If no embedding function is configured.
        """
        if self.embedding_fn is None:
            raise ValueError("Embedding function required for semantic similarity")

        actual_emb = self.embedding_fn(actual).reshape(1, -1)
        expected_emb = self.embedding_fn(expected).reshape(1, -1)
        similarity = float(cosine_similarity(actual_emb, expected_emb)[0][0])

        passed = similarity >= self.threshold
        return AssertionResult(
            passed=passed,
            message=f"Similarity: {similarity:.3f} (threshold: {self.threshold})",
            details={"similarity": similarity},
        )


class JsonSchemaValidation(Assertion):
    """Assert that output is valid JSON conforming to a given schema.

    Args:
        schema: A JSON Schema dict.
    """

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.schema = schema or {}

    def evaluate(self, actual: str, expected: Any = None) -> AssertionResult:
        """Parse JSON and validate against schema.

        Args:
            actual: Model output (should be JSON).
            expected: Ignored (schema comes from constructor).

        Returns:
            Pass/fail result.
        """
        try:
            data = json.loads(actual)
        except json.JSONDecodeError as exc:
            return AssertionResult(passed=False, message=f"Invalid JSON: {exc}")

        if not self.schema:
            return AssertionResult(passed=True, message="Valid JSON (no schema check)")

        try:
            import jsonschema

            jsonschema.validate(data, self.schema)
            return AssertionResult(passed=True, message="JSON schema valid")
        except ImportError:
            return AssertionResult(
                passed=False, message="jsonschema package not installed"
            )
        except jsonschema.ValidationError as exc:
            return AssertionResult(
                passed=False, message=f"Schema validation failed: {exc.message}"
            )


class CustomAssertion(Assertion):
    """Assert using a user-supplied callable.

    Args:
        fn: A function ``(actual, expected) -> bool``.
        name: Human-readable name for logging.
    """

    def __init__(self, fn: Callable[[str, Any], bool], name: str = "custom") -> None:
        self.fn = fn
        self.name = name

    def evaluate(self, actual: str, expected: Any) -> AssertionResult:
        """Run the custom function.

        Args:
            actual: Model output.
            expected: Expected value passed through.

        Returns:
            Pass/fail result.
        """
        try:
            passed = self.fn(actual, expected)
            return AssertionResult(
                passed=passed,
                message=f"{self.name}: {'passed' if passed else 'failed'}",
            )
        except Exception as exc:
            return AssertionResult(passed=False, message=f"{self.name} error: {exc}")


def assertion_factory(config: dict[str, Any]) -> Assertion:
    """Create an assertion instance from a config dict.

    Args:
        config: Must contain ``type`` key; optional ``params`` dict.

    Returns:
        An :class:`Assertion` subclass instance.

    Raises:
        ValueError: If the assertion type is unknown.
    """
    assertion_type = config["type"]
    params = config.get("params", {})

    mapping: dict[str, type[Assertion]] = {
        "exact": ExactMatch,
        "contains": Contains,
        "regex": RegexMatch,
        "semantic": SemanticSimilarity,
        "json_schema": JsonSchemaValidation,
    }

    if assertion_type not in mapping:
        raise ValueError(f"Unknown assertion type: {assertion_type}")

    return mapping[assertion_type](**params)
