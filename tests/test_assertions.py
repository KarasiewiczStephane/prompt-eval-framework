"""Tests for assertion types and assertion factory."""

import numpy as np
import pytest

from src.testing.assertions import (
    Contains,
    CustomAssertion,
    ExactMatch,
    JsonSchemaValidation,
    RegexMatch,
    SemanticSimilarity,
    assertion_factory,
)


class TestExactMatch:
    """ExactMatch assertion."""

    def test_pass_identical(self) -> None:
        r = ExactMatch().evaluate("hello", "hello")
        assert r.passed is True

    def test_fail_different(self) -> None:
        r = ExactMatch().evaluate("hello", "world")
        assert r.passed is False

    def test_strips_whitespace(self) -> None:
        r = ExactMatch().evaluate("  hello  ", "hello")
        assert r.passed is True

    def test_case_insensitive(self) -> None:
        r = ExactMatch(case_sensitive=False).evaluate("Hello", "hello")
        assert r.passed is True

    def test_case_sensitive_fails(self) -> None:
        r = ExactMatch(case_sensitive=True).evaluate("Hello", "hello")
        assert r.passed is False


class TestContains:
    """Contains assertion."""

    def test_pass_substring(self) -> None:
        r = Contains().evaluate("hello world", "world")
        assert r.passed is True

    def test_fail_missing(self) -> None:
        r = Contains().evaluate("hello", "world")
        assert r.passed is False

    def test_case_insensitive(self) -> None:
        r = Contains(case_sensitive=False).evaluate("Hello World", "hello")
        assert r.passed is True


class TestRegexMatch:
    """RegexMatch assertion."""

    def test_pass_pattern(self) -> None:
        r = RegexMatch().evaluate("Error code: 404", r"\d{3}")
        assert r.passed is True
        assert r.details["match"] == "404"

    def test_fail_no_match(self) -> None:
        r = RegexMatch().evaluate("no numbers", r"\d+")
        assert r.passed is False

    def test_multiline(self) -> None:
        text = "line1\nline2\nError here"
        r = RegexMatch().evaluate(text, r"^Error")
        assert r.passed is True


class TestSemanticSimilarity:
    """SemanticSimilarity with mock embeddings."""

    @staticmethod
    def _mock_embedding(text: str) -> np.ndarray:
        """Deterministic embedding: hash-based vector."""
        rng = np.random.RandomState(hash(text) % 2**31)
        return rng.randn(128)

    def test_identical_text_high_similarity(self) -> None:
        r = SemanticSimilarity(
            threshold=0.99, embedding_fn=self._mock_embedding
        ).evaluate("hello", "hello")
        assert r.passed is True
        assert r.details["similarity"] == pytest.approx(1.0, abs=0.01)

    def test_different_text(self) -> None:
        r = SemanticSimilarity(
            threshold=0.99, embedding_fn=self._mock_embedding
        ).evaluate("hello", "completely different text")
        # Different texts with random embeddings won't be >= 0.99
        assert r.passed is False

    def test_missing_embedding_fn_raises(self) -> None:
        with pytest.raises(ValueError, match="Embedding function required"):
            SemanticSimilarity().evaluate("a", "b")


class TestJsonSchemaValidation:
    """JSON schema validation assertion."""

    def test_valid_json_no_schema(self) -> None:
        r = JsonSchemaValidation().evaluate('{"key": "value"}')
        assert r.passed is True

    def test_invalid_json(self) -> None:
        r = JsonSchemaValidation().evaluate("not json")
        assert r.passed is False
        assert "Invalid JSON" in r.message

    def test_valid_against_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        r = JsonSchemaValidation(schema=schema).evaluate('{"name": "Alice"}')
        assert r.passed is True

    def test_invalid_against_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        r = JsonSchemaValidation(schema=schema).evaluate('{"age": 30}')
        assert r.passed is False
        assert "Schema validation failed" in r.message


class TestCustomAssertion:
    """CustomAssertion with user function."""

    def test_pass(self) -> None:
        def fn(actual: str, expected: object) -> bool:
            return len(actual) > 0

        r = CustomAssertion(fn, name="non_empty").evaluate("hi", None)
        assert r.passed is True

    def test_fail(self) -> None:
        def fn(actual: str, expected: object) -> bool:
            return actual == expected

        r = CustomAssertion(fn).evaluate("a", "b")
        assert r.passed is False

    def test_exception_is_caught(self) -> None:
        def fn(actual: str, expected: object) -> bool:
            return 1 / 0  # type: ignore[return-value]

        r = CustomAssertion(fn).evaluate("a", "b")
        assert r.passed is False
        assert "error" in r.message


class TestAssertionFactory:
    """assertion_factory should create correct types from config."""

    def test_creates_exact(self) -> None:
        a = assertion_factory({"type": "exact"})
        assert isinstance(a, ExactMatch)

    def test_creates_contains(self) -> None:
        a = assertion_factory({"type": "contains"})
        assert isinstance(a, Contains)

    def test_creates_regex(self) -> None:
        a = assertion_factory({"type": "regex"})
        assert isinstance(a, RegexMatch)

    def test_creates_semantic_with_params(self) -> None:
        a = assertion_factory({"type": "semantic", "params": {"threshold": 0.9}})
        assert isinstance(a, SemanticSimilarity)
        assert a.threshold == 0.9

    def test_creates_json_schema(self) -> None:
        a = assertion_factory({"type": "json_schema", "params": {"schema": {}}})
        assert isinstance(a, JsonSchemaValidation)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown assertion type"):
            assertion_factory({"type": "nonexistent"})
