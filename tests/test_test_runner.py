"""Tests for TestCase and TestSuite loading and filtering."""

from pathlib import Path

import pytest
import yaml

from src.testing.test_runner import TestSuite


@pytest.fixture()
def suite_yaml(tmp_path: Path) -> Path:
    """Write a sample suite YAML and return its path."""
    data = {
        "name": "demo_suite",
        "prompt": "demo",
        "description": "A demo suite",
        "tags": ["demo"],
        "test_cases": [
            {
                "id": "tc_1",
                "name": "Basic test",
                "input": {"x": 1},
                "expected": "one",
                "assertion": "exact",
                "tags": ["basic"],
            },
            {
                "id": "tc_2",
                "name": "Regex test",
                "input": {"x": 2},
                "expected": r"\d+",
                "assertion": "regex",
                "tags": ["regex", "advanced"],
            },
            {
                "name": "Auto ID test",
                "input": {"x": 3},
                "expected": "three",
                "tags": ["basic"],
            },
        ],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(data))
    return path


class TestTestSuiteFromYaml:
    """TestSuite.from_yaml should correctly parse YAML."""

    def test_loads_suite_name(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        assert suite.name == "demo_suite"

    def test_loads_prompt_name(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        assert suite.prompt_name == "demo"

    def test_loads_description(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        assert suite.description == "A demo suite"

    def test_loads_test_cases(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        assert len(suite.test_cases) == 3

    def test_auto_generates_id(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        assert suite.test_cases[2].id == "tc_2"  # 0-indexed: third item

    def test_assertion_defaults_to_contains(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        assert suite.test_cases[2].assertion_type == "contains"

    def test_timeout_default(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        assert suite.test_cases[0].timeout == 30.0


class TestFilterByTags:
    """Tag-based filtering on test suites."""

    def test_include_tags(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        filtered = suite.filter_by_tags(include=["basic"])
        assert len(filtered) == 2

    def test_exclude_tags(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        filtered = suite.filter_by_tags(exclude=["advanced"])
        assert len(filtered) == 2

    def test_include_and_exclude(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        filtered = suite.filter_by_tags(
            include=["basic", "regex"], exclude=["advanced"]
        )
        assert all("advanced" not in tc.tags for tc in filtered)

    def test_no_match_returns_empty(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        filtered = suite.filter_by_tags(include=["nonexistent"])
        assert filtered == []

    def test_no_filters_returns_all(self, suite_yaml: Path) -> None:
        suite = TestSuite.from_yaml(suite_yaml)
        filtered = suite.filter_by_tags()
        assert len(filtered) == 3
