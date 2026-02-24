"""Tests for Jinja2 variable resolution and prompt rendering."""

import pytest

from src.prompts.template_manager import FewShotExample, ModelConfig, PromptTemplate
from src.prompts.variable_resolver import RenderResult, VariableResolver, build_messages


@pytest.fixture()
def resolver() -> VariableResolver:
    """Fresh resolver instance."""
    return VariableResolver()


@pytest.fixture()
def sample_template() -> PromptTemplate:
    """A template with two variables and one few-shot example."""
    return PromptTemplate(
        name="test",
        system_prompt="You help {{ role }}.",
        user_prompt="Hello {{ name }}, tell me about {{ topic }}.",
        variables=["name", "topic"],
        model_config=ModelConfig(),
        few_shot_examples=[
            FewShotExample(user="Ask {{ name }}", assistant="Sure {{ name }}!")
        ],
    )


class TestExtractVariables:
    """extract_variables should discover Jinja2 placeholders."""

    def test_simple_template(self, resolver: VariableResolver) -> None:
        vars_ = resolver.extract_variables("Hello {{ name }}")
        assert vars_ == {"name"}

    def test_multiple_variables(self, resolver: VariableResolver) -> None:
        vars_ = resolver.extract_variables("{{ a }} and {{ b }} or {{ c }}")
        assert vars_ == {"a", "b", "c"}

    def test_no_variables(self, resolver: VariableResolver) -> None:
        vars_ = resolver.extract_variables("No variables here")
        assert vars_ == set()

    def test_conditionals(self, resolver: VariableResolver) -> None:
        tmpl = "{% if flag %}{{ value }}{% endif %}"
        vars_ = resolver.extract_variables(tmpl)
        assert "flag" in vars_
        assert "value" in vars_

    def test_loop_variables(self, resolver: VariableResolver) -> None:
        tmpl = "{% for item in items %}{{ item }}{% endfor %}"
        vars_ = resolver.extract_variables(tmpl)
        assert "items" in vars_


class TestValidate:
    """validate should catch missing variables."""

    def test_all_provided(
        self, resolver: VariableResolver, sample_template: PromptTemplate
    ) -> None:
        missing = resolver.validate(
            sample_template, {"name": "Alice", "topic": "ML", "role": "students"}
        )
        assert missing == []

    def test_missing_variable(
        self, resolver: VariableResolver, sample_template: PromptTemplate
    ) -> None:
        missing = resolver.validate(sample_template, {"name": "Alice"})
        assert "topic" in missing

    def test_extra_variables_ok(
        self, resolver: VariableResolver, sample_template: PromptTemplate
    ) -> None:
        missing = resolver.validate(
            sample_template,
            {"name": "A", "topic": "B", "role": "C", "extra": "ignored"},
        )
        assert missing == []


class TestRender:
    """render should substitute all variables into prompts."""

    def test_renders_system_and_user(
        self, resolver: VariableResolver, sample_template: PromptTemplate
    ) -> None:
        result = resolver.render(
            sample_template, {"name": "Alice", "topic": "ML", "role": "students"}
        )
        assert "students" in result.system_prompt
        assert "Alice" in result.user_prompt
        assert "ML" in result.user_prompt

    def test_renders_few_shot(
        self, resolver: VariableResolver, sample_template: PromptTemplate
    ) -> None:
        result = resolver.render(
            sample_template, {"name": "Bob", "topic": "AI", "role": "devs"}
        )
        assert len(result.few_shot_messages) == 2
        assert "Bob" in result.few_shot_messages[0]["content"]
        assert "Bob" in result.few_shot_messages[1]["content"]

    def test_missing_variable_raises(
        self, resolver: VariableResolver, sample_template: PromptTemplate
    ) -> None:
        with pytest.raises(ValueError, match="Missing required variables"):
            resolver.render(sample_template, {"name": "Alice"})

    def test_nested_dict_variable(self, resolver: VariableResolver) -> None:
        tpl = PromptTemplate(
            name="nested",
            system_prompt="Sys",
            user_prompt="{{ user.name }} from {{ user.city }}",
            variables=[],
            model_config=ModelConfig(),
        )
        result = resolver.render(tpl, {"user": {"name": "Alice", "city": "NYC"}})
        assert "Alice" in result.user_prompt
        assert "NYC" in result.user_prompt

    def test_list_variable(self, resolver: VariableResolver) -> None:
        tpl = PromptTemplate(
            name="list_test",
            system_prompt="Sys",
            user_prompt="{% for i in items %}{{ i }} {% endfor %}",
            variables=["items"],
            model_config=ModelConfig(),
        )
        result = resolver.render(tpl, {"items": ["a", "b", "c"]})
        assert "a" in result.user_prompt
        assert "b" in result.user_prompt

    def test_jinja2_filter(self, resolver: VariableResolver) -> None:
        tpl = PromptTemplate(
            name="filter",
            system_prompt="Sys",
            user_prompt="{{ name | upper }}",
            variables=["name"],
            model_config=ModelConfig(),
        )
        result = resolver.render(tpl, {"name": "alice"})
        assert result.user_prompt == "ALICE"


class TestBuildMessages:
    """build_messages should produce a proper message list."""

    def test_basic_structure(self) -> None:
        rr = RenderResult(
            system_prompt="Be helpful",
            user_prompt="Hi",
            few_shot_messages=[],
        )
        msgs = build_messages(rr)
        assert msgs[0] == {"role": "system", "content": "Be helpful"}
        assert msgs[-1] == {"role": "user", "content": "Hi"}

    def test_with_few_shot(self) -> None:
        rr = RenderResult(
            system_prompt="Sys",
            user_prompt="Q",
            few_shot_messages=[
                {"role": "user", "content": "ex_q"},
                {"role": "assistant", "content": "ex_a"},
            ],
        )
        msgs = build_messages(rr)
        assert len(msgs) == 4
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["content"] == "Q"
