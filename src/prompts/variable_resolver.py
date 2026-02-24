"""Jinja2-based variable resolution and prompt rendering.

Validates that all required variables are provided before rendering,
and builds OpenAI/Anthropic-compatible message lists.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, StrictUndefined, meta

from src.prompts.template_manager import PromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    """The fully-rendered prompt ready for API submission.

    Attributes:
        system_prompt: Rendered system message.
        user_prompt: Rendered user message.
        few_shot_messages: Rendered few-shot examples as role/content dicts.
    """

    system_prompt: str
    user_prompt: str
    few_shot_messages: list[dict[str, str]] = field(default_factory=list)


class VariableResolver:
    """Extract, validate, and render Jinja2 template variables.

    Uses ``StrictUndefined`` so any reference to an undefined variable
    raises immediately rather than silently inserting an empty string.
    """

    def __init__(self) -> None:
        self.env = Environment(undefined=StrictUndefined)

    def extract_variables(self, template_str: str) -> set[str]:
        """Parse a Jinja2 template string and return undeclared variable names.

        Args:
            template_str: A Jinja2 template string.

        Returns:
            Set of variable names referenced in the template.
        """
        ast = self.env.parse(template_str)
        return meta.find_undeclared_variables(ast)

    def validate(
        self, template: PromptTemplate, variables: dict[str, Any]
    ) -> list[str]:
        """Check that all required variables are present.

        Args:
            template: The prompt template to validate against.
            variables: Provided variable mapping.

        Returns:
            List of missing variable names (empty if all present).
        """
        required = self.extract_variables(template.system_prompt)
        required |= self.extract_variables(template.user_prompt)
        required |= set(template.variables)

        missing = sorted(required - set(variables.keys()))
        return missing

    def render(
        self, template: PromptTemplate, variables: dict[str, Any]
    ) -> RenderResult:
        """Render system prompt, user prompt, and few-shot examples.

        Args:
            template: The prompt template to render.
            variables: Variable mapping for substitution.

        Returns:
            A :class:`RenderResult` with all prompts rendered.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = self.validate(template, variables)
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        system = self.env.from_string(template.system_prompt).render(**variables)
        user = self.env.from_string(template.user_prompt).render(**variables)

        few_shot: list[dict[str, str]] = []
        for example in template.few_shot_examples:
            few_shot.append(
                {
                    "role": "user",
                    "content": self.env.from_string(example.user).render(**variables),
                }
            )
            few_shot.append(
                {
                    "role": "assistant",
                    "content": self.env.from_string(example.assistant).render(
                        **variables
                    ),
                }
            )

        return RenderResult(
            system_prompt=system,
            user_prompt=user,
            few_shot_messages=few_shot,
        )


def build_messages(render_result: RenderResult) -> list[dict[str, str]]:
    """Build an OpenAI/Anthropic-compatible message list from a render result.

    Args:
        render_result: Output from :meth:`VariableResolver.render`.

    Returns:
        List of ``{role, content}`` dicts ready for API submission.
    """
    messages = [{"role": "system", "content": render_result.system_prompt}]
    messages.extend(render_result.few_shot_messages)
    messages.append({"role": "user", "content": render_result.user_prompt})
    return messages
