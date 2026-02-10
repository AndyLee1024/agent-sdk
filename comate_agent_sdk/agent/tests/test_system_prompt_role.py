import logging

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.agent.system_prompt import SystemPromptConfig, resolve_system_prompt


GENERAL_OPENING_LINE = (
    "You are Comate, an interactive CLI tool that helps users as a general AI agent."
)
SOFTWARE_ENGINEERING_OPENING_LINE = (
    "You are Comate, an interactive CLI tool that helps users with software engineering tasks."
)


class _FakeChatModel:
    def __init__(self):
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("This test should not call the LLM")


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def test_default_role_uses_general_opening_line() -> None:
    prompt = resolve_system_prompt(None)
    assert _first_non_empty_line(prompt) == GENERAL_OPENING_LINE


def test_general_role_uses_general_opening_line() -> None:
    prompt = resolve_system_prompt(None, role="general")
    assert _first_non_empty_line(prompt) == GENERAL_OPENING_LINE


def test_unknown_role_falls_back_to_general_and_logs_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="comate_agent_sdk.agent.system_prompt"):
        prompt = resolve_system_prompt(None, role="data_analyst")

    assert _first_non_empty_line(prompt) == GENERAL_OPENING_LINE
    assert "fallback to 'general'" in caplog.text


def test_string_system_prompt_override_ignores_role() -> None:
    assert resolve_system_prompt("CUSTOM_PROMPT", role="general") == "CUSTOM_PROMPT"


def test_system_prompt_config_override_ignores_role() -> None:
    prompt = resolve_system_prompt(
        SystemPromptConfig(content="OVERRIDE_PROMPT", mode="override"),
        role="general",
    )
    assert prompt == "OVERRIDE_PROMPT"


def test_system_prompt_append_uses_role_aware_default_prompt() -> None:
    prompt = resolve_system_prompt(
        SystemPromptConfig(content="APPENDED_PROMPT", mode="append"),
        role="general",
    )
    assert _first_non_empty_line(prompt) == GENERAL_OPENING_LINE
    assert prompt.endswith("APPENDED_PROMPT")


def test_runtime_role_propagation_and_header_opening_line() -> None:
    template = Agent(
        llm=_FakeChatModel(),  # type: ignore[arg-type]
        config=AgentConfig(
            tools=(),
            agents=(),
            offload_enabled=False,
            setting_sources=None,
            role="general",
        ),
    )
    runtime = template.create_runtime()
    assert template.role == "general"
    assert runtime.role == "general"
    assert _first_non_empty_line(runtime.messages[0].text) == GENERAL_OPENING_LINE
