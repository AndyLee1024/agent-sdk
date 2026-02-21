from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.events import AgentEvent, HiddenUserMessageEvent
from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


def is_interrupt_requested(agent: "AgentRuntime") -> bool:
    controller = getattr(agent, "_run_controller", None)
    return bool(controller and controller.is_interrupted)


async def drain_hidden_events(agent: "AgentRuntime") -> AsyncIterator[AgentEvent]:
    for content in agent.drain_hidden_user_messages():
        yield HiddenUserMessageEvent(content=content)


async def fire_session_start_if_needed(agent: "AgentRuntime") -> None:
    if agent._hooks_session_started:
        return
    await agent.run_hook_event("SessionStart")
    agent._hooks_session_started = True


async def fire_user_prompt_submit(
    agent: "AgentRuntime",
    message: str | list[ContentPartTextParam | ContentPartImageParam],
) -> None:
    prompt = message if isinstance(message, str) else "[multi-modal]"
    await agent.run_hook_event("UserPromptSubmit", prompt=prompt)
