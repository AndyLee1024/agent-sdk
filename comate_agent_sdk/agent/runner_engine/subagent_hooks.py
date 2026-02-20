from __future__ import annotations

import json
from typing import TYPE_CHECKING

from comate_agent_sdk.llm.messages import ToolCall

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


def resolve_subagent_hook_payload(tool_call: ToolCall) -> tuple[str, str]:
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        args = {}

    subagent_name = args.get("subagent_type", "Task")
    description = args.get("description", subagent_name)
    if not isinstance(subagent_name, str) or not subagent_name.strip():
        subagent_name = "Task"
    if not isinstance(description, str) or not description.strip():
        description = subagent_name
    return subagent_name.strip(), description.strip()


async def fire_subagent_hook(
    agent: "AgentRuntime",
    *,
    event_name: str,
    tool_call: ToolCall,
    subagent_status: str | None = None,
) -> None:
    subagent_name, description = resolve_subagent_hook_payload(tool_call)
    await agent.run_hook_event(
        event_name,
        tool_call_id=tool_call.id,
        subagent_name=subagent_name,
        subagent_description=description,
        subagent_status=subagent_status,
    )
