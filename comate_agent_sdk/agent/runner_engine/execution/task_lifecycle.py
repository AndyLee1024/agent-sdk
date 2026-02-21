from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from comate_agent_sdk.agent.events import (
    AgentEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    SubagentStopEvent,
)
from comate_agent_sdk.llm.messages import ToolMessage

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


TaskStopStatus = Literal["completed", "error", "timeout", "cancelled"]


def classify_task_result(tool_result: ToolMessage) -> tuple[TaskStopStatus, str | None]:
    status: TaskStopStatus = "completed"
    error_msg: str | None = None
    if tool_result.is_error:
        error_msg = tool_result.text
        lowered = tool_result.text.lower()
        if "timeout" in lowered:
            status = "timeout"
        elif "cancelled" in lowered:
            status = "cancelled"
        else:
            status = "error"
    return status, error_msg


class TaskLifecycleEmitter:
    def __init__(self, agent: "AgentRuntime") -> None:
        self._agent = agent

    async def emit_start(
        self,
        *,
        tool_call_id: str,
        subagent_name: str,
        description: str,
    ) -> AsyncIterator[AgentEvent]:
        await self._agent.run_hook_event(
            "SubagentStart",
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            subagent_description=description,
            subagent_status="running",
        )
        yield SubagentStartEvent(
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            description=description,
        )
        yield SubagentProgressEvent(
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            description=description,
            status="running",
            elapsed_ms=0.0,
            tokens=0,
        )

    async def emit_stop(
        self,
        *,
        tool_call_id: str,
        subagent_name: str,
        description: str,
        duration_ms: float,
        tokens: int,
        tool_result: ToolMessage,
    ) -> AsyncIterator[AgentEvent]:
        status, error_msg = classify_task_result(tool_result)
        await self._agent.run_hook_event(
            "SubagentStop",
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            subagent_description=description,
            subagent_status=status,
        )
        yield SubagentProgressEvent(
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            description=description,
            status=status,
            elapsed_ms=duration_ms,
            tokens=tokens,
        )
        yield SubagentStopEvent(
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            status=status,
            duration_ms=duration_ms,
            error=error_msg,
        )

    async def emit_cancelled(
        self,
        *,
        tool_call_id: str,
        subagent_name: str,
        description: str,
        elapsed_ms: float,
        tokens: int,
        error: str = "Interrupted by user",
    ) -> AsyncIterator[AgentEvent]:
        await self._agent.run_hook_event(
            "SubagentStop",
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            subagent_description=description,
            subagent_status="cancelled",
        )
        yield SubagentProgressEvent(
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            description=description,
            status="cancelled",
            elapsed_ms=elapsed_ms,
            tokens=tokens,
        )
        yield SubagentStopEvent(
            tool_call_id=tool_call_id,
            subagent_name=subagent_name,
            status="cancelled",
            duration_ms=elapsed_ms,
            error=error,
        )
