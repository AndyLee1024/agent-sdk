from __future__ import annotations

from typing import TYPE_CHECKING

from comate_agent_sdk.agent.events import StepStartEvent, StopEvent, TextEvent, ToolResultEvent

from . import query_stream as query_stream_module
from .compaction import precheck_and_compact

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


async def run_query(agent: "AgentRuntime", message: str) -> str:
    """Sync query now consumes stream events as the single execution source of truth."""

    terminal_text: str | None = None
    ask_user_result: str | None = None

    original_precheck = query_stream_module.precheck_and_compact
    query_stream_module.precheck_and_compact = precheck_and_compact

    try:
        async for event in query_stream_module.run_query_stream(agent, message):
            if isinstance(event, TextEvent):
                terminal_text = event.content
                continue

            if isinstance(event, StepStartEvent):
                # Text emitted before tool calls belongs to that iteration, not final sync output.
                terminal_text = None
                continue

            if isinstance(event, ToolResultEvent):
                if event.tool == "AskUserQuestion" and not event.is_error:
                    ask_user_result = event.result
                continue

            if isinstance(event, StopEvent):
                if event.reason == "waiting_for_input":
                    return ask_user_result or ""
                if event.reason in {"completed", "max_iterations"}:
                    return terminal_text or ""
                if event.reason == "interrupted":
                    return ""

        return terminal_text or ""
    finally:
        query_stream_module.precheck_and_compact = original_precheck
