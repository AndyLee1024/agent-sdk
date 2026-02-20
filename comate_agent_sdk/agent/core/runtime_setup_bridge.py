from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam, ToolCall, ToolMessage

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core.runtime import AgentRuntime
    from comate_agent_sdk.agent.events import AgentEvent
    from comate_agent_sdk.llm.views import ChatInvokeCompletion


class RuntimeSetupBridgeMixin:
    def _destroy_ephemeral_messages(self: "AgentRuntime") -> None:
        from comate_agent_sdk.agent.history import destroy_ephemeral_messages

        destroy_ephemeral_messages(self)

    async def _execute_tool_call(self: "AgentRuntime", tool_call: ToolCall) -> ToolMessage:
        from comate_agent_sdk.agent.tool_exec import execute_tool_call

        return await execute_tool_call(self, tool_call)

    def _extract_screenshot(self: "AgentRuntime", tool_message: ToolMessage) -> str | None:
        from comate_agent_sdk.agent.tool_exec import extract_screenshot

        return extract_screenshot(tool_message)

    async def _invoke_llm(self: "AgentRuntime") -> "ChatInvokeCompletion":
        from comate_agent_sdk.agent.llm import invoke_llm

        return await invoke_llm(self)

    async def _generate_max_iterations_summary(self: "AgentRuntime") -> str:
        from comate_agent_sdk.agent.runner_engine import generate_max_iterations_summary

        return await generate_max_iterations_summary(self)

    async def _check_and_compact(self: "AgentRuntime", response: "ChatInvokeCompletion") -> bool:
        from comate_agent_sdk.agent.runner_engine import check_and_compact

        compacted, _, _ = await check_and_compact(self, response)
        return compacted

    async def query(self: "AgentRuntime", message: str) -> str:
        from comate_agent_sdk.agent.runner_engine import run_query

        return await run_query(self, message)

    async def query_stream(
        self: "AgentRuntime",
        message: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> AsyncIterator["AgentEvent"]:
        from comate_agent_sdk.agent.runner_engine import run_query_stream

        async for event in run_query_stream(self, message):
            yield event

    def _setup_tool_strategy(self: "AgentRuntime") -> None:
        from comate_agent_sdk.agent.setup import setup_tool_strategy

        setup_tool_strategy(self)

    def _setup_agent_loop(self: "AgentRuntime") -> None:
        from comate_agent_sdk.agent.setup import setup_agent_loop

        setup_agent_loop(self)

    def _setup_subagents(self: "AgentRuntime") -> None:
        from comate_agent_sdk.agent.setup import setup_subagents

        setup_subagents(self)

    def _setup_memory(self: "AgentRuntime") -> None:
        from comate_agent_sdk.agent.setup import setup_memory

        setup_memory(self)

    def _setup_skills(self: "AgentRuntime") -> None:
        from comate_agent_sdk.agent.setup import setup_skills

        setup_skills(self)

    async def _execute_skill_call(self: "AgentRuntime", tool_call: ToolCall) -> ToolMessage:
        from comate_agent_sdk.agent.setup import execute_skill_call

        return await execute_skill_call(self, tool_call)

    def _rebuild_skill_tool(self: "AgentRuntime") -> None:
        from comate_agent_sdk.agent.setup import rebuild_skill_tool

        rebuild_skill_tool(self)
