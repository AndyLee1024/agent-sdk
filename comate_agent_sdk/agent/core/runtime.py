from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from comate_agent_sdk.agent.compaction import CompactionService
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.options import RuntimeAgentOptions
from comate_agent_sdk.context import ContextIR
from comate_agent_sdk.context.fs import ContextFileSystem
from comate_agent_sdk.context.usage_tracker import ContextUsageTracker
from comate_agent_sdk.llm.base import BaseChatModel
from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam, ToolCall, ToolMessage
from comate_agent_sdk.tokens import TokenCost
from comate_agent_sdk.tools.decorator import Tool

from .runtime_context import RuntimeContextMixin
from .runtime_hooks import RuntimeHooksMixin
from .runtime_mcp import RuntimeMcpMixin


@dataclass
class AgentRuntime(
    RuntimeContextMixin,
    RuntimeHooksMixin,
    RuntimeMcpMixin,
):
    """运行态 Agent（会话/任务独占，可变）。"""

    llm: BaseChatModel | None = None
    level: LLMLevel | None = None
    options: RuntimeAgentOptions = field(default_factory=RuntimeAgentOptions)
    name: str | None = None
    template: "AgentTemplate | None" = field(default=None, repr=False)
    header_snapshot: dict[str, Any] | None = field(default=None, repr=False)
    _is_subagent: bool = field(default=False, repr=False)
    _parent_token_cost: TokenCost | None = field(default=None, repr=False)
    _subagent_run_id: str | None = field(default=None, repr=False)
    _subagent_source_prefix: str | None = field(default=None, repr=False, init=False)

    _context: ContextIR = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _tool_map: dict[str, Tool] = field(default_factory=dict, repr=False, init=False)
    _compaction_service: CompactionService | None = field(default=None, repr=False, init=False)
    _token_cost: TokenCost = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _context_usage_tracker: ContextUsageTracker = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _context_fs: ContextFileSystem | None = field(default=None, repr=False, init=False)
    _session_id: str = field(default="", repr=False, init=False)
    _run_controller: "SessionRunController | None" = field(default=None, repr=False, init=False)
    _hook_engine: Any | None = field(default=None, repr=False, init=False)
    _pending_hidden_user_messages: list[str] = field(default_factory=list, repr=False, init=False)
    _hooks_session_started: bool = field(default=False, repr=False, init=False)
    _hooks_session_ended: bool = field(default=False, repr=False, init=False)
    _stop_hook_block_count: int = field(default=0, repr=False, init=False)
    _stop_hook_block_limit: int = field(default=3, repr=False, init=False)

    _mcp_manager: Any | None = field(default=None, repr=False, init=False)
    _mcp_loaded: bool = field(default=False, repr=False, init=False)
    _mcp_dirty: bool = field(default=False, repr=False, init=False)
    _mcp_pending_tool_names: list[str] = field(default_factory=list, repr=False, init=False)
    _mcp_load_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, init=False)
    _lock_header_from_snapshot: bool = field(default=False, repr=False, init=False)
    _tools_allowlist_mode: bool = field(default=False, repr=False, init=False)
    _requested_tool_names: list[str] = field(default_factory=list, repr=False, init=False)
    _mode: Literal["act", "plan"] = field(default="act", repr=False, init=False)
    _active_mode_snapshot: Literal["act", "plan"] | None = field(default=None, repr=False, init=False)
    _pending_plan_approval: dict[str, str] | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.options, RuntimeAgentOptions):
            raise TypeError(
                f"options 必须是 RuntimeAgentOptions，收到：{type(self.options).__name__}"
            )

        from comate_agent_sdk.agent.init import init_runtime_from_template

        init_runtime_from_template(self)

    def get_mode(self) -> Literal["act", "plan"]:
        return self._mode

    def set_mode(self, mode: Literal["act", "plan"] | str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"act", "plan"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self._mode = normalized  # type: ignore[assignment]
        self._context.set_plan_mode(self._mode == "plan")

    def cycle_mode(self) -> Literal["act", "plan"]:
        next_mode: Literal["act", "plan"] = "plan" if self._mode == "act" else "act"
        self.set_mode(next_mode)
        return next_mode

    def arm_plan_approval(
        self,
        *,
        plan_path: str,
        summary: str,
        execution_prompt: str,
    ) -> None:
        self._pending_plan_approval = {
            "plan_path": plan_path,
            "summary": summary,
            "execution_prompt": execution_prompt,
        }

    def has_pending_plan_approval(self) -> bool:
        return self._pending_plan_approval is not None

    def pending_plan_approval(self) -> dict[str, str] | None:
        if self._pending_plan_approval is None:
            return None
        return dict(self._pending_plan_approval)

    def approve_plan(self) -> str:
        pending = self._pending_plan_approval
        self._pending_plan_approval = None
        self.set_mode("act")
        if pending is None:
            return "请基于已批准的计划开始执行。"
        execution_prompt = str(pending.get("execution_prompt", "")).strip()
        if execution_prompt:
            return execution_prompt
        return "请基于已批准的计划开始执行。"

    def reject_plan(self) -> None:
        self._pending_plan_approval = None
        self.set_mode("plan")

    # ── 懒加载委托方法（解决循环导入）────────────────────────────

    def _destroy_ephemeral_messages(self) -> None:
        from comate_agent_sdk.agent.history import destroy_ephemeral_messages

        destroy_ephemeral_messages(self)

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        from comate_agent_sdk.agent.tool_exec import execute_tool_call

        return await execute_tool_call(self, tool_call)

    def _extract_screenshot(self, tool_message: ToolMessage) -> str | None:
        from comate_agent_sdk.agent.tool_exec import extract_screenshot

        return extract_screenshot(tool_message)

    async def _invoke_llm(self) -> "ChatInvokeCompletion":
        from comate_agent_sdk.agent.llm import invoke_llm

        return await invoke_llm(self)

    async def _generate_max_iterations_summary(self) -> str:
        from comate_agent_sdk.agent.runner_engine import generate_max_iterations_summary

        return await generate_max_iterations_summary(self)

    async def _check_and_compact(self, response: "ChatInvokeCompletion") -> bool:
        from comate_agent_sdk.agent.runner_engine import check_and_compact

        compacted, _, _ = await check_and_compact(self, response)
        return compacted

    async def query(self, message: str) -> str:
        from comate_agent_sdk.agent.runner_engine import run_query

        return await run_query(self, message)

    async def query_stream(
        self,
        message: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> AsyncIterator["AgentEvent"]:
        from comate_agent_sdk.agent.runner_engine import run_query_stream

        async for event in run_query_stream(self, message):
            yield event

    def _setup_tool_strategy(self) -> None:
        from comate_agent_sdk.agent.setup import setup_tool_strategy

        setup_tool_strategy(self)

    def _setup_agent_loop(self) -> None:
        from comate_agent_sdk.agent.setup import setup_agent_loop

        setup_agent_loop(self)

    def _setup_subagents(self) -> None:
        from comate_agent_sdk.agent.setup import setup_subagents

        setup_subagents(self)

    def _setup_memory(self) -> None:
        from comate_agent_sdk.agent.setup import setup_memory

        setup_memory(self)

    def _setup_skills(self) -> None:
        from comate_agent_sdk.agent.setup import setup_skills

        setup_skills(self)

    async def _execute_skill_call(self, tool_call: ToolCall) -> ToolMessage:
        from comate_agent_sdk.agent.setup import execute_skill_call

        return await execute_skill_call(self, tool_call)

    def _rebuild_skill_tool(self) -> None:
        from comate_agent_sdk.agent.setup import rebuild_skill_tool

        rebuild_skill_tool(self)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core.template import AgentTemplate
    from comate_agent_sdk.agent.events import AgentEvent
    from comate_agent_sdk.agent.interrupt import SessionRunController
    from comate_agent_sdk.llm.views import ChatInvokeCompletion
