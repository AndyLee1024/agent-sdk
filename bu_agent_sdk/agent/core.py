from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from bu_agent_sdk.agent.compaction import CompactionConfig, CompactionService
from bu_agent_sdk.agent.llm_levels import LLMLevel
from bu_agent_sdk.agent.system_prompt import SystemPromptType, resolve_system_prompt
from bu_agent_sdk.context import ContextIR
from bu_agent_sdk.context.fs import ContextFileSystem
from bu_agent_sdk.context.offload import OffloadPolicy
from bu_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from bu_agent_sdk.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ToolCall,
    ToolMessage,
)
from bu_agent_sdk.observability import observe
from bu_agent_sdk.tokens import TokenCost, UsageSummary
from bu_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("bu_agent_sdk.agent")

if TYPE_CHECKING:
    from bu_agent_sdk.agent.events import AgentEvent
    from bu_agent_sdk.llm.views import ChatInvokeCompletion


@dataclass
class Agent:
    """
    Simple agentic loop that manages tool calling and message history.

    The agent will:
    1. Send the task to the LLM with available tools
    2. If the LLM returns tool calls, execute them and add results to history
    3. Repeat until the LLM returns a text response without tool calls
    4. Return the final response

    When compaction is enabled, the agent will automatically compress the
    conversation history when token usage exceeds the configured threshold.

    Attributes:
        llm: The language model to use for the agent.
        tools: List of Tool instances. If None, uses tools from the built-in registry.
        system_prompt: Optional system prompt to guide the agent.
        max_iterations: Maximum number of LLM calls before stopping.
        tool_choice: How the LLM should choose tools ('auto', 'required', 'none').
        compaction: Optional configuration for automatic context compaction.
        include_cost: Whether to calculate costs (requires fetching pricing data).
        dependency_overrides: Optional dict to override tool dependencies.
    """

    llm: BaseChatModel | None = None
    level: LLMLevel | None = None
    tools: list[Tool] | None = None
    system_prompt: SystemPromptType = None
    max_iterations: int = 200  # 200 steps max for now
    tool_choice: ToolChoice = "auto"
    compaction: CompactionConfig | None = None
    include_cost: bool = False
    dependency_overrides: dict | None = None
    ephemeral_storage_path: Path | None = None
    """Path to store destroyed ephemeral message content. If None, content is discarded."""
    ephemeral_keep_recent: int | None = None
    """Default keep_recent for ephemeral tools. Overrides tool's _ephemeral value."""

    # Context FileSystem 配置
    offload_enabled: bool = True
    """是否启用上下文卸载到文件系统"""
    offload_token_threshold: int = 2000
    """超过此 token 数的条目才会被卸载"""
    offload_root_path: str | None = None
    """卸载文件存储根目录。None 使用默认 ~/.agent/context/{session_id}"""
    offload_policy: OffloadPolicy | None = None
    """可选：细粒度卸载策略（按类型开关与阈值）。None 时使用默认策略。"""
    llm_max_retries: int = 5
    """Maximum retries for LLM errors at the agent level (matches browser-use default)."""
    llm_retry_base_delay: float = 1.0
    """Base delay in seconds for exponential backoff on LLM retries."""
    llm_retry_max_delay: float = 60.0
    """Maximum delay in seconds between LLM retry attempts."""
    llm_retryable_status_codes: set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    """HTTP status codes that trigger retries (matches browser-use)."""

    # Subagent support
    name: str | None = None
    agents: list | None = None  # type: ignore  # list[AgentDefinition]
    """List of AgentDefinition for creating subagents."""
    tool_registry: object | None = None  # type: ignore  # ToolRegistry
    """Tool registry for resolving tools by name (e.g. for subagents). If None, will be built from tools."""
    project_root: Path | None = None
    """Project root directory for discovering subagents. Defaults to cwd."""
    _is_subagent: bool = field(default=False, repr=False)
    """Internal flag to prevent nested subagents."""

    # Task (subagent) 并行执行配置
    task_parallel_enabled: bool = True
    """是否允许并行执行同一轮中的多个 Task tool call。"""
    task_parallel_max_concurrency: int = 4
    """并行执行 Task 的最大并发数。"""

    # Skill support
    skills: list | None = None  # type: ignore  # list[SkillDefinition]
    """List of SkillDefinition for Skill support. Auto-discovered if None."""

    # Memory support
    memory: object | None = None  # type: ignore  # MemoryConfig
    """Memory configuration for loading static background knowledge."""

    # Settings 配置
    setting_sources: tuple[Literal["user", "project"], ...] | None = ("user", "project")
    """控制加载哪些文件系统设置。默认加载 user 和 project。

    - "user": 加载 ~/.agent/settings.json 和 ~/.agent/AGENTS.md（当 project 无 AGENTS.md 时生效）
    - "project": 加载 .agent/settings.json 和 AGENTS.md
    - None 或 (): 不加载任何配置文件（向后兼容模式）
    """

    llm_levels: dict[LLMLevel, BaseChatModel] | None = None
    """三档 LLM（LOW/MID/HIGH）。用于工具内二次模型调用（如 WebFetch），默认可由 env 覆盖。"""

    session_id: str | None = None
    """Optional session id override (UUID string). Used to locate session storage."""

    # Internal state
    _context: ContextIR = field(default=None, repr=False)  # type: ignore  # 在 __post_init__ 中初始化
    _tool_map: dict[str, Tool] = field(default_factory=dict, repr=False)
    _compaction_service: CompactionService | None = field(default=None, repr=False)
    _token_cost: TokenCost = field(default=None, repr=False)  # type: ignore
    _parent_token_cost: TokenCost | None = field(default=None, repr=False)
    _context_fs: ContextFileSystem | None = field(default=None, repr=False)
    _session_id: str = field(default="", repr=False)

    def __post_init__(self):
        from bu_agent_sdk.agent.init import agent_post_init

        agent_post_init(self)

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for all registered tools."""
        return [t.definition for t in self.tools]

    @property
    def messages(self) -> list[BaseMessage]:
        """Get the current message history (read-only copy).

        Returns the lowered representation of the ContextIR,
        compatible with the old _messages format.
        """
        return self._context.lower()

    @property
    def token_cost(self) -> TokenCost:
        """Get the token cost service for direct access to usage tracking."""
        return self._token_cost

    @property
    def _effective_level(self) -> LLMLevel | None:
        """返回当前 Agent 实际使用的档位（用于 usage 打标）。"""
        if self.level is not None:
            return self.level

        if self.llm is None or self.llm_levels is None:
            return None

        for lv, llm_inst in self.llm_levels.items():
            if llm_inst is self.llm:
                return lv

        return None

    async def get_usage(self) -> UsageSummary:
        """Get usage summary for the agent."""
        return await self._token_cost.get_usage_summary()

    async def get_context_info(self):
        """获取当前上下文使用情况的详细信息。"""
        from bu_agent_sdk.context.info import ContextInfo, _build_categories

        # 获取 budget 状态
        budget = self._context.get_budget_status()

        # 获取模型信息
        model_name = self.llm.model
        context_limit = await self._compaction_service.get_model_context_limit(model_name)
        compact_threshold = await self._compaction_service.get_threshold_for_model(model_name)

        # 估算 Tool Definitions token 数
        tool_defs_tokens = 0
        if self.tools:
            import json

            tool_defs_json = json.dumps(
                [t.definition.model_dump() for t in self.tools],
                ensure_ascii=False,
            )
            tool_defs_tokens = self._context.token_counter.count(tool_defs_json)

        # 构建类别信息
        categories = _build_categories(budget.tokens_by_type, self._context)

        # 检查是否启用压缩
        compaction_enabled = self._compaction_service.config.enabled if self._compaction_service else True

        return ContextInfo(
            model_name=model_name,
            context_limit=context_limit,
            compact_threshold=compact_threshold,
            compact_threshold_ratio=budget.compact_threshold_ratio,
            total_tokens=budget.total_tokens,
            header_tokens=budget.header_tokens,
            conversation_tokens=budget.conversation_tokens,
            tool_definitions_tokens=tool_defs_tokens,
            categories=categories,
            compaction_enabled=compaction_enabled,
        )

    def chat(
        self,
        *,
        session_id: str | None = None,
        fork_session: str | None = None,
        storage_root: Path | None = None,
        message_source: (
            AsyncIterator[str | list[ContentPartTextParam | ContentPartImageParam]]
            | Iterable[str | list[ContentPartTextParam | ContentPartImageParam]]
            | None
        ) = None,
    ):
        from bu_agent_sdk.agent.chat_session import ChatSession

        if fork_session is not None and session_id is not None:
            raise ValueError("session_id and fork_session cannot be used together")

        if fork_session is not None:
            base = ChatSession.resume(self, session_id=fork_session)
            return base.fork_session(storage_root=storage_root, message_source=message_source)

        if session_id is not None:
            return ChatSession.resume(
                self,
                session_id=session_id,
                storage_root=storage_root,
                message_source=message_source,
            )
        return ChatSession(self, storage_root=storage_root, message_source=message_source)

    def _resolve_system_prompt(self) -> str:
        return resolve_system_prompt(self.system_prompt)

    def clear_history(self):
        from bu_agent_sdk.agent.history import clear_history

        clear_history(self)

    def load_history(self, messages: list[BaseMessage]) -> None:
        from bu_agent_sdk.agent.history import load_history

        load_history(self, messages)

    def _destroy_ephemeral_messages(self) -> None:
        from bu_agent_sdk.agent.history import destroy_ephemeral_messages

        destroy_ephemeral_messages(self)

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        from bu_agent_sdk.agent.tool_exec import execute_tool_call

        return await execute_tool_call(self, tool_call)

    def _extract_screenshot(self, tool_message: ToolMessage) -> str | None:
        from bu_agent_sdk.agent.tool_exec import extract_screenshot

        return extract_screenshot(tool_message)

    async def _invoke_llm(self) -> "ChatInvokeCompletion":
        from bu_agent_sdk.agent.llm import invoke_llm

        return await invoke_llm(self)

    async def _generate_max_iterations_summary(self) -> str:
        from bu_agent_sdk.agent.runner import generate_max_iterations_summary

        return await generate_max_iterations_summary(self)

    async def _check_and_compact(self, response: "ChatInvokeCompletion") -> bool:
        from bu_agent_sdk.agent.runner import check_and_compact

        return await check_and_compact(self, response)

    @observe(name="agent_query")
    async def query(self, message: str) -> str:
        from bu_agent_sdk.agent.runner import query

        return await query(self, message)

    @observe(name="agent_query_stream")
    async def query_stream(
        self, message: str | list[ContentPartTextParam | ContentPartImageParam]
    ) -> AsyncIterator["AgentEvent"]:
        from bu_agent_sdk.agent.runner_stream import query_stream

        async for event in query_stream(self, message):
            yield event

    def _setup_tool_strategy(self) -> None:
        from bu_agent_sdk.agent.setup import setup_tool_strategy

        setup_tool_strategy(self)

    def _setup_subagents(self) -> None:
        from bu_agent_sdk.agent.setup import setup_subagents

        setup_subagents(self)

    def _setup_memory(self) -> None:
        from bu_agent_sdk.agent.setup import setup_memory

        setup_memory(self)

    def _setup_skills(self) -> None:
        from bu_agent_sdk.agent.setup import setup_skills

        setup_skills(self)

    async def _execute_skill_call(self, tool_call: ToolCall) -> ToolMessage:
        from bu_agent_sdk.agent.setup import execute_skill_call

        return await execute_skill_call(self, tool_call)

    def _rebuild_skill_tool(self) -> None:
        from bu_agent_sdk.agent.setup import rebuild_skill_tool

        rebuild_skill_tool(self)
