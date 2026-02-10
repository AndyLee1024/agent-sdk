from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from comate_agent_sdk.agent.compaction import CompactionConfig, CompactionService
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.options import AgentConfig, RuntimeAgentOptions, build_runtime_options
from comate_agent_sdk.agent.system_prompt import SystemPromptType, resolve_system_prompt
from comate_agent_sdk.context import ContextIR
from comate_agent_sdk.context.fs import ContextFileSystem
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from comate_agent_sdk.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ToolCall,
    ToolMessage,
)
from comate_agent_sdk.observability import observe
from comate_agent_sdk.tokens import TokenCost, UsageSummary
from comate_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.events import AgentEvent
    from comate_agent_sdk.context.env import EnvOptions
    from comate_agent_sdk.llm.views import ChatInvokeCompletion


@dataclass(frozen=True)
class AgentTemplate:
    """不可变 Agent 模板。"""

    llm: BaseChatModel | None = None
    level: LLMLevel | None = None
    config: AgentConfig = field(default_factory=AgentConfig)
    name: str | None = None

    # Template 解析后缓存（不可变）
    _resolved_agents: tuple | None = field(default=None, repr=False, init=False)
    _resolved_skills: tuple | None = field(default=None, repr=False, init=False)
    _resolved_memory: object | None = field(default=None, repr=False, init=False)
    _resolved_settings: object | None = field(default=None, repr=False, init=False)
    _resolved_llm_levels: dict[LLMLevel, BaseChatModel] | None = field(
        default=None,
        repr=False,
        init=False,
    )
    _resolved_llm: BaseChatModel | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        from comate_agent_sdk.agent.init import build_template

        build_template(self)

    @property
    def tools(self) -> tuple[Tool | str, ...] | None:
        return self.config.tools

    @property
    def system_prompt(self) -> SystemPromptType:
        return self.config.system_prompt

    @property
    def role(self) -> str | None:
        return self.config.role

    @property
    def max_iterations(self) -> int:
        return self.config.max_iterations

    @property
    def tool_choice(self) -> ToolChoice:
        return self.config.tool_choice

    @property
    def resolved_llm(self) -> BaseChatModel:
        if self._resolved_llm is None:
            raise ValueError("Template has no resolved llm")
        return self._resolved_llm

    @property
    def resolved_llm_levels(self) -> dict[LLMLevel, BaseChatModel] | None:
        return self._resolved_llm_levels

    @property
    def resolved_agents(self) -> tuple | None:
        return self._resolved_agents

    @property
    def resolved_skills(self) -> tuple | None:
        return self._resolved_skills

    @property
    def resolved_memory(self) -> object | None:
        return self._resolved_memory

    def create_runtime(
        self,
        *,
        session_id: str | None = None,
        offload_root_path: str | None = None,
        parent_token_cost: TokenCost | None = None,
        is_subagent: bool = False,
        name: str | None = None,
        subagent_run_id: str | None = None,
    ) -> "AgentRuntime":
        runtime_options = build_runtime_options(
            config=self.config,
            resolved_agents=self._resolved_agents,
            resolved_skills=self._resolved_skills,
            resolved_memory=self._resolved_memory,
            resolved_llm_levels=self._resolved_llm_levels,
            session_id=session_id or self.config.session_id,
            offload_root_path=offload_root_path or self.config.offload_root_path,
        )
        return AgentRuntime(
            llm=self.resolved_llm,
            level=self.level,
            options=runtime_options,
            name=name if name is not None else self.name,
            template=self,
            _is_subagent=is_subagent,
            _parent_token_cost=parent_token_cost,
            _subagent_run_id=subagent_run_id,
        )

    @observe(name="agent_query")
    async def query(self, message: str) -> str:
        runtime = self.create_runtime()
        return await runtime.query(message)

    @observe(name="agent_query_stream")
    async def query_stream(
        self, message: str | list[ContentPartTextParam | ContentPartImageParam]
    ) -> AsyncIterator["AgentEvent"]:
        runtime = self.create_runtime()
        async for event in runtime.query_stream(message):
            yield event

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
        from comate_agent_sdk.agent.chat_session import ChatSession

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

        return ChatSession(
            self,
            storage_root=storage_root,
            message_source=message_source,
        )


@dataclass
class AgentRuntime:
    """运行态 Agent（会话/任务独占，可变）。"""

    llm: BaseChatModel | None = None
    level: LLMLevel | None = None
    options: RuntimeAgentOptions = field(default_factory=RuntimeAgentOptions)
    name: str | None = None
    template: AgentTemplate | None = field(default=None, repr=False)
    _is_subagent: bool = field(default=False, repr=False)
    _parent_token_cost: TokenCost | None = field(default=None, repr=False)
    _subagent_run_id: str | None = field(default=None, repr=False)
    _subagent_source_prefix: str | None = field(default=None, repr=False, init=False)

    # Internal state
    _context: ContextIR = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _tool_map: dict[str, Tool] = field(default_factory=dict, repr=False, init=False)
    _compaction_service: CompactionService | None = field(default=None, repr=False, init=False)
    _token_cost: TokenCost = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _context_fs: ContextFileSystem | None = field(default=None, repr=False, init=False)
    _session_id: str = field(default="", repr=False, init=False)

    # MCP internal state
    _mcp_manager: Any | None = field(default=None, repr=False, init=False)
    _mcp_loaded: bool = field(default=False, repr=False, init=False)
    _mcp_dirty: bool = field(default=False, repr=False, init=False)
    _mcp_pending_tool_names: list[str] = field(default_factory=list, repr=False, init=False)
    _mcp_load_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, init=False)
    _tools_allowlist_mode: bool = field(default=False, repr=False, init=False)
    _requested_tool_names: list[str] = field(default_factory=list, repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.options, RuntimeAgentOptions):
            raise TypeError(
                f"options 必须是 RuntimeAgentOptions，收到：{type(self.options).__name__}"
            )

        from comate_agent_sdk.agent.init import init_runtime_from_template

        init_runtime_from_template(self)

    @property
    def tools(self) -> list[Tool | str] | None:
        return self.options.tools

    @tools.setter
    def tools(self, value: list[Tool | str] | None) -> None:
        self.options.tools = value

    @property
    def system_prompt(self) -> SystemPromptType:
        return self.options.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: SystemPromptType) -> None:
        self.options.system_prompt = value

    @property
    def role(self) -> str | None:
        return self.options.role

    @role.setter
    def role(self, value: str | None) -> None:
        self.options.role = value

    @property
    def max_iterations(self) -> int:
        return self.options.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self.options.max_iterations = value

    @property
    def tool_choice(self) -> ToolChoice:
        return self.options.tool_choice

    @tool_choice.setter
    def tool_choice(self, value: ToolChoice) -> None:
        self.options.tool_choice = value

    @property
    def compaction(self) -> CompactionConfig | None:
        return self.options.compaction

    @compaction.setter
    def compaction(self, value: CompactionConfig | None) -> None:
        self.options.compaction = value

    @property
    def include_cost(self) -> bool:
        return self.options.include_cost

    @include_cost.setter
    def include_cost(self, value: bool) -> None:
        self.options.include_cost = value

    @property
    def emit_compaction_meta_events(self) -> bool:
        return self.options.emit_compaction_meta_events

    @emit_compaction_meta_events.setter
    def emit_compaction_meta_events(self, value: bool) -> None:
        self.options.emit_compaction_meta_events = value

    @property
    def dependency_overrides(self) -> dict | None:
        return self.options.dependency_overrides

    @dependency_overrides.setter
    def dependency_overrides(self, value: dict | None) -> None:
        self.options.dependency_overrides = value

    @property
    def ephemeral_storage_path(self) -> Path | None:
        return self.options.ephemeral_storage_path

    @ephemeral_storage_path.setter
    def ephemeral_storage_path(self, value: Path | None) -> None:
        self.options.ephemeral_storage_path = value

    @property
    def ephemeral_keep_recent(self) -> int | None:
        return self.options.ephemeral_keep_recent

    @ephemeral_keep_recent.setter
    def ephemeral_keep_recent(self, value: int | None) -> None:
        self.options.ephemeral_keep_recent = value

    @property
    def offload_enabled(self) -> bool:
        return self.options.offload_enabled

    @offload_enabled.setter
    def offload_enabled(self, value: bool) -> None:
        self.options.offload_enabled = value

    @property
    def offload_token_threshold(self) -> int:
        return self.options.offload_token_threshold

    @offload_token_threshold.setter
    def offload_token_threshold(self, value: int) -> None:
        self.options.offload_token_threshold = value

    @property
    def offload_root_path(self) -> str | None:
        return self.options.offload_root_path

    @offload_root_path.setter
    def offload_root_path(self, value: str | None) -> None:
        self.options.offload_root_path = value

    @property
    def offload_policy(self) -> OffloadPolicy | None:
        return self.options.offload_policy

    @offload_policy.setter
    def offload_policy(self, value: OffloadPolicy | None) -> None:
        self.options.offload_policy = value

    @property
    def llm_max_retries(self) -> int:
        return self.options.llm_max_retries

    @llm_max_retries.setter
    def llm_max_retries(self, value: int) -> None:
        self.options.llm_max_retries = value

    @property
    def llm_retry_base_delay(self) -> float:
        return self.options.llm_retry_base_delay

    @llm_retry_base_delay.setter
    def llm_retry_base_delay(self, value: float) -> None:
        self.options.llm_retry_base_delay = value

    @property
    def llm_retry_max_delay(self) -> float:
        return self.options.llm_retry_max_delay

    @llm_retry_max_delay.setter
    def llm_retry_max_delay(self, value: float) -> None:
        self.options.llm_retry_max_delay = value

    @property
    def llm_retryable_status_codes(self) -> set[int]:
        return self.options.llm_retryable_status_codes

    @llm_retryable_status_codes.setter
    def llm_retryable_status_codes(self, value: set[int]) -> None:
        self.options.llm_retryable_status_codes = value

    @property
    def precheck_buffer_ratio(self) -> float:
        return self.options.precheck_buffer_ratio

    @precheck_buffer_ratio.setter
    def precheck_buffer_ratio(self, value: float) -> None:
        self.options.precheck_buffer_ratio = value

    @property
    def token_count_timeout_ms(self) -> int:
        return self.options.token_count_timeout_ms

    @token_count_timeout_ms.setter
    def token_count_timeout_ms(self, value: int) -> None:
        self.options.token_count_timeout_ms = value

    @property
    def agents(self) -> list | None:
        return self.options.agents

    @agents.setter
    def agents(self, value: list | None) -> None:
        self.options.agents = value

    @property
    def tool_registry(self) -> object | None:
        return self.options.tool_registry

    @tool_registry.setter
    def tool_registry(self, value: object | None) -> None:
        self.options.tool_registry = value

    @property
    def project_root(self) -> Path | None:
        return self.options.project_root

    @project_root.setter
    def project_root(self, value: Path | None) -> None:
        self.options.project_root = value

    @property
    def task_parallel_enabled(self) -> bool:
        return self.options.task_parallel_enabled

    @task_parallel_enabled.setter
    def task_parallel_enabled(self, value: bool) -> None:
        self.options.task_parallel_enabled = value

    @property
    def task_parallel_max_concurrency(self) -> int:
        return self.options.task_parallel_max_concurrency

    @task_parallel_max_concurrency.setter
    def task_parallel_max_concurrency(self, value: int) -> None:
        self.options.task_parallel_max_concurrency = value

    @property
    def skills(self) -> list | None:
        return self.options.skills

    @skills.setter
    def skills(self, value: list | None) -> None:
        self.options.skills = value

    @property
    def memory(self) -> object | None:
        return self.options.memory

    @memory.setter
    def memory(self, value: object | None) -> None:
        self.options.memory = value

    @property
    def setting_sources(self) -> tuple[Literal["user", "project"], ...] | None:
        return self.options.setting_sources

    @setting_sources.setter
    def setting_sources(self, value: tuple[Literal["user", "project"], ...] | None) -> None:
        self.options.setting_sources = value

    @property
    def env_options(self) -> "EnvOptions | None":
        return self.options.env_options

    @env_options.setter
    def env_options(self, value: "EnvOptions | None") -> None:
        self.options.env_options = value

    @property
    def mcp_enabled(self) -> bool:
        return self.options.mcp_enabled

    @mcp_enabled.setter
    def mcp_enabled(self, value: bool) -> None:
        self.options.mcp_enabled = value

    @property
    def mcp_servers(self) -> Any:
        return self.options.mcp_servers

    @mcp_servers.setter
    def mcp_servers(self, value: Any) -> None:
        self.options.mcp_servers = value

    @property
    def llm_levels(self) -> dict[LLMLevel, BaseChatModel] | None:
        return self.options.llm_levels

    @llm_levels.setter
    def llm_levels(self, value: dict[LLMLevel, BaseChatModel] | None) -> None:
        self.options.llm_levels = value

    @property
    def session_id(self) -> str | None:
        return self.options.session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self.options.session_id = value

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self.tools]

    @property
    def messages(self) -> list[BaseMessage]:
        return self._context.lower()

    @property
    def token_cost(self) -> TokenCost:
        return self._token_cost

    @property
    def _effective_level(self) -> LLMLevel | None:
        if self.level is not None:
            return self.level

        if self.llm is None or self.llm_levels is None:
            return None

        for lv, llm_inst in self.llm_levels.items():
            if llm_inst is self.llm:
                return lv

        return None

    async def get_usage(
        self,
        *,
        model: str | None = None,
        source_prefix: str | None = None,
    ) -> UsageSummary:
        return await self._token_cost.get_usage_summary(
            model=model,
            source_prefix=source_prefix,
        )

    async def get_context_info(self):
        from comate_agent_sdk.context.info import ContextInfo, _build_categories

        budget = self._context.get_budget_status()

        model_name = self.llm.model
        context_limit = await self._compaction_service.get_model_context_limit(model_name)
        compact_threshold = await self._compaction_service.get_threshold_for_model(model_name)

        tool_defs_tokens = 0
        if self.tools:
            import json

            tool_defs_json = json.dumps(
                [t.definition.model_dump() for t in self.tools],
                ensure_ascii=False,
            )
            tool_defs_tokens = self._context.token_counter.count(tool_defs_json)

        categories = _build_categories(budget.tokens_by_type, self._context)

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

    def _resolve_system_prompt(self) -> str:
        return resolve_system_prompt(self.system_prompt, role=self.role)

    def clear_history(self):
        from comate_agent_sdk.agent.history import clear_history

        clear_history(self)

    def load_history(self, messages: list[BaseMessage]) -> None:
        from comate_agent_sdk.agent.history import load_history

        load_history(self, messages)

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
        from comate_agent_sdk.agent.runner import generate_max_iterations_summary

        return await generate_max_iterations_summary(self)

    async def _check_and_compact(self, response: "ChatInvokeCompletion") -> bool:
        from comate_agent_sdk.agent.runner import check_and_compact

        compacted, _, _ = await check_and_compact(self, response)
        return compacted

    async def query(self, message: str) -> str:
        from comate_agent_sdk.agent.runner import query

        return await query(self, message)

    async def query_stream(
        self, message: str | list[ContentPartTextParam | ContentPartImageParam]
    ) -> AsyncIterator["AgentEvent"]:
        from comate_agent_sdk.agent.runner_stream import query_stream

        async for event in query_stream(self, message):
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

    # ===== MCP =====

    def invalidate_mcp_tools(self, *, reason: str = "") -> None:
        self._mcp_dirty = True
        if reason:
            logger.debug(f"已标记 MCP tools dirty：{reason}")

    async def ensure_mcp_tools_loaded(self, *, force: bool = False) -> None:
        if not bool(self.mcp_enabled):
            if self._mcp_loaded:
                self._remove_mcp_tools_from_agent()
            self._mcp_loaded = True
            self._mcp_dirty = False
            return

        if self._mcp_loaded and not force and not self._mcp_dirty:
            return

        async with self._mcp_load_lock:
            if self._mcp_loaded and not force and not self._mcp_dirty:
                return

            from comate_agent_sdk.mcp.config import resolve_mcp_servers
            from comate_agent_sdk.mcp.manager import McpManager

            servers = resolve_mcp_servers(self.mcp_servers, project_root=self.project_root)
            if not servers:
                self._remove_mcp_tools_from_agent()
                self._mcp_loaded = True
                self._mcp_dirty = False
                return

            if self._mcp_manager is not None:
                try:
                    await self._mcp_manager.aclose()  # type: ignore[union-attr]
                except Exception as e:
                    logger.warning(f"关闭旧 MCP manager 失败（忽略）：{e}")
                finally:
                    self._mcp_manager = None

            manager = McpManager(servers)
            await manager.start()
            mcp_tools = manager.tools

            if self.tool_registry is not None:
                try:
                    if hasattr(self.tool_registry, "all") and hasattr(self.tool_registry, "unregister"):
                        for t in self.tool_registry.all():  # type: ignore[attr-defined]
                            if getattr(t, "_comate_agent_sdk_mcp_tool", False) is True:
                                self.tool_registry.unregister(t.name)  # type: ignore[attr-defined]

                    for t in mcp_tools:
                        self.tool_registry.register(t)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"更新 tool_registry 的 MCP tools 失败：{e}")

            self._apply_mcp_tools_to_agent(mcp_tools)

            if mcp_tools:
                overview = manager.build_overview_text()
                meta = manager.build_metadata()
                try:
                    self._context.set_mcp_tools(overview, metadata=meta)
                except Exception as e:
                    logger.warning(f"注入 MCP tools 到 ContextIR 失败：{e}")
            else:
                try:
                    self._context.remove_mcp_tools()
                except Exception:
                    pass
                if servers:
                    logger.warning(
                        f"配置了 {len(servers)} 个 MCP server，但未加载到任何工具。"
                        "请检查 server 连接状态。"
                    )

            if self._tools_allowlist_mode and self._mcp_pending_tool_names:
                missing: list[str] = []
                resolved: list[Tool] = []
                mcp_by_name = {t.name: t for t in mcp_tools}
                for name in self._mcp_pending_tool_names:
                    tool_obj = mcp_by_name.get(name)
                    if tool_obj is None:
                        missing.append(name)
                    else:
                        resolved.append(tool_obj)

                if missing:
                    raise ValueError(f"未找到 MCP tool(s): {missing}")

                existing_names = {t.name for t in self.tools if isinstance(t, Tool)}  # type: ignore[arg-type]
                new_tools = list(self.tools)
                for t in resolved:
                    if t.name not in existing_names:
                        new_tools.append(t)
                self.tools[:] = new_tools  # type: ignore[index]

                self._mcp_pending_tool_names = []

            self._tool_map = {t.name: t for t in self.tools if isinstance(t, Tool)}  # type: ignore[arg-type]

            tool_names_from_list = {t.name for t in self.tools if isinstance(t, Tool)}  # type: ignore[arg-type]
            tool_names_from_map = set(self._tool_map.keys())
            if tool_names_from_list != tool_names_from_map:
                logger.error(
                    f"工具一致性检查失败！tools={len(tool_names_from_list)}, "
                    f"_tool_map={len(tool_names_from_map)}, "
                    f"差异={tool_names_from_list ^ tool_names_from_map}"
                )

            mcp_tool_names = [
                t.name
                for t in self.tools  # type: ignore[arg-type]
                if isinstance(t, Tool) and getattr(t, "_comate_agent_sdk_mcp_tool", False)
            ]
            logger.info(f"MCP 工具加载完成：共 {len(mcp_tool_names)} 个工具")
            if mcp_tool_names:
                logger.debug(f"MCP 工具列表: {mcp_tool_names}")

            self._mcp_manager = manager
            self._mcp_loaded = True
            self._mcp_dirty = False

    def _remove_mcp_tools_from_agent(self) -> None:
        try:
            self.tools[:] = [
                t
                for t in self.tools
                if not (
                    isinstance(t, Tool)
                    and getattr(t, "_comate_agent_sdk_mcp_tool", False) is True
                )
            ]  # type: ignore[index]
            self._tool_map = {t.name: t for t in self.tools if isinstance(t, Tool)}  # type: ignore[arg-type]
        except Exception:
            pass

        try:
            if self.tool_registry is not None and hasattr(self.tool_registry, "all") and hasattr(self.tool_registry, "unregister"):
                for t in self.tool_registry.all():  # type: ignore[attr-defined]
                    if getattr(t, "_comate_agent_sdk_mcp_tool", False) is True:
                        self.tool_registry.unregister(t.name)  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            self._context.remove_mcp_tools()
        except Exception:
            pass

    def _apply_mcp_tools_to_agent(self, mcp_tools: list[Tool]) -> None:
        base_tools: list[Tool | str] = [
            t
            for t in self.tools
            if not (
                isinstance(t, Tool)
                and getattr(t, "_comate_agent_sdk_mcp_tool", False) is True
            )
        ]

        logger.debug(
            f"_apply_mcp_tools_to_agent: "
            f"base_tools={len(base_tools)}, "
            f"mcp_tools={len(mcp_tools)}, "
            f"allowlist_mode={self._tools_allowlist_mode}"
        )

        if self._tools_allowlist_mode:
            self.tools[:] = base_tools  # type: ignore[index]
            return

        merged: list[Tool | str] = list(base_tools)
        existing_names = {t.name for t in merged if isinstance(t, Tool)}
        for t in mcp_tools:
            if t.name not in existing_names:
                merged.append(t)

        logger.debug(f"合并后工具数量: {len(merged)}")
        self.tools[:] = merged  # type: ignore[index]
        logger.debug(f"赋值后 self.tools 数量: {len(self.tools)}")


# 公共别名
Agent = AgentTemplate
