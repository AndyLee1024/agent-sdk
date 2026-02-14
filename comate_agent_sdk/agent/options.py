from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from comate_agent_sdk.agent.compaction import CompactionConfig
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.system_prompt import SystemPromptType
from comate_agent_sdk.context.env import EnvOptions
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.llm.base import BaseChatModel, ToolChoice
from comate_agent_sdk.tools.decorator import Tool


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_value(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """不可变 Agent 配置（Template 层）。"""

    # LLM/tool/system prompt
    tools: tuple[Tool | str, ...] | None = None
    system_prompt: SystemPromptType = None
    max_iterations: int = 200
    tool_choice: ToolChoice = "auto"
    compaction: CompactionConfig | None = None
    include_cost: bool = False
    emit_compaction_meta_events: bool = False
    dependency_overrides: Mapping[str, Any] | None = None

    # Ephemeral
    ephemeral_storage_path: Path | None = None
    ephemeral_keep_recent: int | None = None

    # Context FileSystem / offload
    offload_enabled: bool = True
    offload_token_threshold: int = 2000
    offload_root_path: str | None = None
    offload_policy: OffloadPolicy | None = None

    # LLM retries
    llm_max_retries: int = 5
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 60.0
    llm_retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )
    precheck_buffer_ratio: float = 0.12
    token_count_timeout_ms: int = 300

    # Subagent
    agents: tuple[Any, ...] | None = None  # tuple[AgentDefinition, ...]
    project_root: Path | None = None

    # Task (subagent) 并行执行配置
    task_parallel_enabled: bool = True
    task_parallel_max_concurrency: int = 4

    # Skills / memory
    skills: tuple[Any, ...] | None = None  # tuple[SkillDefinition, ...]
    memory: object | None = None  # MemoryConfig

    # Settings / env
    setting_sources: tuple[Literal["user", "project", "local"], ...] | None = (
        "user",
        "project",
        "local",
    )
    permission_mode: str = "default"
    tool_approval_callback: Any | None = None
    env_options: EnvOptions | None = None

    # MCP
    mcp_enabled: bool = True
    mcp_servers: Any = None

    # LLM levels + session
    llm_levels: Mapping[LLMLevel, BaseChatModel] | None = None
    session_id: str | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", _freeze_value(self.tools))
        object.__setattr__(self, "dependency_overrides", _freeze_value(self.dependency_overrides))
        object.__setattr__(self, "llm_retryable_status_codes", frozenset(self.llm_retryable_status_codes))
        object.__setattr__(self, "agents", _freeze_value(self.agents))
        object.__setattr__(self, "skills", _freeze_value(self.skills))
        object.__setattr__(self, "mcp_servers", _freeze_value(self.mcp_servers))
        object.__setattr__(self, "llm_levels", _freeze_value(self.llm_levels))


@dataclass
class RuntimeAgentOptions:
    """运行态可变配置（Runtime 层）。"""

    # LLM/tool/system prompt
    tools: list[Tool | str] | None = None
    system_prompt: SystemPromptType = None
    max_iterations: int = 200
    tool_choice: ToolChoice = "auto"
    compaction: CompactionConfig | None = None
    include_cost: bool = False
    emit_compaction_meta_events: bool = False
    dependency_overrides: dict | None = None

    # Ephemeral
    ephemeral_storage_path: Path | None = None
    ephemeral_keep_recent: int | None = None

    # Context FileSystem / offload
    offload_enabled: bool = True
    offload_token_threshold: int = 2000
    offload_root_path: str | None = None
    offload_policy: OffloadPolicy | None = None

    # LLM retries
    llm_max_retries: int = 5
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 60.0
    llm_retryable_status_codes: set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    precheck_buffer_ratio: float = 0.12
    token_count_timeout_ms: int = 300

    # Subagent
    agents: list | None = None  # list[AgentDefinition]
    tool_registry: object | None = None  # ToolRegistry
    project_root: Path | None = None

    # Task (subagent) 并行执行配置
    task_parallel_enabled: bool = True
    task_parallel_max_concurrency: int = 4

    # Skills / memory
    skills: list | None = None  # list[SkillDefinition]
    memory: object | None = None  # MemoryConfig

    # Settings / env
    setting_sources: tuple[Literal["user", "project", "local"], ...] | None = (
        "user",
        "project",
        "local",
    )
    permission_mode: str = "default"
    tool_approval_callback: Any | None = None
    env_options: EnvOptions | None = None

    # MCP
    mcp_enabled: bool = True
    mcp_servers: Any = None

    # LLM levels + session
    llm_levels: dict[LLMLevel, BaseChatModel] | None = None
    session_id: str | None = None
    role: str | None = None


def _thaw_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {k: v for k, v in value.items()}


def build_runtime_options(
    *,
    config: AgentConfig,
    resolved_agents: tuple[Any, ...] | None,
    resolved_skills: tuple[Any, ...] | None,
    resolved_memory: object | None,
    resolved_llm_levels: Mapping[LLMLevel, BaseChatModel] | None,
    session_id: str | None,
    offload_root_path: str | None,
) -> RuntimeAgentOptions:
    return RuntimeAgentOptions(
        tools=list(config.tools) if config.tools is not None else None,
        system_prompt=config.system_prompt,
        max_iterations=config.max_iterations,
        tool_choice=config.tool_choice,
        compaction=config.compaction,
        include_cost=config.include_cost,
        emit_compaction_meta_events=config.emit_compaction_meta_events,
        dependency_overrides=_thaw_mapping(config.dependency_overrides),
        ephemeral_storage_path=config.ephemeral_storage_path,
        ephemeral_keep_recent=config.ephemeral_keep_recent,
        offload_enabled=config.offload_enabled,
        offload_token_threshold=config.offload_token_threshold,
        offload_root_path=offload_root_path,
        offload_policy=config.offload_policy,
        llm_max_retries=config.llm_max_retries,
        llm_retry_base_delay=config.llm_retry_base_delay,
        llm_retry_max_delay=config.llm_retry_max_delay,
        llm_retryable_status_codes=set(config.llm_retryable_status_codes),
        precheck_buffer_ratio=config.precheck_buffer_ratio,
        token_count_timeout_ms=config.token_count_timeout_ms,
        agents=list(resolved_agents) if resolved_agents is not None else None,
        tool_registry=None,
        project_root=config.project_root,
        task_parallel_enabled=config.task_parallel_enabled,
        task_parallel_max_concurrency=config.task_parallel_max_concurrency,
        skills=list(resolved_skills) if resolved_skills is not None else None,
        memory=resolved_memory,
        setting_sources=config.setting_sources,
        permission_mode=config.permission_mode,
        tool_approval_callback=config.tool_approval_callback,
        env_options=config.env_options,
        mcp_enabled=config.mcp_enabled,
        mcp_servers=config.mcp_servers,
        llm_levels=dict(resolved_llm_levels) if resolved_llm_levels is not None else None,
        session_id=session_id,
        role=config.role,
    )
