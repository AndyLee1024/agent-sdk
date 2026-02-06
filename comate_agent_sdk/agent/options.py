from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from comate_agent_sdk.agent.compaction import CompactionConfig
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.system_prompt import SystemPromptType
from comate_agent_sdk.context.env import EnvOptions
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.llm.base import BaseChatModel, ToolChoice
from comate_agent_sdk.tools.decorator import Tool


@dataclass
class ComateAgentOptions:
    """Configuration dataclass for Comate queries."""

    # LLM/tool/system prompt
    tools: list[Tool | str] | None = None
    system_prompt: SystemPromptType = None
    max_iterations: int = 200
    tool_choice: ToolChoice = "auto"
    compaction: CompactionConfig | None = None
    include_cost: bool = False
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

    # Subagent
    agents: list | None = None  # type: ignore  # list[AgentDefinition]
    tool_registry: object | None = None  # type: ignore  # ToolRegistry
    project_root: Path | None = None

    # Task (subagent) 并行执行配置
    task_parallel_enabled: bool = True
    task_parallel_max_concurrency: int = 4

    # Skills / memory
    skills: list | None = None  # type: ignore  # list[SkillDefinition]
    memory: object | None = None  # type: ignore  # MemoryConfig

    # Settings / env
    setting_sources: tuple[Literal["user", "project"], ...] | None = ("user", "project")
    env_options: EnvOptions | None = None

    # MCP
    mcp_enabled: bool = True
    mcp_servers: Any = None

    # LLM levels + session
    llm_levels: dict[LLMLevel, BaseChatModel] | None = None
    session_id: str | None = None

