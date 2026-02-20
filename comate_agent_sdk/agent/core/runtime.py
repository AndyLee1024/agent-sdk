from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from comate_agent_sdk.agent.compaction import CompactionService
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.options import RuntimeAgentOptions
from comate_agent_sdk.context import ContextIR
from comate_agent_sdk.context.fs import ContextFileSystem
from comate_agent_sdk.context.usage_tracker import ContextUsageTracker
from comate_agent_sdk.llm.base import BaseChatModel
from comate_agent_sdk.tokens import TokenCost
from comate_agent_sdk.tools.decorator import Tool

from .runtime_context import RuntimeContextMixin
from .runtime_hooks import RuntimeHooksMixin
from .runtime_mcp import RuntimeMcpMixin
from .runtime_setup_bridge import RuntimeSetupBridgeMixin


@dataclass
class AgentRuntime(
    RuntimeContextMixin,
    RuntimeHooksMixin,
    RuntimeSetupBridgeMixin,
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

    def __post_init__(self) -> None:
        if not isinstance(self.options, RuntimeAgentOptions):
            raise TypeError(
                f"options 必须是 RuntimeAgentOptions，收到：{type(self.options).__name__}"
            )

        from comate_agent_sdk.agent.init import init_runtime_from_template

        init_runtime_from_template(self)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core.template import AgentTemplate
    from comate_agent_sdk.agent.interrupt import SessionRunController
