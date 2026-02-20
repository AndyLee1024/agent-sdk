from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.options import AgentConfig, build_runtime_options
from comate_agent_sdk.agent.system_prompt import SystemPromptType
from comate_agent_sdk.llm.base import BaseChatModel, ToolChoice
from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam
from comate_agent_sdk.observability import observe
from comate_agent_sdk.tokens import TokenCost
from comate_agent_sdk.tools.decorator import Tool

if TYPE_CHECKING:
    from comate_agent_sdk.agent.events import AgentEvent


@dataclass(frozen=True)
class AgentTemplate:
    """不可变 Agent 模板。"""

    llm: BaseChatModel | None = None
    level: LLMLevel | None = None
    config: AgentConfig = field(default_factory=AgentConfig)
    name: str | None = None

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
        from comate_agent_sdk.agent.core.runtime import AgentRuntime

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

    def create_runtime_from_snapshot(
        self,
        *,
        header_snapshot: dict[str, Any],
        session_id: str | None = None,
        offload_root_path: str | None = None,
        parent_token_cost: TokenCost | None = None,
        is_subagent: bool = False,
        name: str | None = None,
        subagent_run_id: str | None = None,
    ) -> "AgentRuntime":
        """从持久化 header_snapshot 恢复 runtime（用于 session resume/fork）。"""
        from comate_agent_sdk.agent.core.runtime import AgentRuntime

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
            header_snapshot=header_snapshot,
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


if TYPE_CHECKING:
    from comate_agent_sdk.agent.core.runtime import AgentRuntime
