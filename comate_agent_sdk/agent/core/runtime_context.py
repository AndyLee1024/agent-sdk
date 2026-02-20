from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.system_prompt import resolve_system_prompt
from comate_agent_sdk.llm.base import ToolDefinition
from comate_agent_sdk.llm.messages import BaseMessage
from comate_agent_sdk.tokens import UsageSummary

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core.runtime import AgentRuntime


class RuntimeContextMixin:
    @property
    def tool_definitions(self: "AgentRuntime") -> list[ToolDefinition]:
        from comate_agent_sdk.agent.tool_visibility import visible_tools

        visible = visible_tools(
            self.options.tools,
            is_subagent=bool(self._is_subagent),
        )
        return [t.definition for t in visible]

    @property
    def messages(self: "AgentRuntime") -> list[BaseMessage]:
        return self._context.lower()

    @property
    def token_cost(self: "AgentRuntime") -> "TokenCost":
        return self._token_cost

    @property
    def _effective_level(self: "AgentRuntime") -> LLMLevel | None:
        if self.level is not None:
            return self.level

        if self.llm is None or self.options.llm_levels is None:
            return None

        for lv, llm_inst in self.options.llm_levels.items():
            if llm_inst is self.llm:
                return lv

        return None

    async def get_usage(
        self: "AgentRuntime",
        *,
        model: str | None = None,
        source_prefix: str | None = None,
    ) -> UsageSummary:
        return await self._token_cost.get_usage_summary(
            model=model,
            source_prefix=source_prefix,
        )

    async def get_context_info(self: "AgentRuntime"):
        from comate_agent_sdk.context.info import ContextInfo, _build_categories

        budget = self._context.get_budget_status()

        model_name = self.llm.model
        context_limit = await self._compaction_service.get_model_context_limit(model_name)
        compact_threshold = await self._compaction_service.get_threshold_for_model(model_name)

        tool_definitions = self.tool_definitions if self.options.tools else []
        tool_defs_tokens = 0
        if tool_definitions:
            import json

            tool_defs_json = json.dumps(
                [d.model_dump() for d in tool_definitions],
                ensure_ascii=False,
            )
            tool_defs_tokens = self._context.token_counter.count(tool_defs_json)

        used_tokens_message_only = budget.total_tokens
        used_tokens_with_tools = used_tokens_message_only + tool_defs_tokens

        tracker = self._context_usage_tracker
        ir_total = self._context.total_tokens
        last_step_reported_tokens = tracker.context_usage if tracker is not None else 0
        next_step_estimated_tokens = (
            tracker.estimate_precheck(ir_total)
            if tracker is not None and tracker.context_usage > 0
            else used_tokens_with_tools
        )

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
            used_tokens_message_only=used_tokens_message_only,
            used_tokens_with_tools=used_tokens_with_tools,
            next_step_estimated_tokens=next_step_estimated_tokens,
            last_step_reported_tokens=last_step_reported_tokens,
            categories=categories,
            compaction_enabled=compaction_enabled,
        )

    def add_hidden_user_message(self: "AgentRuntime", content: str) -> None:
        text = content.strip()
        if not text:
            return
        self.add_hook_hidden_user_message(text, hook_name="runtime")

    def add_hook_hidden_user_message(
        self: "AgentRuntime",
        content: str,
        *,
        hook_name: str | None = None,
        related_tool_call_id: str | None = None,
    ) -> None:
        """Hook hidden user message 注入入口（自动遵循 tool barrier）。"""
        self._context.add_hook_hidden_user_message(
            content,
            hook_name=hook_name,
            related_tool_call_id=related_tool_call_id,
        )
        flushed = self._context.pop_flushed_hook_injection_texts()
        if flushed:
            self._pending_hidden_user_messages.extend(flushed)

    def drain_hidden_user_messages(self: "AgentRuntime") -> list[str]:
        flushed = self._context.pop_flushed_hook_injection_texts()
        if flushed:
            self._pending_hidden_user_messages.extend(flushed)
        if not self._pending_hidden_user_messages:
            return []
        pending = list(self._pending_hidden_user_messages)
        self._pending_hidden_user_messages.clear()
        return pending

    def _resolve_system_prompt(self: "AgentRuntime") -> str:
        return resolve_system_prompt(self.options.system_prompt, role=self.options.role)

    def clear_history(self: "AgentRuntime"):
        from comate_agent_sdk.agent.history import clear_history

        clear_history(self)

    def load_history(self: "AgentRuntime", messages: list[BaseMessage]) -> None:
        from comate_agent_sdk.agent.history import load_history

        load_history(self, messages)

    def set_plan_mode(self: "AgentRuntime", enabled: bool) -> None:
        """启用/禁用 plan mode。"""
        self._context.set_plan_mode(enabled)


if TYPE_CHECKING:
    from comate_agent_sdk.tokens import TokenCost
