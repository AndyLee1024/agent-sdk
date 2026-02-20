from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core.runtime import AgentRuntime
    from comate_agent_sdk.agent.hooks.models import AggregatedHookOutcome, HookInput


class RuntimeHooksMixin:
    def build_hook_input(self: "AgentRuntime", event_name: str, **kwargs: Any) -> "HookInput":
        from comate_agent_sdk.agent.hooks.models import HookInput

        cwd = str((self.options.project_root or Path.cwd()).resolve())
        return HookInput(
            session_id=self._session_id,
            cwd=cwd,
            permission_mode=str(self.options.permission_mode or "default"),
            hook_event_name=event_name,
            **kwargs,
        )

    async def run_hook_event(
        self: "AgentRuntime",
        event_name: str,
        **kwargs: Any,
    ) -> "AggregatedHookOutcome | None":
        if self._hook_engine is None:
            return None
        hook_input = self.build_hook_input(event_name, **kwargs)
        return await self._hook_engine.run_event(event_name, hook_input)

    def register_python_hook(
        self: "AgentRuntime",
        *,
        event_name: str,
        callback: Any,
        matcher: str = "*",
        order: int = 0,
        name: str | None = None,
    ) -> None:
        if self._hook_engine is None:
            raise ValueError("Hook engine is not initialized")
        self._hook_engine.register_python_hook(
            event_name=event_name,
            callback=callback,
            matcher=matcher,
            order=order,
            name=name,
        )

    def reset_stop_hook_state(self: "AgentRuntime") -> None:
        self._stop_hook_block_count = 0

    def mark_stop_blocked_once(self: "AgentRuntime") -> bool:
        self._stop_hook_block_count += 1
        return self._stop_hook_block_count <= self._stop_hook_block_limit
