from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


class ToolTxnCommitter:
    """Guarantee tool transaction is committed at most once per assistant turn."""

    def __init__(self, agent: "AgentRuntime", *, assistant_message, tool_calls) -> None:
        self._agent = agent
        self._assistant_message = assistant_message
        self._tool_calls = list(tool_calls)
        self._committed = False

    @property
    def committed(self) -> bool:
        return self._committed

    async def commit_if_needed(self, results_by_id: dict) -> bool:
        if self._committed:
            return False
        self._agent._context.add_messages_atomic(
            [self._assistant_message] + [results_by_id[call.id] for call in self._tool_calls]
        )
        self._committed = True
        return True
