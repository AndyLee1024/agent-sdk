from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.events import AgentEvent, SubagentProgressEvent, UsageDeltaEvent

from ..lifecycle import drain_hidden_events

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime
    from comate_agent_sdk.agent.runner_engine.execution.state import RunningTaskState


class EventDrainHub:
    """Centralized queue draining for usage/subagent/hidden events."""

    def __init__(
        self,
        agent: "AgentRuntime",
        *,
        usage_queue: asyncio.Queue[UsageDeltaEvent],
        subagent_event_queue: asyncio.Queue[AgentEvent],
        running_tasks: dict[str, "RunningTaskState"],
        usage_tokens_for_progress: Callable[[UsageDeltaEvent], int],
        usage_belongs_to_task: Callable[[UsageDeltaEvent, "RunningTaskState"], bool],
    ) -> None:
        self._agent = agent
        self._usage_queue = usage_queue
        self._subagent_event_queue = subagent_event_queue
        self._running_tasks = running_tasks
        self._usage_tokens_for_progress = usage_tokens_for_progress
        self._usage_belongs_to_task = usage_belongs_to_task

    async def drain_usage_events(self) -> AsyncIterator[AgentEvent]:
        while True:
            try:
                event = self._usage_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            yield event
            delta_tokens = self._usage_tokens_for_progress(event)
            if delta_tokens <= 0:
                continue
            for state in self._running_tasks.values():
                if state.usage_tracking_mode == "stream":
                    continue
                if not self._usage_belongs_to_task(event, state):
                    continue
                state.tokens += delta_tokens
                elapsed_ms = (time.monotonic() - state.started_at_monotonic) * 1000
                yield SubagentProgressEvent(
                    tool_call_id=state.tool_call_id,
                    subagent_name=state.subagent_name,
                    description=state.description,
                    status="running",
                    elapsed_ms=elapsed_ms,
                    tokens=state.tokens,
                )
                break

    async def drain_subagent_events(self) -> AsyncIterator[AgentEvent]:
        while True:
            try:
                event = self._subagent_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            yield event

    async def drain_hidden_events(self) -> AsyncIterator[AgentEvent]:
        async for event in drain_hidden_events(self._agent):
            yield event
