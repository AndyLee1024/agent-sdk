from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from comate_agent_sdk.agent.events import AgentEvent, UsageDeltaEvent
from comate_agent_sdk.llm.messages import ToolMessage


@dataclass
class RunningTaskState:
    tool_call_id: str
    subagent_name: str
    description: str
    source_prefix: str
    started_at_monotonic: float
    tokens: int = 0
    usage_tracking_mode: Literal["queue", "stream"] = "queue"


@dataclass
class TaskStreamOut:
    """Result holder for streaming Task execution."""

    tool_message: ToolMessage | None = None


@dataclass
class TurnEngineState:
    """Mutable turn-scoped execution state shared across components."""

    usage_queue: asyncio.Queue[UsageDeltaEvent] = field(default_factory=asyncio.Queue)
    subagent_event_queue: asyncio.Queue[AgentEvent] = field(default_factory=asyncio.Queue)
    running_tasks: dict[str, RunningTaskState] = field(default_factory=dict)
    stop_reason: Literal["waiting_for_input", "interrupted"] | None = None
