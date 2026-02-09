from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ToolStatus = Literal["running", "success", "error"]


@dataclass
class ToolRunState:
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    args_summary: str
    status: ToolStatus = "running"
    pulse_frame: int = 0
    result_preview: str = ""
    started_at_monotonic: float = 0.0
    is_task: bool = False
    subagent_name: str = ""
    task_desc: str = ""
    subagent_source_prefix: str = ""
    baseline_source_tokens: int = 0
    task_tokens: int = 0
    last_progress_render_ts: float = 0.0
    last_progress_tokens: int = 0


@dataclass(frozen=True)
class TodoItemState:
    content: str
    status: str
    priority: str
