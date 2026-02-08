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


@dataclass(frozen=True)
class TodoItemState:
    content: str
    status: str
    priority: str

