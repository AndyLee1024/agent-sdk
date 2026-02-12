from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rich.text import Text

ToolStatus = Literal["running", "success", "error"]
HistoryEntryType = Literal["user", "assistant", "tool_call", "tool_result", "system"]


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


@dataclass
class HistoryEntry:
    entry_type: HistoryEntryType
    text: str | Text  # 支持普通字符串或 Rich Text 对象
    is_error: bool = False


@dataclass(frozen=True)
class TodoItemState:
    content: str
    status: str
    priority: str
