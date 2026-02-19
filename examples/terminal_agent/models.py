from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rich.text import Text

ToolStatus = Literal["running", "success", "error"]
HistoryEntryType = Literal["user", "assistant", "tool_call", "tool_result", "system", "thinking", "elapsed"]


class LoadingStateType(Enum):
    """Loading 状态类型，用于决定渲染策略。"""

    TOOL_CALL = "tool_call"      # 工具调用 - 静态显示
    THINKING = "thinking"        # 思考中 - 流光效果
    ANIMATION = "animation"      # 动画状态（如 vibeing）- 流光效果
    IDLE = "idle"                # 空闲状态 - 无显示


@dataclass
class LoadingState:
    """语义化的 loading 状态，携带类型信息和显示数据。

    Attributes:
        type: Loading 状态类型，决定渲染策略
        text: 显示文本内容
        metadata: 额外元数据（如工具名、耗时、token 数等）
    """

    type: LoadingStateType
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tool_call(cls, text: str, **metadata) -> LoadingState:
        """创建工具调用状态的工厂方法。"""
        return cls(type=LoadingStateType.TOOL_CALL, text=text, metadata=metadata)

    @classmethod
    def thinking(cls, text: str, **metadata) -> LoadingState:
        """创建思考状态的工厂方法。"""
        return cls(type=LoadingStateType.THINKING, text=text, metadata=metadata)

    @classmethod
    def animation(cls, text: str, **metadata) -> LoadingState:
        """创建动画状态的工厂方法。"""
        return cls(type=LoadingStateType.ANIMATION, text=text, metadata=metadata)

    @classmethod
    def idle(cls) -> LoadingState:
        """创建空闲状态的工厂方法。"""
        return cls(type=LoadingStateType.IDLE, text="")


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
