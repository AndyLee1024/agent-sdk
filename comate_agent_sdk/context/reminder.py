"""系统提醒模块

动态注入 <system-reminder> 标签，不污染核心上下文。
Lowering 时包装为 UserMessage(content="<system-reminder>...</system-reminder>", is_meta=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ReminderPosition(Enum):
    """Reminder 注入位置"""

    END = "end"                       # 末尾
    LAST_USER = "last_user"           # 最后一个 UserMessage 之后
    LAST_TOOL_RESULT = "last_tool_result"  # 最后一个 ToolMessage 之后
    AFTER_SYSTEM = "after_system"     # SystemMessage 之后


@dataclass
class SystemReminder:
    """系统提醒

    在 lowering 时注入到指定位置，不修改核心 IR。

    Attributes:
        name: 提醒名称（唯一标识，用于移除）
        content: 提醒内容（会被包裹在 <system-reminder> 标签中）
        position: 注入位置
        one_shot: 是否一次性（注入后自动移除）
        condition: 可选条件函数（返回 True 时才注入）
    """

    name: str
    content: str
    position: ReminderPosition = ReminderPosition.END
    one_shot: bool = False
    condition: Callable[[], bool] | None = None

    def should_inject(self) -> bool:
        """检查是否应该注入"""
        if self.condition is not None:
            return self.condition()
        return True

    def wrap_content(self) -> str:
        """将内容包裹在 system-reminder 标签中"""
        return f"<system-reminder>\n{self.content}\n</system-reminder>"
