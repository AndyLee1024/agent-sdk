"""Context 可观测性模块

所有 context 变异产生事件，可追溯审计。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from bu_agent_sdk.context.items import ItemType

logger = logging.getLogger("bu_agent_sdk.context.observer")


class EventType(Enum):
    """上下文事件类型"""

    ITEM_ADDED = "item_added"
    ITEM_REMOVED = "item_removed"
    ITEM_DESTROYED = "item_destroyed"
    COMPACTION_PERFORMED = "compaction_performed"
    REMINDER_REGISTERED = "reminder_registered"
    REMINDER_REMOVED = "reminder_removed"
    CONTEXT_CLEARED = "context_cleared"
    CONVERSATION_REPLACED = "conversation_replaced"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class ContextEvent:
    """上下文变更事件

    Attributes:
        event_type: 事件类型
        item_type: 相关条目类型（可选）
        item_id: 相关条目 ID（可选）
        detail: 事件描述
        timestamp: 事件时间戳
    """

    event_type: EventType
    item_type: ItemType | None = None
    item_id: str | None = None
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


# 最大事件日志保留数
MAX_EVENT_LOG_SIZE = 1000


@dataclass
class ContextEventBus:
    """上下文事件总线

    订阅者可监听所有上下文变更事件，用于日志、审计等。

    Attributes:
        _observers: 事件回调列表
        _event_log: 事件日志（最多保留 MAX_EVENT_LOG_SIZE 条）
    """

    _observers: list[Callable[[ContextEvent], None]] = field(default_factory=list)
    _event_log: deque[ContextEvent] = field(
        default_factory=lambda: deque(maxlen=MAX_EVENT_LOG_SIZE)
    )

    def subscribe(self, callback: Callable[[ContextEvent], None]) -> None:
        """订阅事件"""
        self._observers.append(callback)

    def unsubscribe(self, callback: Callable[[ContextEvent], None]) -> None:
        """取消订阅"""
        self._observers = [cb for cb in self._observers if cb is not callback]

    def emit(self, event: ContextEvent) -> None:
        """发送事件"""
        self._event_log.append(event)
        logger.debug(
            f"ContextEvent: {event.event_type.value} "
            f"item_type={event.item_type.value if event.item_type else 'N/A'} "
            f"item_id={event.item_id or 'N/A'} "
            f"detail={event.detail}"
        )
        for callback in self._observers:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Event observer error: {e}")

    @property
    def event_log(self) -> list[ContextEvent]:
        """获取事件日志（副本）"""
        return list(self._event_log)
