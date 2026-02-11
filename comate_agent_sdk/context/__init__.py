"""Context IR 模块

结构化管理 Agent 上下文，替代扁平的 _messages: list[BaseMessage]。

主要组件：
- ContextIR: 上下文中间表示
- ContextItem / Segment: 核心数据类型
- LoweringPipeline: IR → API messages 转换
- SelectiveCompactionPolicy: 选择性压缩
- SystemReminder: 动态提醒注入
- BudgetConfig / BudgetStatus: 预算控制
- ContextEventBus: 可观测性
- ContextFileSystem: 上下文卸载到文件系统
- OffloadPolicy: 卸载策略配置
"""

from comate_agent_sdk.context.budget import BudgetConfig, BudgetStatus, TokenCounter
from comate_agent_sdk.context.accounting import ContextTokenAccounting, NextStepEstimate
from comate_agent_sdk.context.compaction import (
    CompactionStrategy,
    SelectiveCompactionPolicy,
    TypeCompactionRule,
)
from comate_agent_sdk.context.env import EnvOptions
from comate_agent_sdk.context.fs import ContextFileSystem, OffloadedMeta
from comate_agent_sdk.context.info import ContextCategoryInfo, ContextInfo
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import (
    DEFAULT_PRIORITIES,
    ContextItem,
    ItemType,
    Segment,
    SegmentName,
)
from comate_agent_sdk.context.lower import LoweringPipeline
from comate_agent_sdk.context.memory import MemoryConfig
from comate_agent_sdk.context.observer import (
    ContextEvent,
    ContextEventBus,
    EventType,
)
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.context.reminder import ReminderPosition, SystemReminder

__all__ = [
    # 核心
    "ContextIR",
    "ContextItem",
    "ItemType",
    "Segment",
    "SegmentName",
    "DEFAULT_PRIORITIES",
    # Lowering
    "LoweringPipeline",
    # Compaction
    "CompactionStrategy",
    "SelectiveCompactionPolicy",
    "TypeCompactionRule",
    # Budget
    "BudgetConfig",
    "BudgetStatus",
    "TokenCounter",
    "ContextTokenAccounting",
    "NextStepEstimate",
    # Info
    "ContextInfo",
    "ContextCategoryInfo",
    # Env
    "EnvOptions",
    # Memory
    "MemoryConfig",
    # Reminder
    "SystemReminder",
    "ReminderPosition",
    # Observer
    "ContextEvent",
    "ContextEventBus",
    "EventType",
    # FileSystem
    "ContextFileSystem",
    "OffloadedMeta",
    "OffloadPolicy",
]
