from __future__ import annotations

from collections.abc import Iterable

from comate_agent_sdk.tools.decorator import Tool

# Subagent 不允许暴露的系统工具：
# - Task：避免嵌套 subagent 调度
# - AskUserQuestion：避免 subagent 直接向用户发问，保持交互收敛在主 agent
SUBAGENT_HIDDEN_TOOL_NAMES = frozenset({"Task", "AskUserQuestion"})


def visible_tools(
    tools: Iterable[Tool | object],
    *,
    is_subagent: bool,
) -> list[Tool]:
    """按 runtime 身份过滤可见工具列表。"""
    normalized = [t for t in tools if isinstance(t, Tool)]
    if not is_subagent:
        return normalized
    return [t for t in normalized if t.name not in SUBAGENT_HIDDEN_TOOL_NAMES]


def hidden_tool_names(
    tools: Iterable[Tool | object],
    *,
    is_subagent: bool,
) -> list[str]:
    """返回当前上下文下应隐藏且实际命中的工具名（去重+排序）。"""
    if not is_subagent:
        return []
    names = {
        t.name
        for t in tools
        if isinstance(t, Tool) and t.name in SUBAGENT_HIDDEN_TOOL_NAMES
    }
    return sorted(names)
