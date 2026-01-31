"""
System tool context for built-in Claude Code style tools.

设计目标：
- 为内置系统工具提供稳定的 project_root 与默认工作目录
- 通过 contextvars 在单次 tool call 执行期间注入上下文（并发安全）
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemToolContext:
    project_root: Path


_SYSTEM_TOOL_CONTEXT: ContextVar[SystemToolContext | None] = ContextVar(
    "bu_agent_sdk_system_tool_context",
    default=None,
)


def get_system_tool_context() -> SystemToolContext:
    """Dependency provider for system tools.

    Notes:
        - 由 Agent 在执行 tool call 时注入（见 Agent._execute_tool_call）
        - 当用户直接调用 Tool.execute（例如单元测试）时，也可以手动注入
    """
    ctx = _SYSTEM_TOOL_CONTEXT.get()
    if ctx is None:
        raise RuntimeError(
            "SystemToolContext 未注入。请通过 Agent 执行工具，或在测试中使用 bind_system_tool_context()。"
        )
    return ctx


@contextmanager
def bind_system_tool_context(project_root: Path) -> Iterator[None]:
    """临时注入 SystemToolContext（并发安全）。"""
    root = project_root.resolve()
    token = _SYSTEM_TOOL_CONTEXT.set(SystemToolContext(project_root=root))
    try:
        yield
    finally:
        _SYSTEM_TOOL_CONTEXT.reset(token)

