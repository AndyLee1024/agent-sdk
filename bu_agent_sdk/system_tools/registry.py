from __future__ import annotations

import logging

from bu_agent_sdk.system_tools.tools import SYSTEM_TOOLS
from bu_agent_sdk.tools.decorator import Tool
from bu_agent_sdk.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def get_system_tools() -> list[Tool]:
    return list(SYSTEM_TOOLS)


def register_system_tools(registry: ToolRegistry) -> None:
    """注册 SDK 内置系统工具到指定 registry。"""
    for t in SYSTEM_TOOLS:
        try:
            registry.register(t)
        except Exception as e:
            logger.warning(f"注册系统工具失败：{t.name}：{e}")

