"""
全局工具注册表
"""

import logging

from bu_agent_sdk.tools.decorator import Tool


class ToolRegistry:
    """全局工具注册表

    用于管理和查找工具，支持按名称注册和检索工具。
    """

    def __init__(self):
        """初始化空的工具注册表"""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具

        Args:
            tool: 要注册的 Tool 实例

        Raises:
            ValueError: 如果工具名称已存在
        """
        if tool.name in self._tools:
            logging.warning(
                f"Tool '{tool.name}' already registered, overwriting with new definition"
            )
        self._tools[tool.name] = tool
        logging.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool:
        """根据名称获取工具

        Args:
            name: 工具名称

        Returns:
            对应的 Tool 实例

        Raises:
            KeyError: 如果工具不存在
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def filter(self, names: list[str]) -> list[Tool]:
        """根据名称列表筛选工具

        Args:
            names: 工具名称列表

        Returns:
            对应的 Tool 实例列表（忽略不存在的工具名）
        """
        tools = []
        for name in names:
            if name in self._tools:
                tools.append(self._tools[name])
            else:
                logging.warning(f"Tool '{name}' not found in registry, skipping")
        return tools

    def all(self) -> list[Tool]:
        """获取所有已注册的工具

        Returns:
            所有 Tool 实例的列表
        """
        return list(self._tools.values())

    def names(self) -> list[str]:
        """获取所有已注册的工具名称

        Returns:
            工具名称列表
        """
        return list(self._tools.keys())

    def clear(self) -> None:
        """清空注册表"""
        self._tools.clear()
        logging.debug("Tool registry cleared")

    def unregister(self, name: str) -> Tool | None:
        """移除一个工具（若不存在则返回 None）。"""
        tool = self._tools.pop(name, None)
        if tool is not None:
            logging.debug(f"Unregistered tool: {name}")
        return tool

    def __len__(self) -> int:
        """返回注册表中的工具数量"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools
