"""
Tools framework for building agentic applications.

This module provides:
- @tool decorator for creating type-safe tools from functions
- Depends for dependency injection
- Built-in default registry (only SDK-provided tools)

Example:
    from comate_agent_sdk.tools import tool, Depends

    # Define a simple tool (not globally registered)
    @tool("Add two numbers together")
    async def add(a: int, b: int) -> int:
        return a + b

    # Define a tool with dependency injection
    async def get_db():
        return DatabaseConnection()

    @tool("Query the database")
    async def query(sql: str, db: Depends(get_db)) -> str:
        return await db.execute(sql)

    # Get the default registry with all registered tools
    from comate_agent_sdk.tools import get_default_registry
    registry = get_default_registry()
"""

from comate_agent_sdk.tools.decorator import Tool, ToolContent, tool
from comate_agent_sdk.tools.depends import DependencyOverrides, Depends
from comate_agent_sdk.tools.registry import ToolRegistry

# ====== SDK 内置默认 Registry ======
_default_registry = ToolRegistry()

# 注册 SDK 内置系统工具（Claude Code 风格）
try:
    from comate_agent_sdk.system_tools import register_system_tools

    register_system_tools(_default_registry)
except Exception:
    # 避免 import 时阻塞 SDK 基本可用性
    # 详细错误通过 logger 记录在 system_tools 内部
    pass


def get_default_registry() -> ToolRegistry:
    """获取 SDK 内置默认工具注册表

    该 registry 仅用于 SDK 内置工具。@tool 默认不会做任何全局注册，
    调用方需要显式将 Tool 传给 Agent（tools=[...]）或注册到自定义 registry。

    Returns:
        全局默认的 ToolRegistry 实例

    Example:
        from comate_agent_sdk.tools import tool, get_default_registry

        @tool("搜索")
        async def search(query: str) -> str:
            return "..."

        # 仅包含 SDK 内置工具（可为空）
        registry = get_default_registry()

        agent = Agent(
            llm=llm,
            config=AgentConfig(
                tools=registry.all(),
            ),
        )
    """
    return _default_registry


def reset_default_registry() -> None:
    """重置全局默认 registry（主要用于测试隔离）

    清空全局 registry 中的所有工具，创建一个新的空 registry。
    这在单元测试中很有用，可以避免测试之间的工具污染。

    Example:
        def test_my_agent():
            reset_default_registry()  # 清理之前测试的工具

            @tool("测试工具")
            async def test_tool():
                pass

            agent = Agent(llm=llm)
    """
    global _default_registry
    _default_registry = ToolRegistry()
    try:
        from comate_agent_sdk.system_tools import register_system_tools

        register_system_tools(_default_registry)
    except Exception:
        pass


__all__ = [
    "tool",
    "Tool",
    "ToolContent",
    "Depends",
    "DependencyOverrides",
    "ToolRegistry",
    "get_default_registry",
    "reset_default_registry",
]
