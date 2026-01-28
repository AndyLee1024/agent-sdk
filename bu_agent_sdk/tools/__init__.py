"""
Tools framework for building agentic applications.

This module provides:
- @tool decorator for creating type-safe tools from functions
- Depends for dependency injection
- Global default registry for automatic tool registration

Example:
    from bu_agent_sdk.tools import tool, Depends

    # Define a simple tool (auto-registers to global registry)
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
    from bu_agent_sdk.tools import get_default_registry
    registry = get_default_registry()
"""

from bu_agent_sdk.tools.decorator import Tool, ToolContent, tool
from bu_agent_sdk.tools.depends import DependencyOverrides, Depends
from bu_agent_sdk.tools.registry import ToolRegistry

# ====== 全局默认 Registry ======
_default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    """获取全局默认工具注册表

    所有使用 @tool 装饰器且未指定 registry 参数的工具
    都会自动注册到这个全局 registry。

    Returns:
        全局默认的 ToolRegistry 实例

    Example:
        from bu_agent_sdk.tools import tool, get_default_registry

        @tool("搜索")
        async def search(query: str) -> str:
            return "..."

        # 获取包含所有已注册工具的 registry
        registry = get_default_registry()

        agent = Agent(
            llm=llm,
            tools=registry.all(),
            tool_registry=registry,
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
