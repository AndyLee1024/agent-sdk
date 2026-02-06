from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from comate_agent_sdk.mcp.types import SdkMcpTool

TFunc = TypeVar("TFunc", bound=Callable[..., Awaitable[Any]])


def mcp_tool(name: str, description: str) -> Callable[[TFunc], TFunc]:
    """为 SDK in-process MCP server 定义工具（基于 FastMCP 的类型注解 schema 生成）。

    约束：
    - 必须是 async function
    - tool name 由装饰器显式提供（不会使用 Python 函数名）
    - 输入 schema 由函数签名类型注解生成（推荐显式参数，而不是 args: dict）
    """

    def decorator(func: TFunc) -> TFunc:
        if not inspect.iscoroutinefunction(func):
            raise TypeError("mcp_tool 只能装饰 async function")

        # 标注元数据，供 create_sdk_mcp_server 注册到 FastMCP
        setattr(func, "_comate_agent_sdk_mcp_tool_name", str(name))
        setattr(func, "_comate_agent_sdk_mcp_tool_description", str(description))

        # 仅用于类型检查/IDE 提示（运行时无影响）
        _ = func  # noqa: F841
        return func

    return decorator


def is_sdk_mcp_tool(obj: Any) -> bool:
    return isinstance(obj, SdkMcpTool)

