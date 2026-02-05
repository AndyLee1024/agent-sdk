from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from bu_agent_sdk.mcp.sdk_tool import is_sdk_mcp_tool
from bu_agent_sdk.mcp.types import McpSdkServerConfig, SdkMcpTool

logger = logging.getLogger("bu_agent_sdk.mcp.sdk_server")


def create_sdk_mcp_server(
    *,
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool] | None = None,
) -> McpSdkServerConfig:
    """创建一个 in-process MCP server（基于 FastMCP）。

    Args:
        name: server 唯一标识（用于展示，不参与 tool name 映射；映射使用 Agent.mcp_servers 的 alias）
        version: 版本字符串（用于 metadata；FastMCP 不强制使用）
        tools: 通过 @mcp_tool 装饰的工具函数列表
    """
    server = FastMCP(name=name)

    # 记录版本信息（仅用于调试/元数据）
    setattr(server, "_bu_agent_sdk_version", str(version))

    if tools:
        for fn in tools:
            if not is_sdk_mcp_tool(fn):
                raise TypeError(
                    "create_sdk_mcp_server(tools=...) 只接受通过 @mcp_tool 定义的工具函数"
                )

            tool_name = getattr(fn, "_bu_agent_sdk_mcp_tool_name", None)
            tool_desc = getattr(fn, "_bu_agent_sdk_mcp_tool_description", None)

            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("sdk mcp tool 缺少合法的 name")
            if not isinstance(tool_desc, str):
                tool_desc = ""

            # 注册到 FastMCP（schema 来自函数签名类型注解）
            server.tool(name=tool_name, description=tool_desc)(fn)  # type: ignore[arg-type]

            logger.debug(f"已注册 SDK MCP tool: {name}.{tool_name}")

    return {
        "type": "sdk",
        "name": name,
        "instance": server,
    }

