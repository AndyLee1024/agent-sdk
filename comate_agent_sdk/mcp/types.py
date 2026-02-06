from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict, TypeAlias, runtime_checkable


class McpSdkServerConfig(TypedDict):
    """Configuration for SDK MCP servers created with create_sdk_mcp_server()."""

    type: Literal["sdk"]
    name: str
    instance: Any  # FastMCP instance


class McpStdioServerConfig(TypedDict):
    """MCP stdio server configuration.

    Notes:
        - type 字段允许缺省（向后兼容），默认按 stdio 处理。
    """

    type: NotRequired[Literal["stdio"]]
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]


class McpSSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]


class McpHttpServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]


McpServerConfig: TypeAlias = (
    McpStdioServerConfig | McpSSEServerConfig | McpHttpServerConfig | McpSdkServerConfig
)


McpServersInput: TypeAlias = dict[str, McpServerConfig] | str | Path | None


@runtime_checkable
class SdkMcpTool(Protocol):
    _comate_agent_sdk_mcp_tool_name: str
    _comate_agent_sdk_mcp_tool_description: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

