from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session

from comate_agent_sdk.llm.base import ToolDefinition
from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL
from comate_agent_sdk.mcp.types import McpServerConfig
from comate_agent_sdk.mcp.utils import sanitize_tool_name
from comate_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("comate_agent_sdk.mcp.manager")

_MCP_TOOL_MARKER_ATTR = "_comate_agent_sdk_mcp_tool"
_MCP_TOOL_MARKER_VALUE = True


@dataclass(frozen=True)
class McpToolInfo:
    server_alias: str
    server_type: str
    remote_name: str
    mapped_name: str
    description: str
    input_schema: dict[str, Any]


def is_mcp_tool(tool: Tool) -> bool:
    return getattr(tool, _MCP_TOOL_MARKER_ATTR, False) is True


class McpManager:
    """MCP tools 管理器：负责连接 server、拉取 tools、以及将其封装为 SDK Tool。"""

    def __init__(
        self,
        servers: dict[str, McpServerConfig],
        *,
        connect_timeout_s: float = 10.0,
        call_timeout_s: float = 60.0,
    ) -> None:
        self._servers = dict(servers)
        self._connect_timeout_s = float(connect_timeout_s)
        self._call_timeout_s = float(call_timeout_s)

        self._exit_stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tool_info_by_mapped: dict[str, McpToolInfo] = {}
        self._tools: list[Tool] = []

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    @property
    def tool_infos(self) -> list[McpToolInfo]:
        return list(self._tool_info_by_mapped.values())

    async def start(self) -> None:
        if self._exit_stack is not None:
            return

        self._exit_stack = AsyncExitStack()

        for alias, cfg in self._servers.items():
            try:
                session = await self._exit_stack.enter_async_context(
                    self._connect(alias, cfg)
                )
                self._sessions[alias] = session

                tool_list = await self._list_all_tools(session)
                for t in tool_list:
                    mapped = self._map_tool_name(alias, t.name)
                    desc = (t.description or "").strip()
                    schema = self._normalize_input_schema(t.inputSchema)

                    info = McpToolInfo(
                        server_alias=alias,
                        server_type=self._server_type(cfg),
                        remote_name=t.name,
                        mapped_name=mapped,
                        description=desc,
                        input_schema=schema,
                    )
                    self._tool_info_by_mapped[mapped] = info

                logger.info(
                    f"已加载 MCP server={alias} tools={len(tool_list)}"
                )
            except Exception as e:
                logger.warning(f"MCP server 连接/加载失败，已跳过：{alias}：{e}")

        self._tools = [self._build_tool(mapped, info) for mapped, info in self._tool_info_by_mapped.items()]

    async def aclose(self) -> None:
        if self._exit_stack is None:
            return
        try:
            await self._exit_stack.aclose()
        finally:
            self._exit_stack = None
            self._sessions.clear()
            self._tool_info_by_mapped.clear()
            self._tools = []

    async def call_tool(self, mapped_name: str, arguments: dict[str, Any]) -> str | list[ContentPartTextParam | ContentPartImageParam]:
        info = self._tool_info_by_mapped.get(mapped_name)
        if info is None:
            return f"Error: Unknown MCP tool '{mapped_name}'"

        session = self._sessions.get(info.server_alias)
        if session is None:
            return f"Error: MCP server '{info.server_alias}' is not connected"

        try:
            result = await session.call_tool(
                info.remote_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=self._call_timeout_s),
            )
        except Exception as e:
            return f"Error: MCP tool call failed: {type(e).__name__}: {e}"

        return self._convert_call_tool_result(result)

    def build_overview_text(self) -> str:
        lines: list[str] = ["<mcp_tools>"]
        for mapped in sorted(self._tool_info_by_mapped.keys()):
            info = self._tool_info_by_mapped[mapped]
            desc = info.description or ""
            lines.append(f"- {mapped}: {desc}")
        lines.append("</mcp_tools>")
        return "\n".join(lines)

    def build_metadata(self) -> dict[str, Any]:
        tools_meta = [
            {
                "server_alias": i.server_alias,
                "server_type": i.server_type,
                "remote_name": i.remote_name,
                "mapped_name": i.mapped_name,
                "description": i.description,
                "input_schema": i.input_schema,
            }
            for i in self.tool_infos
        ]
        return {"tools": tools_meta}

    # ===== internal helpers =====

    def _server_type(self, cfg: McpServerConfig) -> str:
        t = cfg.get("type")  # type: ignore[attr-defined]
        if t is None:
            return "stdio"
        return str(t)

    def _map_tool_name(self, server_alias: str, remote_tool_name: str) -> str:
        alias = sanitize_tool_name(server_alias)
        tool = sanitize_tool_name(remote_tool_name)
        return f"mcp__{alias}__{tool}"

    def _normalize_input_schema(self, schema: Any) -> dict[str, Any]:
        if isinstance(schema, dict):
            # FastMCP / MCP server 往往会返回标准 JSON Schema（type=object + properties）。
            # 这里尽量原样保留，避免与 server 的实际校验/期望不一致。
            if schema.get("type") == "object" or "properties" in schema:
                return schema

        # 最小修复：无 schema 或非 object -> 空 object
        return {"type": "object", "properties": {}, "required": []}

    def _build_tool(self, mapped_name: str, info: McpToolInfo) -> Tool:
        async def _handler(**kwargs: Any) -> Any:
            return await self.call_tool(mapped_name, kwargs)

        tool = Tool(func=_handler, description=info.description or "", name=mapped_name, ephemeral=False)
        # 标记来源，方便刷新/移除
        setattr(tool, _MCP_TOOL_MARKER_ATTR, _MCP_TOOL_MARKER_VALUE)

        tool._definition = ToolDefinition(  # type: ignore[attr-defined]
            name=mapped_name,
            description=info.description or "",
            parameters=info.input_schema,
            strict=False,
        )
        return tool

    async def _list_all_tools(self, session: ClientSession) -> list[mcp_types.Tool]:
        tools: list[mcp_types.Tool] = []
        cursor: str | None = None

        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(list(result.tools or []))
            cursor = result.nextCursor
            if not cursor:
                break

        return tools

    def _convert_call_tool_result(
        self, result: mcp_types.CallToolResult
    ) -> str | list[ContentPartTextParam | ContentPartImageParam]:
        if bool(result.isError):
            text = self._render_mcp_content_as_text(result.content)
            return f"Error: {text}" if text else "Error: MCP tool returned error"

        parts: list[ContentPartTextParam | ContentPartImageParam] = []
        for item in result.content or []:
            if isinstance(item, mcp_types.TextContent):
                parts.append(ContentPartTextParam(text=item.text))
                continue
            if isinstance(item, mcp_types.ImageContent):
                mime = item.mimeType or "image/png"
                url = f"data:{mime};base64,{item.data}"
                parts.append(
                    ContentPartImageParam(
                        image_url=ImageURL(url=url, media_type=mime)  # type: ignore[arg-type]
                    )
                )
                continue

            # 其他类型：转为文本，避免丢信息
            try:
                parts.append(ContentPartTextParam(text=item.model_dump_json()))
            except Exception:
                parts.append(ContentPartTextParam(text=str(item)))

        # 若只有单段 text，则直接返回 str，减少序列化噪音
        if len(parts) == 1 and isinstance(parts[0], ContentPartTextParam):
            return parts[0].text

        return parts

    def _render_mcp_content_as_text(self, content: list[mcp_types.Content] | None) -> str:
        if not content:
            return ""
        texts: list[str] = []
        for item in content:
            if isinstance(item, mcp_types.TextContent):
                texts.append(item.text)
                continue
            try:
                texts.append(item.model_dump_json())
            except Exception:
                texts.append(str(item))
        return "\n".join(texts).strip()

    @asynccontextmanager
    async def _connect(self, alias: str, cfg: McpServerConfig):
        server_type = self._server_type(cfg)

        if server_type == "sdk":
            instance = cfg.get("instance")  # type: ignore[attr-defined]
            if instance is None:
                raise ValueError("sdk server 缺少 instance")
            async with create_connected_server_and_client_session(instance) as session:
                yield session
            return

        if server_type == "stdio":
            command = cfg.get("command")  # type: ignore[attr-defined]
            if not isinstance(command, str) or not command.strip():
                raise ValueError("stdio server 缺少 command")
            args = cfg.get("args") or []  # type: ignore[attr-defined]
            env = cfg.get("env")  # type: ignore[attr-defined]
            params = StdioServerParameters(command=command, args=list(args), env=env)
            async with stdio_client(params) as (read_stream, write_stream):
                session = ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self._connect_timeout_s),
                )
                async with session:
                    await session.initialize()
                    yield session
            return

        if server_type == "sse":
            url = cfg.get("url")  # type: ignore[attr-defined]
            if not isinstance(url, str) or not url.strip():
                raise ValueError("sse server 缺少 url")
            headers = cfg.get("headers")  # type: ignore[attr-defined]
            async with sse_client(
                url=url,
                headers=headers,
                timeout=self._connect_timeout_s,
            ) as (read_stream, write_stream):
                session = ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self._connect_timeout_s),
                )
                async with session:
                    await session.initialize()
                    yield session
            return

        if server_type == "http":
            url = cfg.get("url")  # type: ignore[attr-defined]
            if not isinstance(url, str) or not url.strip():
                raise ValueError("http server 缺少 url")
            headers = cfg.get("headers")  # type: ignore[attr-defined]

            http_client: httpx.AsyncClient | None = None
            try:
                if headers:
                    http_client = httpx.AsyncClient(headers=headers, timeout=self._connect_timeout_s)
                    await self._exit_stack.enter_async_context(http_client)  # type: ignore[union-attr]
                async with streamable_http_client(url, http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _get_session_id,
                ):
                    session = ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=self._connect_timeout_s),
                    )
                    async with session:
                        await session.initialize()
                        yield session
            finally:
                # http_client 由 exit_stack 接管时无需手动关闭
                pass
            return

        raise ValueError(f"Unsupported MCP server type: {server_type} (alias={alias})")
