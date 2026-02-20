from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from comate_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core.runtime import AgentRuntime


class RuntimeMcpMixin:
    def invalidate_mcp_tools(self: "AgentRuntime", *, reason: str = "") -> None:
        self._mcp_dirty = True
        if reason:
            logger.debug(f"已标记 MCP tools dirty：{reason}")

    async def ensure_mcp_tools_loaded(self: "AgentRuntime", *, force: bool = False) -> None:
        if not bool(self.options.mcp_enabled):
            if self._mcp_loaded:
                self._remove_mcp_tools_from_agent()
            self._mcp_loaded = True
            self._mcp_dirty = False
            return

        if self._mcp_loaded and not force and not self._mcp_dirty:
            return

        async with self._mcp_load_lock:
            if self._mcp_loaded and not force and not self._mcp_dirty:
                return

            from comate_agent_sdk.mcp.config import resolve_mcp_servers
            from comate_agent_sdk.mcp.manager import McpManager

            servers = resolve_mcp_servers(
                self.options.mcp_servers,
                project_root=self.options.project_root,
            )
            if not servers:
                self._remove_mcp_tools_from_agent()
                self._mcp_loaded = True
                self._mcp_dirty = False
                return

            if self._mcp_manager is not None:
                try:
                    await self._mcp_manager.aclose()  # type: ignore[union-attr]
                except Exception as exc:
                    logger.warning(f"关闭旧 MCP manager 失败（忽略）：{exc}")
                finally:
                    self._mcp_manager = None

            manager = McpManager(servers)
            await manager.start()
            mcp_tools = manager.tools

            if self.options.tool_registry is not None:
                try:
                    if hasattr(self.options.tool_registry, "all") and hasattr(
                        self.options.tool_registry,
                        "unregister",
                    ):
                        for tool in self.options.tool_registry.all():  # type: ignore[attr-defined]
                            if getattr(tool, "_comate_agent_sdk_mcp_tool", False) is True:
                                self.options.tool_registry.unregister(tool.name)  # type: ignore[attr-defined]

                    for tool in mcp_tools:
                        self.options.tool_registry.register(tool)  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.warning(f"更新 tool_registry 的 MCP tools 失败：{exc}")

            self._apply_mcp_tools_to_agent(mcp_tools)

            lock_header = bool(getattr(self, "_lock_header_from_snapshot", False))
            if lock_header:
                logger.info("MCP header auto-refresh disabled by snapshot lock")
            elif mcp_tools:
                overview = manager.build_overview_text()
                meta = manager.build_metadata()
                try:
                    self._context.set_mcp_tools(overview, metadata=meta)
                except Exception as exc:
                    logger.warning(f"注入 MCP tools 到 ContextIR 失败：{exc}")
            else:
                try:
                    self._context.remove_mcp_tools()
                except Exception:
                    pass
                if servers:
                    logger.warning(
                        f"配置了 {len(servers)} 个 MCP server，但未加载到任何工具。"
                        "请检查 server 连接状态。"
                    )

            if self._tools_allowlist_mode and self._mcp_pending_tool_names:
                missing: list[str] = []
                resolved: list[Tool] = []
                mcp_by_name = {tool.name: tool for tool in mcp_tools}
                for name in self._mcp_pending_tool_names:
                    tool_obj = mcp_by_name.get(name)
                    if tool_obj is None:
                        missing.append(name)
                    else:
                        resolved.append(tool_obj)

                if missing:
                    raise ValueError(f"未找到 MCP tool(s): {missing}")

                existing_names = {
                    tool.name
                    for tool in self.options.tools
                    if isinstance(tool, Tool)
                }
                new_tools = list(self.options.tools)
                for tool in resolved:
                    if tool.name not in existing_names:
                        new_tools.append(tool)
                self.options.tools[:] = new_tools

                self._mcp_pending_tool_names = []

            self._tool_map = {
                tool.name: tool
                for tool in self.options.tools
                if isinstance(tool, Tool)
            }

            tool_names_from_list = {
                tool.name
                for tool in self.options.tools
                if isinstance(tool, Tool)
            }
            tool_names_from_map = set(self._tool_map.keys())
            if tool_names_from_list != tool_names_from_map:
                logger.error(
                    f"工具一致性检查失败！tools={len(tool_names_from_list)}, "
                    f"_tool_map={len(tool_names_from_map)}, "
                    f"差异={tool_names_from_list ^ tool_names_from_map}"
                )

            mcp_tool_names = [
                tool.name
                for tool in self.options.tools
                if isinstance(tool, Tool)
                and getattr(tool, "_comate_agent_sdk_mcp_tool", False)
            ]
            logger.info(f"MCP 工具加载完成：共 {len(mcp_tool_names)} 个工具")
            if mcp_tool_names:
                logger.debug(f"MCP 工具列表: {mcp_tool_names}")

            self._mcp_manager = manager
            self._mcp_loaded = True
            self._mcp_dirty = False

    def _remove_mcp_tools_from_agent(self: "AgentRuntime") -> None:
        try:
            self.options.tools[:] = [
                tool
                for tool in self.options.tools
                if not (
                    isinstance(tool, Tool)
                    and getattr(tool, "_comate_agent_sdk_mcp_tool", False) is True
                )
            ]
            self._tool_map = {
                tool.name: tool
                for tool in self.options.tools
                if isinstance(tool, Tool)
            }
        except Exception:
            pass

        try:
            if (
                self.options.tool_registry is not None
                and hasattr(self.options.tool_registry, "all")
                and hasattr(self.options.tool_registry, "unregister")
            ):
                for tool in self.options.tool_registry.all():  # type: ignore[attr-defined]
                    if getattr(tool, "_comate_agent_sdk_mcp_tool", False) is True:
                        self.options.tool_registry.unregister(tool.name)  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            self._context.remove_mcp_tools()
        except Exception:
            pass

    def _apply_mcp_tools_to_agent(self: "AgentRuntime", mcp_tools: list[Tool]) -> None:
        base_tools: list[Tool | str] = [
            tool
            for tool in self.options.tools
            if not (
                isinstance(tool, Tool)
                and getattr(tool, "_comate_agent_sdk_mcp_tool", False) is True
            )
        ]

        logger.debug(
            f"_apply_mcp_tools_to_agent: "
            f"base_tools={len(base_tools)}, "
            f"mcp_tools={len(mcp_tools)}, "
            f"allowlist_mode={self._tools_allowlist_mode}"
        )

        if self._tools_allowlist_mode:
            self.options.tools[:] = base_tools
            return

        merged: list[Tool | str] = list(base_tools)
        existing_names = {tool.name for tool in merged if isinstance(tool, Tool)}
        for tool in mcp_tools:
            if tool.name not in existing_names:
                merged.append(tool)

        logger.debug(f"合并后工具数量: {len(merged)}")
        self.options.tools[:] = merged
        logger.debug(f"赋值后 self.options.tools 数量: {len(self.options.tools)}")
