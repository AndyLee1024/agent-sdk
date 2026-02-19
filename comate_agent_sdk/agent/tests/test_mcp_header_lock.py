from __future__ import annotations

import unittest
from unittest.mock import patch

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.tools import tool


class _FakeChatModel:
    def __init__(self) -> None:
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("This test should not call the LLM")


@tool("fake mcp tool")
async def MCPDemo() -> str:
    return "ok"


setattr(MCPDemo, "_comate_agent_sdk_mcp_tool", True)


class _FakeMcpManager:
    def __init__(self, servers):  # type: ignore[no-untyped-def]
        self._servers = servers
        self.tools = [MCPDemo]

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def build_overview_text(self) -> str:
        return "NEW_MCP_OVERVIEW"

    def build_metadata(self) -> dict[str, str]:
        return {"source": "fake"}


class TestMcpHeaderLock(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_lock_blocks_mcp_header_updates(self) -> None:
        template = Agent(
            llm=_FakeChatModel(),  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(),
                agents=(),
                offload_enabled=False,
                setting_sources=None,
            ),
        )
        runtime = template.create_runtime()
        runtime._lock_header_from_snapshot = True

        header_calls: list[str] = []
        runtime._context.set_mcp_tools = lambda *args, **kwargs: header_calls.append("set")  # type: ignore[method-assign]
        runtime._context.remove_mcp_tools = lambda *args, **kwargs: header_calls.append("remove")  # type: ignore[method-assign]

        with (
            patch("comate_agent_sdk.mcp.config.resolve_mcp_servers", return_value=[{"name": "fake"}]),
            patch("comate_agent_sdk.mcp.manager.McpManager", _FakeMcpManager),
        ):
            await runtime.ensure_mcp_tools_loaded(force=True)

        self.assertEqual(header_calls, [])
