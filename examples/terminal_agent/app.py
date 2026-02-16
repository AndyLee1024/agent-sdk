from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from rich.console import Console

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig, ChatSession
from comate_agent_sdk.context import EnvOptions
from comate_agent_sdk.tools import tool

from terminal_agent.event_renderer import EventRenderer
from terminal_agent.logo import print_logo
from terminal_agent.rpc_stdio import StdioRPCBridge
from terminal_agent.status_bar import StatusBar
from terminal_agent.tui import TerminalAgentTUI

console = Console()
logger = logging.getLogger(__name__)
logging.getLogger("comate_agent_sdk.system_tools.tools").setLevel(logging.ERROR)
logging.getLogger("comate_agent_sdk.mcp.manager").setLevel(logging.ERROR)


@tool("Add two numbers 涉及到加法运算 必须使用这个工具")
async def add(a: int, b: int) -> int:
    return a + b


def _build_agent() -> Agent:
    context7_api_key = os.getenv("CONTEXT7_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    exa_tools = (
        "web_search_exa,"
        "web_search_advanced_exa,"
        "get_code_context_exa,"
        "crawling_exa"
    )
    exa_url = "https://mcp.exa.ai/mcp"
    if exa_api_key:
        exa_url = f"{exa_url}?exaApiKey={exa_api_key}&tools={exa_tools}"
    else:
        exa_url = f"{exa_url}?tools={exa_tools}"

    return Agent(
        config=AgentConfig(
            role="software_engineering",
            env_options=EnvOptions(system_env=True, git_env=True),
            use_streaming_task=True,  # 启用流式 Task（实时显示嵌套工具调用）
            mcp_servers={
                "context7": {
                    "type": "http",
                    "url": "https://mcp.context7.com/mcp",
                    "headers": {
                        "CONTEXT7_API_KEY": context7_api_key,
                    },
                },
                "exa_search": {
                    "type": "http",
                    "url": exa_url,
                },
            },
        )
    )


def _resolve_session(agent: Agent, session_id: str | None) -> tuple[ChatSession, str]:
    if session_id:
        return ChatSession.resume(agent, session_id=session_id), "resume"
    return ChatSession(agent), "new"


async def _preload_mcp(session: ChatSession, con: Console) -> None:
    """Pre-load MCP tools and print connection status under the logo."""
    from terminal_agent.startup import print_error, print_success, print_warning

    try:
        await session._agent.ensure_mcp_tools_loaded()
    except Exception as e:
        print_error(con, f"MCP initialization failed: {e}")
        return

    mgr = session._agent._mcp_manager
    if mgr is None:
        return

    for alias, reason in mgr.failed_servers:
        print_warning(con, f"MCP server '{alias}' skipped: {reason}")

    loaded = mgr.tool_infos
    if loaded:
        count = len(loaded)
        aliases = sorted({i.server_alias for i in loaded})
        print_success(con, f"MCP loaded: {', '.join(aliases)} ({count} tools)")


async def run(*, rpc_stdio: bool = False, session_id: str | None = None) -> None:
    agent = _build_agent()
    session, mode = _resolve_session(agent, session_id)

    if rpc_stdio:
        bridge = StdioRPCBridge(session)
        try:
            await bridge.run()
        finally:
            await session.close()
        return

    print_logo(console)
    await _preload_mcp(session, console)
    status_bar = StatusBar(session)
    if mode == "resume":
        await status_bar.refresh()

    renderer = EventRenderer(project_root=Path.cwd())

    # 配置 TUI logging handler（将 SDK 日志输出到 TUI）
    from terminal_agent.logging_adapter import setup_tui_logging
    setup_tui_logging(renderer)

    tui = TerminalAgentTUI(session, status_bar, renderer)
    tui.add_resume_history(mode)

    try:
        await tui.run()
    finally:
        await session.close()

    console.print(
        f"[dim]Goodbye. Resume with: [bold cyan]comate resume "
        f"{session.session_id}[/][/]"
    )


if __name__ == "__main__":
    argv_session_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(session_id=argv_session_id))
