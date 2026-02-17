"""Generic startup status display utilities for CLI applications.

Usage:
    from terminal_agent.startup import print_warning, print_success, print_error

    print_warning(console, "MCP server 'ctx7' skipped: header is null")
    print_success(console, "MCP loaded: exa_search (4 tools)")
    print_error(console, "Failed to initialize config")
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from rich.console import Console
from rich.text import Text

from terminal_agent.animations import PULSE_GLYPHS, _cyan_sweep_text


def print_status(console: Console, icon: str, style: str, message: str) -> None:
    """Print a styled status line to the console."""
    console.print(f"[{style}]{icon} {message}[/]")


def print_warning(console: Console, message: str) -> None:
    """Print a yellow warning line (⚠)."""
    print_status(console, "⚠", "yellow", message)


def print_success(console: Console, message: str) -> None:
    """Print a dim success line (✓)."""
    print_status(console, "✓", "dim", message)


def print_error(console: Console, message: str) -> None:
    """Print a red error line (✗)."""
    print_status(console, "✗", "red", message)


async def _run_mcp_animation(console: Console, server_names: list[str]) -> None:
    """Background task: render sweep animation until cancelled."""
    names = server_names if server_names else ["mcp"]
    servers_str = ", ".join(names)
    frame = 0
    while True:
        glyph = PULSE_GLYPHS[frame % len(PULSE_GLYPHS)]
        msg = f"{glyph}  Starting up mcp servers: {servers_str} "
        sweep = _cyan_sweep_text(msg, frame)
        console.print(sweep, end="\r")
        frame += 1
        await asyncio.sleep(0.1)


@asynccontextmanager
async def mcp_connecting_animation(
    console: Console, server_names: list[str]
) -> AsyncGenerator[None, None]:
    """Async context manager that shows a sweep animation while MCP connects."""
    task = asyncio.create_task(_run_mcp_animation(console, server_names))
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Clear the animation line
        console.print(Text(" " * 60), end="\r")
