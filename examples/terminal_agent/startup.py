"""Generic startup status display utilities for CLI applications.

Usage:
    from terminal_agent.startup import print_warning, print_success, print_error

    print_warning(console, "MCP server 'ctx7' skipped: header is null")
    print_success(console, "MCP loaded: exa_search (4 tools)")
    print_error(console, "Failed to initialize config")
"""

from __future__ import annotations

from rich.console import Console


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
