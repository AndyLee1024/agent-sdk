from __future__ import annotations

from rich.console import Console

ASSISTANT_PREFIX = "⏺"
ASSISTANT_PREFIX_STYLE = "bold cyan"
ASSISTANT_MESSAGE_GAP_LINES = 1
USER_PREFIX = ">"
USER_PREFIX_STYLE = "bold ansicyan"

TOOL_RUNNING_STYLE = "bold blue"
TOOL_SUCCESS_STYLE = "bold green"
TOOL_ERROR_STYLE = "bold red"
TOOL_RUNNING_PREFIX = "→"
TOOL_SUCCESS_PREFIX = "✓"
TOOL_ERROR_PREFIX = "✖"

THINKING_PREFIX = "💭"
THINKING_STYLE = "dim"  # 灰色


def print_assistant_prefix_line(console: Console, content: str) -> None:
    """Print assistant prefix and content on the same visual line."""
    console.print(
        f"[{ASSISTANT_PREFIX_STYLE}]{ASSISTANT_PREFIX}[/] {content}",
        new_line_start=True,
        soft_wrap=True,
    )


def print_assistant_gap(console: Console) -> None:
    for _ in range(ASSISTANT_MESSAGE_GAP_LINES):
        console.print()
