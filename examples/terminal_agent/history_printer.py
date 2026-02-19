from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console, Group
from rich.text import Text

from terminal_agent.models import HistoryEntry


def _entry_prefix(entry: HistoryEntry) -> str:
    if entry.entry_type == "user":
        return ">"
    if entry.entry_type == "assistant":
        return "⏺"
    if entry.entry_type == "tool_call":
        return "→"
    if entry.entry_type == "tool_result":
        return "●"
    if entry.entry_type == "thinking":
        return "💭"
    return "•"


def _entry_content(
    entry: HistoryEntry,
    *,
    terminal_width: int,
    render_markdown_to_plain: Callable[..., str],
) -> str:
    if entry.entry_type != "assistant":
        return str(entry.text)
    width = max(terminal_width - 6, 40)
    return render_markdown_to_plain(str(entry.text), width=width)


def render_history_group(
    console: Console,
    entries: list[HistoryEntry],
    *,
    terminal_width: int,
    render_markdown_to_plain: Callable[..., str],
) -> Group | None:
    if not entries:
        return None

    renderables: list[Any] = []
    for entry in entries:
        # Thinking entries: 灰色显示，无前缀
        if entry.entry_type == "thinking":
            content = str(entry.text)
            content_lines = content.splitlines() or [""]
            line_text = Text()
            line_text.append("  ", style="dim")
            line_text.append(content_lines[0], style="dim")
            for line in content_lines[1:]:
                line_text.append("\n")
                line_text.append("  ", style="dim")
                line_text.append(line, style="dim")
            renderables.append(line_text)
            renderables.append(Text(""))
            continue

        # Elapsed entries: 灰色显示，无前缀
        if entry.entry_type == "elapsed":
            line_text = Text()
            line_text.append(str(entry.text), style="dim")
            renderables.append(line_text)
            renderables.append(Text(""))
            continue

        if entry.entry_type == "tool_result":
            prefix_style = "bold red" if entry.is_error else "bold green"

            if isinstance(entry.text, Text):
                # Rich Text object — preserve styled content (e.g. colored diff)
                line_text = Text()
                line_text.append("● ", style=prefix_style)
                line_text.append_text(entry.text)
                renderables.append(line_text)
                renderables.append(Text(""))
                continue

            content = str(entry.text)
            content_lines = content.splitlines() or [""]

            line_text = Text()
            line_text.append("● ", style=prefix_style)
            line_text.append(content_lines[0])
            for line in content_lines[1:]:
                line_text.append("\n")
                line_text.append("  ")
                line_text.append(line)

            renderables.append(line_text)
            renderables.append(Text(""))
            continue

        if hasattr(entry.text, "__rich_console__"):
            prefix = _entry_prefix(entry)
            prefixed = Text(f"{prefix} ", style="bold")
            prefixed.append_text(entry.text)  # type: ignore[arg-type]
            renderables.append(prefixed)
        else:
            prefix = _entry_prefix(entry)
            content = _entry_content(
                entry,
                terminal_width=terminal_width,
                render_markdown_to_plain=render_markdown_to_plain,
            )
            content_lines = content.splitlines() or [""]
            lines = [f"{prefix} {content_lines[0]}"]
            for line in content_lines[1:]:
                lines.append(f"  {line}")
            renderables.append("\n".join(lines))

        renderables.append(Text(""))

    if not renderables:
        return None

    return Group(*renderables)


async def print_history_group_async(console: Console, group: Group) -> None:
    console.print(group)


def print_history_group_sync(console: Console, group: Group) -> None:
    console.print(group)
