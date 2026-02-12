from __future__ import annotations

from collections import Counter
import json
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text

from terminal_agent.models import TodoItemState

_ALLOWED_STATUS = {"pending", "in_progress", "completed"}
_ALLOWED_PRIORITY = {"high", "medium", "low"}
_MAX_VISIBLE_ITEMS = 6


def _parse_todos_value(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _as_todos(args: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "todos" in args:
        parsed = _parse_todos_value(args["todos"])
        if parsed is not None:
            return parsed
    params = args.get("params")
    if isinstance(params, dict):
        parsed = _parse_todos_value(params.get("todos"))
        if parsed is not None:
            return parsed
    return None


def _normalize_todo(item: dict[str, Any]) -> TodoItemState | None:
    content = str(item.get("content", "")).strip()
    if not content:
        return None
    status = str(item.get("status", "pending")).strip().lower()
    if status not in _ALLOWED_STATUS:
        status = "pending"
    priority = str(item.get("priority", "medium")).strip().lower()
    if priority not in _ALLOWED_PRIORITY:
        priority = "medium"
    return TodoItemState(content=content, status=status, priority=priority)


def extract_todos(args: dict[str, Any]) -> list[TodoItemState] | None:
    raw = _as_todos(args)
    if raw is None:
        return None
    todos: list[TodoItemState] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_todo(item)
        if normalized is not None:
            todos.append(normalized)
    return todos


def _status_chip(status: str) -> str:
    if status == "completed":
        return "[green]✔[/green]"
    if status == "in_progress":
        return "[cyan]◉[/cyan]"
    return "[yellow]○[/yellow]"


def _priority_style(priority: str) -> str:
    if priority == "high":
        return "bold red"
    if priority == "low":
        return "dim"
    return "white"


def _status_label(status: str) -> str:
    if status == "completed":
        return "completed"
    if status == "in_progress":
        return "in_progress"
    return "pending"


def _priority_label(priority: str) -> str:
    if priority in _ALLOWED_PRIORITY:
        return priority
    return "medium"


class TodoStateStore:
    """Track latest todo snapshot and expose a renderable bottom layer."""

    def __init__(self) -> None:
        self._current: list[TodoItemState] = []
        self._last_change: str = ""

    def maybe_render_update(self, tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name != "TodoWrite":
            return False
        todos = extract_todos(args)
        if todos is None:
            return False
        self.update(todos)
        return True

    def update(self, todos: list[TodoItemState]) -> None:
        previous_by_content = {todo.content: todo for todo in self._current}
        self._last_change = self._compute_last_change(previous_by_content, todos)
        self._current = list(todos)

    def _compute_last_change(
        self,
        previous_by_content: dict[str, TodoItemState],
        todos: list[TodoItemState],
    ) -> str:
        for todo in todos:
            prev = previous_by_content.get(todo.content)
            if prev is None:
                return f"+ 新增: {todo.content} ({_status_label(todo.status)})"
            if prev.status != todo.status:
                return (
                    f"~ 状态: {todo.content} "
                    f"{_status_label(prev.status)} -> {_status_label(todo.status)}"
                )
            if prev.priority != todo.priority:
                return (
                    f"~ 优先级: {todo.content} "
                    f"{_priority_label(prev.priority)} -> {_priority_label(todo.priority)}"
                )
        if previous_by_content and not todos:
            return "~ 清空 todo 列表"
        return "= 无变化"

    def _summary_counts(self) -> Counter[str]:
        return Counter(todo.status for todo in self._current)

    def has_open_todos(self) -> bool:
        counts = self._summary_counts()
        return counts.get("pending", 0) > 0 or counts.get("in_progress", 0) > 0

    def visible_lines(self, *, max_visible_items: int = _MAX_VISIBLE_ITEMS) -> list[str]:
        if not self._current:
            return []

        counts = self._summary_counts()
        lines: list[str] = [
            (
                f"TODO  pending={counts.get('pending', 0)} | "
                f"in_progress={counts.get('in_progress', 0)} | "
                f"completed={counts.get('completed', 0)}"
            )
        ]

        open_items = [todo for todo in self._current if todo.status != "completed"]
        done_items = [todo for todo in self._current if todo.status == "completed"]
        display_items = (open_items + done_items)[: max(max_visible_items, 0)]
        display_open_items = [todo for todo in display_items if todo.status != "completed"]
        display_done_items = [todo for todo in display_items if todo.status == "completed"]

        for todo in display_open_items:
            marker = "◉" if todo.status == "in_progress" else "○"
            lines.append(f"  {marker} {todo.content} ({_priority_label(todo.priority)})")

        if display_open_items and display_done_items:
            lines.append("  ──────────────────────────────────────────────")

        for todo in display_done_items:
            lines.append(f"  ✔ ~~{todo.content}~~ ({_priority_label(todo.priority)})")

        hidden = len(self._current) - len(display_items)
        if hidden > 0:
            lines.append(f"  （还有 {hidden} 项未显示）")

        if self._last_change:
            lines.append(f"最近变更: {self._last_change}")

        return lines

    def renderable(self) -> RenderableType | None:
        plain_lines = self.visible_lines(max_visible_items=_MAX_VISIBLE_ITEMS)
        if not plain_lines:
            return None

        lines: list[RenderableType] = []
        summary_line = plain_lines[0]
        lines.append(Text(summary_line, style="bold cyan"))
        for idx, content in enumerate(plain_lines[1:], 1):
            if idx == len(plain_lines) - 1 and content.startswith("最近变更:"):
                lines.append(Text(content, style="dim"))
                continue
            if content.startswith("  （还有 "):
                lines.append(Text(content, style="dim"))
                continue
            lines.append(Text(content))
        return Group(*lines)
