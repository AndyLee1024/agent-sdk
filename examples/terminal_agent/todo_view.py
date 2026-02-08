from __future__ import annotations

from collections import Counter
from typing import Any

from rich.console import Console

from terminal_agent.models import TodoItemState

_ALLOWED_STATUS = {"pending", "in_progress", "completed"}
_ALLOWED_PRIORITY = {"high", "medium", "low"}


def _as_todos(args: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "todos" in args and isinstance(args["todos"], list):
        return args["todos"]
    params = args.get("params")
    if isinstance(params, dict) and isinstance(params.get("todos"), list):
        return params["todos"]
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


class TodoDiffView:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._current: list[TodoItemState] = []

    def maybe_render_update(self, tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name != "TodoWrite":
            return False
        todos = extract_todos(args)
        if todos is None:
            return False
        self.render_update(todos)
        return True

    def render_update(self, todos: list[TodoItemState]) -> None:
        previous_by_content = {todo.content: todo for todo in self._current}
        counts = Counter(todo.status for todo in todos)
        summary = (
            f"pending={counts.get('pending', 0)} | "
            f"in_progress={counts.get('in_progress', 0)} | "
            f"completed={counts.get('completed', 0)}"
        )
        self._console.print(f"[cyan]🗂 TODO Updated ({summary})[/]")
        for todo in todos:
            prev = previous_by_content.get(todo.content)
            change = "[green]+ new[/]"
            if prev is not None and prev.status != todo.status:
                change = f"[yellow]~ {prev.status} -> {todo.status}[/]"
            elif prev is not None and prev.priority != todo.priority:
                change = f"[yellow]~ priority {prev.priority} -> {todo.priority}[/]"
            elif prev is not None:
                change = "[dim]= unchanged[/]"

            content = todo.content
            if todo.status == "completed":
                content = f"[strike]{content}[/strike]"
            self._console.print(
                f"  {_status_chip(todo.status)} {content} "
                f"[{_priority_style(todo.priority)}]({todo.priority})[/] {change}"
            )
        self._current = list(todos)
