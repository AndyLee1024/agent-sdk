from __future__ import annotations

import json
from typing import Any

from rich.console import Console

from terminal_agent.models import ToolRunState
from terminal_agent.todo_view import extract_todos

_PULSE_GLYPHS: tuple[str, ...] = ("◐", "◓", "◑", "◒")
_HIDDEN_ARG_TOOLS: frozenset[str] = frozenset({"askuserquestion"})


def _truncate(content: str, max_len: int = 280) -> str:
    if len(content) <= max_len:
        return content
    return f"{content[:max_len]}..."


def _lookup_arg(args: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in args:
            return args.get(key)
    params = args.get("params")
    if isinstance(params, dict):
        for key in keys:
            if key in params:
                return params.get(key)
    return None


def _compact_json(value: Any, max_len: int = 220) -> str:
    try:
        content = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        content = str(value)
    return _truncate(content, max_len=max_len)


def _should_hide_tool_args(tool_name: str) -> bool:
    return tool_name.lower() in _HIDDEN_ARG_TOOLS


def summarize_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    if _should_hide_tool_args(tool_name):
        return ""
    lowered = tool_name.lower()
    if lowered == "write":
        path = _lookup_arg(args, "file_path", "path")
        return f"path={path}" if path else _compact_json(args)
    if lowered == "edit":
        path = _lookup_arg(args, "file_path", "path")
        old_len = len(str(_lookup_arg(args, "old_string") or ""))
        new_len = len(str(_lookup_arg(args, "new_string") or ""))
        return f"path={path} old_len={old_len} new_len={new_len}" if path else _compact_json(args)
    if lowered == "read":
        path = _lookup_arg(args, "file_path", "path")
        offset = _lookup_arg(args, "offset")
        limit = _lookup_arg(args, "limit")
        return f"path={path} offset={offset} limit={limit}" if path else _compact_json(args)
    if lowered in {"grep", "glob", "ls"}:
        pattern = _lookup_arg(args, "pattern")
        path = _lookup_arg(args, "path")
        return f"path={path} pattern={pattern}" if (path or pattern) else _compact_json(args)
    if lowered == "bash":
        command = _lookup_arg(args, "command")
        return f"command={_truncate(str(command), 180)}" if command else _compact_json(args)
    if lowered == "webfetch":
        url = _lookup_arg(args, "url")
        return f"url={url}" if url else _compact_json(args)
    if lowered == "todowrite":
        todos = extract_todos(args)
        if todos is None:
            return _compact_json(args)
        pending = sum(1 for todo in todos if todo.status == "pending")
        in_progress = sum(1 for todo in todos if todo.status == "in_progress")
        completed = sum(1 for todo in todos if todo.status == "completed")
        return (
            f"todos={len(todos)} pending={pending} "
            f"in_progress={in_progress} completed={completed}"
        )
    return _compact_json(args)


class ToolEventView:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._state_by_id: dict[str, ToolRunState] = {}
        self._frame = 0

    def render_call(self, tool_name: str, args: dict[str, Any], tool_call_id: str) -> None:
        hide_args = _should_hide_tool_args(tool_name)
        summary = summarize_tool_args(tool_name, args)
        pulse = _PULSE_GLYPHS[self._frame % len(_PULSE_GLYPHS)]
        self._frame += 1
        self._state_by_id[tool_call_id] = ToolRunState(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            args_summary="hidden" if hide_args else summary,
            status="running",
        )
        if summary:
            self._console.print(f"[dim]{pulse} {tool_name} {summary}[/]")
            return
        self._console.print(f"[dim]{pulse} {tool_name}[/]")

    def render_result(
        self,
        tool_name: str,
        tool_call_id: str,
        result: str,
        is_error: bool,
    ) -> None:
        if not is_error:
            return
        state = self._state_by_id.get(tool_call_id)
        args_summary = state.args_summary if state else "unknown"
        preview = _truncate(str(result), 200)
        self._console.print(
            f"[red]✖ {tool_name} failed: {preview}[/]\n"
            f"[dim]  args: {args_summary}[/]"
        )
