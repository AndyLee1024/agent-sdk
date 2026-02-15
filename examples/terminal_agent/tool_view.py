from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text


def _normalize_path_for_display(path_str: str | None, project_root: Path | None) -> str | None:
    """将路径转换为相对于项目根目录的显示格式。"""
    if path_str is None:
        return None
    try:
        path = Path(path_str)
        if project_root is not None and path.is_absolute():
            try:
                relative = path.relative_to(project_root)
                return relative.as_posix()
            except ValueError:
                # 路径不在 project_root 下，保持原样
                pass
        return path_str
    except Exception:
        return path_str


def _normalize_cwd_for_display(cwd: str | None, project_root: Path | None) -> str | None:
    """将工作目录转换为相对于项目根目录的显示格式。"""
    return _normalize_path_for_display(cwd, project_root)

from terminal_agent.message_style import (
    TOOL_ERROR_PREFIX,
    TOOL_ERROR_STYLE,
    TOOL_RUNNING_PREFIX,
    TOOL_RUNNING_STYLE,
    TOOL_SUCCESS_PREFIX,
    TOOL_SUCCESS_STYLE,
)
from terminal_agent.models import TodoItemState, ToolRunState

_PULSE_GLYPHS: tuple[str, ...] = ("◐", "◓", "◑", "◒")
_HIDDEN_ARG_TOOLS: frozenset[str] = frozenset({"askuserquestion"})
_TASK_PROMPT_FALLBACK_LEN = 40
_SWEEP_SPEED_MULTIPLIER = 2
_MAX_FANCY_TASKS = 2
_MAX_FANCY_LINE_LEN = 100

# Todo extraction helpers
_ALLOWED_STATUS = {"pending", "in_progress", "completed"}
_ALLOWED_PRIORITY = {"high", "medium", "low"}


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
    """从 TodoWrite 工具参数中提取 todo 列表。"""
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


def _truncate(content: str, max_len: int = 280) -> str:
    if len(content) <= max_len:
        return content
    return f"{content[:max_len]}..."


def _normalize_inline(content: str) -> str:
    return " ".join(content.split())


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


def _extract_task_identity(args: dict[str, Any]) -> tuple[str, str]:
    raw_subagent = _lookup_arg(args, "subagent_type")
    subagent_name = raw_subagent.strip() if isinstance(raw_subagent, str) else ""
    if not subagent_name:
        subagent_name = "Task"

    raw_desc = _lookup_arg(args, "description")
    description = raw_desc.strip() if isinstance(raw_desc, str) else ""

    if not description:
        raw_prompt = _lookup_arg(args, "prompt")
        if isinstance(raw_prompt, str):
            prompt_text = _normalize_inline(raw_prompt)
            description = _truncate(prompt_text, _TASK_PROMPT_FALLBACK_LEN)

    if not description:
        description = subagent_name

    return subagent_name, description


def _format_tokens(token_count: int) -> str:
    tokens = max(int(token_count), 0)
    if tokens < 1_000:
        return f"{tokens} tok"
    if tokens < 1_000_000:
        compact = f"{tokens / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}k tok"
    compact = f"{tokens / 1_000_000:.1f}".rstrip("0").rstrip(".")
    return f"{compact}m tok"


def _format_duration(seconds: float) -> str:
    elapsed = max(seconds, 0.0)
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    minutes = int(elapsed // 60)
    remaining_seconds = int(elapsed % 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h{remaining_minutes:02d}m"


def _lerp_rgb(
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, ratio))
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * clamped)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * clamped)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * clamped)
    return r, g, b


def _sweep_gradient_text(content: str, frame: int) -> Text:
    text = Text()
    if not content:
        return text

    total = len(content)
    base_rgb = (96, 124, 156)
    mid_rgb = (118, 195, 225)
    high_rgb = (218, 246, 255)

    window = max(4, total // 6)
    cycle = max(total + window * 2, 20)
    center = (frame % cycle) - window

    for idx, ch in enumerate(content):
        distance = abs(idx - center)
        if distance <= window:
            glow = 1.0 - (distance / window)
            if glow >= 0.6:
                r, g, b = _lerp_rgb(mid_rgb, high_rgb, (glow - 0.6) / 0.4)
            else:
                r, g, b = _lerp_rgb(base_rgb, mid_rgb, glow / 0.6)
        else:
            r, g, b = base_rgb
        text.append(ch, style=f"bold rgb({r},{g},{b})")
    return text


def summarize_tool_args(
    tool_name: str, args: dict[str, Any], project_root: Path | None = None
) -> str:
    """Summarize tool arguments for display.

    Args:
        tool_name: Name of the tool being called.
        args: Tool arguments dictionary.
        project_root: Optional project root path. If provided, absolute paths
            will be converted to relative paths for display.

    Returns:
        A compact string representation of the tool arguments.
    """
    if _should_hide_tool_args(tool_name):
        return ""
    lowered = tool_name.lower()
    if lowered == "write":
        path = _lookup_arg(args, "file_path", "path")
        path_display = _normalize_path_for_display(path, project_root)
        return f"path={path_display}" if path_display else _compact_json(args)
    if lowered == "edit":
        path = _lookup_arg(args, "file_path", "path")
        path_display = _normalize_path_for_display(path, project_root)
        return path_display or _compact_json(args)
    if lowered == "multiedit":
        path = _lookup_arg(args, "file_path", "path")
        path_display = _normalize_path_for_display(path, project_root)
        return path_display or _compact_json(args)
    if lowered == "read":
        path = _lookup_arg(args, "file_path", "path")
        path_display = _normalize_path_for_display(path, project_root)
        offset = _lookup_arg(args, "offset_line")
        limit = _lookup_arg(args, "limit_lines")
        return f"path={path_display} offset={offset} limit={limit}" if path_display else _compact_json(args)
    if lowered in {"grep", "glob", "ls"}:
        pattern = _lookup_arg(args, "pattern")
        path = _lookup_arg(args, "path")
        path_display = _normalize_path_for_display(path, project_root)
        return f"path={path_display} pattern={pattern}" if (path_display or pattern) else _compact_json(args)
    if lowered == "bash":
        command = _lookup_arg(args, "command")
        cwd = _lookup_arg(args, "cwd")
        cwd_display = _normalize_cwd_for_display(cwd, project_root)
        if cwd_display:
            return f"cwd={cwd_display} command={_truncate(str(command), 180)}" if command else _compact_json(args)
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
    """Tool state tracker for loading layer + in-message mutable tool lines."""

    def __init__(
        self,
        *,
        fancy_progress_effect: bool = True,
    ) -> None:
        self._state_by_id: dict[str, ToolRunState] = {}
        self._tool_text_refs: dict[str, Text] = {}
        self._frame = 0
        self._latest_total_tokens = 0
        self._latest_tokens_by_source_prefix: dict[str, int] = {}
        self._fancy_progress_effect = fancy_progress_effect

    def reset_turn(self) -> None:
        self._state_by_id.clear()
        self._tool_text_refs.clear()

    @staticmethod
    def _task_title(state: ToolRunState) -> str:
        if state.task_desc and state.task_desc != state.subagent_name:
            return f"{state.subagent_name}({state.task_desc})"
        return state.subagent_name or "Task"

    def _running_states(self, *, only_tasks: bool = False) -> list[ToolRunState]:
        states = [state for state in self._state_by_id.values() if state.status == "running"]
        if only_tasks:
            return [state for state in states if state.is_task]
        return states

    def _running_line(self, state: ToolRunState, pulse: str, now: float) -> str:
        elapsed = _format_duration(now - state.started_at_monotonic)
        if state.is_task:
            token_text = self._task_token_text(state)
            return f"{pulse} {self._task_title(state)} · 运行中 · {elapsed} · {token_text}"

        summary_suffix = (
            f" {state.args_summary}" if state.args_summary and state.args_summary != "hidden" else ""
        )
        return f"{pulse} {state.tool_name}{summary_suffix} · 运行中 · {elapsed}"

    def _should_use_fancy_effect(self, lines: list[str]) -> bool:
        if not self._fancy_progress_effect:
            return False
        if len(lines) > _MAX_FANCY_TASKS:
            return False
        if any(len(line) > _MAX_FANCY_LINE_LEN for line in lines):
            return False
        return True

    def has_running_tasks(self) -> bool:
        return any(state.status == "running" and state.is_task for state in self._state_by_id.values())

    def tick_progress(self) -> None:
        if not self.has_running_tasks():
            return
        self._frame += 1

    def running_subagent_source_prefixes(self) -> set[str]:
        return {
            state.subagent_source_prefix
            for state in self._state_by_id.values()
            if state.status == "running" and state.is_task and state.subagent_source_prefix
        }

    def set_task_source_baseline(self, tool_call_id: str, source_total_tokens: int) -> None:
        state = self._state_by_id.get(tool_call_id)
        if state is None or not state.is_task or not state.subagent_source_prefix:
            return

        normalized = max(int(source_total_tokens), 0)
        state.baseline_source_tokens = normalized
        state.task_tokens = 0
        state.last_progress_tokens = 0
        self._latest_tokens_by_source_prefix[state.subagent_source_prefix] = normalized

    def update_task_progress(
        self,
        *,
        tool_call_id: str,
        tokens: int | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        state = self._state_by_id.get(tool_call_id)
        if state is None or not state.is_task:
            return

        if tokens is not None:
            normalized_tokens = max(int(tokens), 0)
            state.task_tokens = normalized_tokens
            state.last_progress_tokens = normalized_tokens

        if elapsed_ms is not None:
            normalized_elapsed = max(float(elapsed_ms), 0.0)
            state.started_at_monotonic = time.monotonic() - (normalized_elapsed / 1000)

    def _task_token_text(self, state: ToolRunState) -> str:
        if state.subagent_source_prefix:
            return _format_tokens(state.task_tokens)
        return _format_tokens(self._latest_total_tokens)

    def update_usage_tokens(
        self,
        total_tokens: int,
        source_totals: dict[str, int] | None = None,
    ) -> None:
        self._latest_total_tokens = max(int(total_tokens), 0)
        if source_totals:
            for source_prefix, source_tokens in source_totals.items():
                self._latest_tokens_by_source_prefix[source_prefix] = max(int(source_tokens), 0)

        for state in self._state_by_id.values():
            if state.status != "running" or not state.is_task or not state.subagent_source_prefix:
                continue
            current_total = self._latest_tokens_by_source_prefix.get(state.subagent_source_prefix)
            if current_total is None:
                continue
            task_tokens = max(current_total - state.baseline_source_tokens, 0)
            state.task_tokens = task_tokens
            state.last_progress_tokens = task_tokens

    def interrupt_running(self) -> None:
        running_states = [state for state in self._state_by_id.values() if state.status == "running"]
        if not running_states:
            self._state_by_id.clear()
            self._tool_text_refs.clear()
            return
        now = time.monotonic()
        for state in running_states:
            text_ref = self._tool_text_refs.get(state.tool_call_id)
            if text_ref is None:
                continue
            if state.is_task:
                elapsed = _format_duration(now - state.started_at_monotonic)
                token_text = self._task_token_text(state)
                self._overwrite_text_ref(
                    text_ref,
                    f"⏹ {self._task_title(state)} · 已中断 · {elapsed} · {token_text}",
                    "bold yellow",
                )
            else:
                summary_suffix = (
                    f" {state.args_summary}" if state.args_summary and state.args_summary != "hidden" else ""
                )
                self._overwrite_text_ref(
                    text_ref,
                    f"⏹ {state.tool_name}{summary_suffix} 已中断",
                    "bold yellow",
                )
        self._state_by_id.clear()
        self._tool_text_refs.clear()

    @staticmethod
    def _overwrite_text_ref(text_ref: Text, content: str, style: str) -> None:
        text_ref.truncate(0)
        text_ref.append("  ")
        text_ref.append(content, style=style)

    @staticmethod
    def _summary_suffix(summary: str) -> str:
        return f" {summary}" if summary and summary != "hidden" else ""

    def render_call(self, tool_name: str, args: dict[str, Any], tool_call_id: str) -> Text:
        hide_args = _should_hide_tool_args(tool_name)
        summary = summarize_tool_args(tool_name, args)
        now = time.monotonic()
        is_task = tool_name.lower() == "task"

        subagent_name = ""
        task_desc = ""
        subagent_source_prefix = ""
        baseline_source_tokens = 0
        if is_task:
            subagent_name, task_desc = _extract_task_identity(args)
            subagent_source_prefix = f"subagent:{subagent_name}:{tool_call_id}" if subagent_name else ""
            baseline_source_tokens = self._latest_tokens_by_source_prefix.get(subagent_source_prefix, 0)

        state = ToolRunState(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            args_summary="hidden" if hide_args else summary,
            status="running",
            started_at_monotonic=now,
            is_task=is_task,
            subagent_name=subagent_name,
            task_desc=task_desc,
            subagent_source_prefix=subagent_source_prefix,
            baseline_source_tokens=baseline_source_tokens,
            task_tokens=0,
            last_progress_render_ts=now,
            last_progress_tokens=0 if is_task else self._latest_total_tokens,
        )
        self._state_by_id[tool_call_id] = state

        text_ref = Text()
        self._tool_text_refs[tool_call_id] = text_ref
        if state.is_task:
            line_content = f"{TOOL_RUNNING_PREFIX} {self._task_title(state)} · 运行中"
        else:
            line_content = f"{TOOL_RUNNING_PREFIX} {tool_name}{self._summary_suffix(state.args_summary)}"
        self._overwrite_text_ref(text_ref, line_content, TOOL_RUNNING_STYLE)
        return text_ref

    def render_result(
        self,
        tool_name: str,
        tool_call_id: str,
        result: Any,
        is_error: bool,
    ) -> None:
        state = self._state_by_id.pop(tool_call_id, None)
        text_ref = self._tool_text_refs.pop(tool_call_id, None)
        if text_ref is None:
            return

        if state and state.is_task:
            elapsed = _format_duration(time.monotonic() - state.started_at_monotonic)
            token_text = self._task_token_text(state)
            if is_error:
                self._overwrite_text_ref(
                    text_ref,
                    f"{TOOL_ERROR_PREFIX} {self._task_title(state)} · 失败 · {elapsed} · {token_text}",
                    TOOL_ERROR_STYLE,
                )
                return
            self._overwrite_text_ref(
                text_ref,
                f"{TOOL_SUCCESS_PREFIX} {self._task_title(state)} · 完成 · {elapsed} · {token_text}",
                TOOL_SUCCESS_STYLE,
            )
            return

        args_summary = state.args_summary if state else ""
        summary_suffix = self._summary_suffix(args_summary)
        if is_error:
            self._overwrite_text_ref(
                text_ref,
                f"{TOOL_ERROR_PREFIX} {tool_name}{summary_suffix}",
                TOOL_ERROR_STYLE,
            )
            return
        self._overwrite_text_ref(
            text_ref,
            f"{TOOL_SUCCESS_PREFIX} {tool_name}{summary_suffix}",
            TOOL_SUCCESS_STYLE,
        )

    def renderable(self, *, only_tasks: bool = True) -> RenderableType | None:
        running_states = self._running_states(only_tasks=only_tasks)
        if not running_states:
            return None

        now = time.monotonic()
        lines = [
            self._running_line(
                state,
                _PULSE_GLYPHS[(self._frame + idx) % len(_PULSE_GLYPHS)],
                now,
            )
            for idx, state in enumerate(running_states)
        ]

        composed = Text()
        use_fancy_effect = self._should_use_fancy_effect(lines)
        phase = self._frame * _SWEEP_SPEED_MULTIPLIER
        for idx, line in enumerate(lines):
            if use_fancy_effect:
                composed.append_text(_sweep_gradient_text(line, frame=phase + idx * 5))
            else:
                composed.append(line, style="dim")
            if idx < len(lines) - 1:
                composed.append("\n")

        return Panel(
            composed,
            title="⏳ Loading",
            border_style="blue",
            padding=(0, 1),
        )
