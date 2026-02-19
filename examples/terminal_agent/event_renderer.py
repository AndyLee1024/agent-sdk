from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from comate_agent_sdk.agent.events import (
    PreCompactEvent,
    SessionInitEvent,
    StopEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    SubagentToolCallEvent,
    SubagentToolResultEvent,
    TextEvent,
    ThinkingEvent,
    TodoUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageDeltaEvent,
    UserQuestionEvent,
)

from rich.console import RenderableType
from rich.text import Text

from terminal_agent.models import HistoryEntry, LoadingState
from terminal_agent.tool_view import summarize_tool_args
from terminal_agent.env_utils import read_env_int

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_ERROR_SUMMARY_MAX_LEN = 160
_DEFAULT_TOOL_PANEL_MAX_LINES = 4
_DEFAULT_TODO_PANEL_MAX_LINES = 6


def _truncate(content: str, max_len: int = 120) -> str:
    if len(content) <= max_len:
        return content
    return f"{content[:max_len]}..."


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


def _format_tokens(token_count: int) -> str:
    tokens = max(int(token_count), 0)
    if tokens < 1_000:
        return f"{tokens} tok"
    compact = f"{tokens / 1_000:.1f}".rstrip("0").rstrip(".")
    return f"{compact}k tok"


def _extract_task_title(args: dict[str, Any]) -> str:
    description = str(args.get("description", "")).strip()
    if description:
        return description

    subagent_name = str(args.get("subagent_type", "")).strip() or "Task"
    return subagent_name


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _tool_signature(tool_name: str, args_summary: str) -> str:
    normalized = args_summary.strip()
    if normalized:
        return f"{tool_name}({normalized})"
    return f"{tool_name}()"


@dataclass
class _SubagentTool:
    """Subagent 内部的工具调用"""
    tool_name: str
    args_summary: str
    started_at_monotonic: float
    status: Literal["running", "completed", "error"] = "running"
    duration_ms: float = 0.0


@dataclass
class _RunningTool:
    tool_name: str
    title: str
    started_at_monotonic: float
    is_task: bool
    args_summary: str
    progress_tokens: int = 0
    subagent_name: str = ""
    subagent_status: str = ""
    subagent_description: str = ""
    nested_tools: list[tuple[str, _SubagentTool]] = field(default_factory=list)
    show_init: bool = False


class EventRenderer:
    """Convert SDK events to lightweight terminal state for prompt_toolkit UI."""

    def __init__(self, project_root: Path | None = None) -> None:
        self._history: list[HistoryEntry] = []
        self._running_tools: dict[str, _RunningTool] = {}
        self._thinking_content: str = ""
        self._assistant_buffer = ""
        self._loading_state: LoadingState = LoadingState.idle()
        self._current_todos: list[dict[str, Any]] = []
        self._todo_started_at_monotonic: float | None = None
        self._project_root = project_root
        self._tool_error_summary_max_len = read_env_int(
            "AGENT_SDK_TUI_TOOL_ERROR_SUMMARY_MAX_LEN",
            _DEFAULT_TOOL_ERROR_SUMMARY_MAX_LEN,
        )
        self._tool_panel_max_lines = read_env_int(
            "AGENT_SDK_TUI_TOOL_PANEL_MAX_LINES",
            _DEFAULT_TOOL_PANEL_MAX_LINES,
        )
        self._todo_panel_max_lines = read_env_int(
            "AGENT_SDK_TUI_TODO_PANEL_MAX_LINES",
            _DEFAULT_TODO_PANEL_MAX_LINES,
        )
        self._diff_max_lines = read_env_int(
            "AGENT_SDK_TUI_DIFF_MAX_LINES",
            50,
        )
        self._latest_diff_lines: list[str] | None = None

    def start_turn(self) -> None:
        self._flush_assistant_segment()
        self._thinking_content = ""
        self._rebuild_loading_line()

    def seed_user_message(self, content: str) -> None:
        normalized = content.strip()
        if not normalized:
            return
        self._flush_assistant_segment()
        self._history.append(HistoryEntry(entry_type="user", text=normalized))

    def close(self) -> None:
        return

    def finalize_turn(self) -> None:
        self._flush_assistant_segment()
        self._rebuild_loading_line()

    def tick_progress(self) -> None:
        self._rebuild_loading_line()

    def refresh_loading_animation(self) -> None:
        self._rebuild_loading_line()

    def interrupt_turn(self) -> None:
        if self._running_tools:
            self._history.append(
                HistoryEntry(
                    entry_type="system",
                    text=f"已中断当前任务（{len(self._running_tools)} 个运行中工具）",
                )
            )
        self._running_tools.clear()
        self._flush_assistant_segment()
        self._thinking_content = ""
        self._rebuild_loading_line()

    def history_entries(self) -> list[HistoryEntry]:
        return list(self._history)

    def reset_history_view(self) -> None:
        """重置 history 视图状态（用于会话切换后的重新加载）。"""
        self._history = []
        self._running_tools.clear()
        self._thinking_content = ""
        self._assistant_buffer = ""
        self._loading_state = LoadingState.idle()
        self._current_todos = []
        self._todo_started_at_monotonic = None
        self._latest_diff_lines = None

    @property
    def latest_diff_lines(self) -> list[str] | None:
        return self._latest_diff_lines

    def has_running_tools(self) -> bool:
        return bool(self._running_tools)

    def compute_required_tool_panel_lines(self) -> int:
        """计算显示所有 running tools 所需的最小行数。"""
        if not self._running_tools:
            return 0
        total_lines = 0
        for tool_call_id, state in self._running_tools.items():
            if state.is_task:
                total_lines += 1  # 主标题行
                # 嵌套工具（最多 3 个）
                total_lines += min(len(state.nested_tools), 3)
                # init 行（仅在创建后、首个有效子事件前显示）
                if state.show_init:
                    total_lines += 1
            else:
                total_lines += 1
        return total_lines

    def has_active_todos(self) -> bool:
        return bool(self._current_todos)

    def loading_state(self) -> LoadingState:
        """返回语义化的 loading 状态，用于 UI 层决定渲染策略。"""
        return self._loading_state

    def loading_line(self) -> str:
        """兼容旧接口，返回 loading 状态的文本内容。"""
        return self._loading_state.text

    def append_system_message(self, content: str, *, is_error: bool = False) -> None:
        normalized = content.strip()
        if not normalized:
            return
        self._flush_assistant_segment()
        self._history.append(
            HistoryEntry(entry_type="system", text=normalized, is_error=is_error)
        )

    def append_elapsed_message(self, content: str) -> None:
        """追加一条灰色无前缀的计时统计行到 history scrollback."""
        normalized = content.strip()
        if not normalized:
            return
        self._flush_assistant_segment()
        self._history.append(HistoryEntry(entry_type="elapsed", text=normalized))

    def append_assistant_message(self, content: str) -> None:
        normalized = content.strip()
        if not normalized:
            return
        self._flush_assistant_segment()
        self._history.append(HistoryEntry(entry_type="assistant", text=normalized))

    def tool_panel_entries(self, *, max_lines: int | None = None) -> list[tuple[int, str]]:
        """Return panel entries for running tools.

        Each entry is a tuple: (indent_level, text_without_dot_prefix).
        indent_level == 0 means a primary tool line; >0 means nested status.
        indent_level < 0 means a meta line (no dot prefix).
        """
        limit = max_lines if max_lines is not None else self._tool_panel_max_lines
        normalized_limit = max(1, int(limit))
        if not self._running_tools:
            return []

        now = time.monotonic()
        entries: list[tuple[int, str]] = []
        tool_items = list(self._running_tools.items())
        for idx, (tool_call_id, state) in enumerate(tool_items):
            elapsed = _format_duration(now - state.started_at_monotonic)
            lines_to_add = 1
            if state.is_task:
                tokens_suffix = (
                    f" · {_format_tokens(state.progress_tokens)}"
                    if state.progress_tokens > 0
                    else ""
                )
                # 主标题行 + 嵌套工具（最多 3 个）+ 状态行（如果有状态）
                lines_to_add = 1 + min(len(state.nested_tools), 3)
                if state.show_init:
                    lines_to_add += 1
            else:
                lines_to_add = 1

            if len(entries) + lines_to_add > normalized_limit:
                remaining_tools = len(tool_items) - idx
                entries.append((-1, f"… (+{remaining_tools})"))
                break

            if state.is_task:
                tokens_suffix = (
                    f" · {_format_tokens(state.progress_tokens)}"
                    if state.progress_tokens > 0
                    else ""
                )
                # 格式：SubagentName(描述)
                subagent_name = state.subagent_name or "Task"
                description = state.subagent_description or state.title
                title = f"{subagent_name}({description})"
                tool_count = len(state.nested_tools)
                tool_count_suffix = f" · +{tool_count} tool uses" if tool_count > 0 else ""
                entries.append((0, f"{title} · {elapsed}{tool_count_suffix}{tokens_suffix}"))

                # 嵌套工具调用（最多显示最近 3 个）
                for child_id, child_tool in state.nested_tools[-3:]:
                    signature = _tool_signature(child_tool.tool_name, child_tool.args_summary)
                    if child_tool.status == "running":
                        child_elapsed = _format_duration(now - child_tool.started_at_monotonic)
                        entries.append((1, f"|_ {signature} · {child_elapsed}"))
                    else:
                        icon = "✓" if child_tool.status == "completed" else "✗"
                        child_duration = _format_duration(child_tool.duration_ms / 1000)
                        entries.append((1, f"|_ {icon} {signature} · {child_duration}"))

                if state.show_init:
                    entries.append((1, "|_ init"))
            else:
                signature = _tool_signature(state.tool_name, state.args_summary)
                entries.append((0, f"{signature} · {elapsed}"))

        return entries[:normalized_limit]

    def todo_panel_lines(self, *, max_lines: int | None = None) -> list[str]:
        limit = max_lines if max_lines is not None else self._todo_panel_max_lines
        normalized_limit = max(1, int(limit))
        todos = list(self._current_todos)
        if not todos:
            return []

        total = len(todos)
        completed = sum(1 for item in todos if item.get("status") == "completed")
        in_progress = sum(1 for item in todos if item.get("status") == "in_progress")
        pending = sum(1 for item in todos if item.get("status") == "pending")
        header = f"📋 Todo ({completed}/{total} completed · {in_progress} in_progress · {pending} pending)"

        lines: list[str] = [header]
        for item in todos:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            status = str(item.get("status", "pending")).strip().lower()
            if status == "completed":
                lines.append(f"  ✓ {content}")
            elif status == "in_progress":
                lines.append(f"  ◉ {content} ⏳")
            else:
                lines.append(f"  ○ {content}")

        if len(lines) <= normalized_limit:
            return lines

        clipped = lines[: normalized_limit - 1]
        clipped.append(f"  … (+{len(lines) - (normalized_limit - 1)})")
        return clipped

    def _flush_assistant_segment(self) -> None:
        if not self._assistant_buffer:
            return
        self._history.append(HistoryEntry(entry_type="assistant", text=self._assistant_buffer))
        self._assistant_buffer = ""

    def _append_assistant_text(self, text: str) -> None:
        self._assistant_buffer += text

    def _append_tool_call(self, tool_name: str, args: dict[str, Any], tool_call_id: str) -> None:
        self._running_tools[tool_call_id] = self._make_running_tool(tool_name, args)

    def append_static_tool_result(
        self,
        signature: str,
        is_error: bool = False,
        diff_lines: list[str] | None = None,
    ) -> None:
        """Append a tool result to history as a static entry (no timer).

        Args:
            signature: 工具签名，例如 "Read(path=xxx)"
            is_error: 是否为错误结果
            diff_lines: optional diff lines for Edit/MultiEdit
        """
        if not is_error and diff_lines and len(diff_lines) > 0:
            self._latest_diff_lines = diff_lines
            text_obj = Text(signature)
            text_obj.append("\n")
            text_obj.append(self._render_diff_text(diff_lines, max_lines=self._diff_max_lines))
            self._history.append(
                HistoryEntry(entry_type="tool_result", text=text_obj, is_error=False)
            )
            return
        self._history.append(
            HistoryEntry(
                entry_type="tool_result",
                text=signature,
                is_error=is_error,
            )
        )

    def _make_running_tool(self, tool_name: str, args: dict[str, Any]) -> _RunningTool:
        """Create a _RunningTool from tool name and args dict."""
        title = tool_name
        is_task = tool_name.lower() == "task"
        summary = summarize_tool_args(tool_name, args, self._project_root).strip()
        subagent_name = ""
        subagent_description = ""
        if is_task:
            title = _extract_task_title(args)
            summary = ""
            subagent_name = str(args.get("subagent_type", "")).strip() or "Task"
            subagent_description = str(args.get("description", "")).strip()

        return _RunningTool(
            tool_name=tool_name,
            title=title,
            started_at_monotonic=time.monotonic(),
            is_task=is_task,
            args_summary=summary,
            show_init=is_task,
            subagent_name=subagent_name,
            subagent_description=subagent_description,
        )

    def _append_tool_result(
        self,
        tool_name: str,
        tool_call_id: str,
        is_error: bool,
        result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        state = self._running_tools.pop(tool_call_id, None)
        if state is None:
            signature = _tool_signature(tool_name, "")
            error_suffix = ""
            if is_error:
                error_suffix = f" · {_truncate(_one_line(result), self._tool_error_summary_max_len)}"
            self._history.append(HistoryEntry(entry_type="tool_result", text=f"{signature}{error_suffix}", is_error=is_error))
            return

        elapsed = _format_duration(time.monotonic() - state.started_at_monotonic)
        if state.is_task:
            subagent_name = state.subagent_name or "Task"
            description = state.subagent_description or state.title
            if description and description != subagent_name:
                task_title = f"{subagent_name}({description})"
            else:
                task_title = subagent_name
            tool_count = len(state.nested_tools)
            tool_count_suffix = f" · +{tool_count} tool uses" if tool_count > 0 else ""
            base = f"{task_title} · {elapsed}{tool_count_suffix}"
        else:
            display_name = "Update" if state.tool_name in ("Edit", "MultiEdit") else state.tool_name
            summary = state.args_summary
            # Append line range for Edit/MultiEdit
            if display_name == "Update" and metadata:
                sl = metadata.get("start_line")
                el = metadata.get("end_line")
                if isinstance(sl, int) and sl > 0:
                    line_suffix = f":L{sl}-L{el}" if isinstance(el, int) and el > sl else f":L{sl}"
                    summary = f"{summary}{line_suffix}" if summary else line_suffix
            signature = _tool_signature(display_name, summary)
            base = f"{signature} · {elapsed}"

        error_suffix = ""
        if is_error:
            error_suffix = f" · {_truncate(_one_line(result), self._tool_error_summary_max_len)}"

        # Render diff for Edit/MultiEdit if metadata contains diff lines
        if not is_error and metadata:
            diff_lines = metadata.get("diff")
            if isinstance(diff_lines, list) and len(diff_lines) > 0:
                self._latest_diff_lines = diff_lines
                text_obj = Text(f"{base}{error_suffix}")
                text_obj.append("\n")
                text_obj.append(self._render_diff_text(diff_lines, max_lines=self._diff_max_lines))
                self._history.append(
                    HistoryEntry(entry_type="tool_result", text=text_obj, is_error=False)
                )
                return

        self._history.append(HistoryEntry(entry_type="tool_result", text=f"{base}{error_suffix}", is_error=is_error))

    @staticmethod
    def _render_diff_text(diff_lines: list[str], max_lines: int = 50) -> Text:
        """Render diff lines as colored Rich Text with line numbers."""
        import re
        import shutil

        result = Text()
        total_lines = len(diff_lines)
        displayed_lines = diff_lines[:max_lines] if max_lines > 0 else diff_lines

        # Get terminal width for padding
        try:
            terminal_width = shutil.get_terminal_size().columns
        except Exception:
            terminal_width = 120

        old_line = 0
        new_line = 0
        hunk_pattern = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

        for line in displayed_lines:
            if line.startswith("---") or line.startswith("+++"):
                result.append(line, style="dim")
                result.append("\n")
                continue

            m = hunk_pattern.match(line)
            if m:
                old_line = int(m.group(1))
                new_line = int(m.group(2))
                result.append(line, style="cyan")
                result.append("\n")
                continue

            if line.startswith("-"):
                result.append(f"{old_line:>4} ", style="#ffffff on #4a0202")
                result.append(line, style="#ffffff on #4a0202")
                # Calculate padding to fill the entire line
                content_width = 5 + len(line)  # line number (4) + space (1) + content
                padding = max(0, terminal_width - content_width)
                result.append(" " * padding, style="#ffffff on #4a0202")
                result.append("\n")
                old_line += 1
            elif line.startswith("+"):
                result.append(f"{new_line:>4} ", style="#c6c6c6 on #2e502e")
                result.append(line, style="#ffffff on #2e502e")
                # Calculate padding to fill the entire line
                content_width = 5 + len(line)  # line number (4) + space (1) + content
                padding = max(0, terminal_width - content_width)
                result.append(" " * padding, style="#ffffff on #2e502e")
                result.append("\n")
                new_line += 1
            else:
                result.append(f"{new_line:>4} ", style="dim")
                result.append(line, style="dim")
                result.append("\n")
                old_line += 1
                new_line += 1

        if max_lines > 0 and total_lines > max_lines:
            result.append(
                f"... ({total_lines - max_lines} more lines, Ctrl+O to expand)",
                style="dim italic",
            )

        return result

    def _rebuild_loading_line(self) -> None:
        if self._thinking_content:
            text = f"🤔 {_truncate(self._thinking_content, 90)}"
            self._loading_state = LoadingState.thinking(
                text=text,
                content=self._thinking_content,
            )
            return

        self._loading_state = LoadingState.idle()

    def _append_questions(self, questions: list[dict[str, Any]]) -> None:
        if not questions:
            return
        self._history.append(
            HistoryEntry(
                entry_type="tool_result",
                text=f"需要输入：共有 {len(questions)} 个问题待回答。",
            )
        )
        for idx, question in enumerate(questions, 1):
            question_text = str(question.get("question", "")).strip()
            header = str(question.get("header", f"问题{idx}")).strip()
            options = question.get("options", [])
            labels: list[str] = []
            if isinstance(options, list):
                for option in options[:3]:
                    if not isinstance(option, dict):
                        continue
                    label = str(option.get("label", "")).strip()
                    if label:
                        labels.append(label)
            choice_preview = f"（可选: {' / '.join(labels)}）" if labels else ""
            self._history.append(
                HistoryEntry(
                    entry_type="tool_result",
                    text=f"  {idx}. {header}: {question_text} {choice_preview}".strip(),
                )
            )

    def _update_todos(self, todos: list[dict[str, Any]]) -> None:
        """更新当前 todo 列表状态。

        Todo panel is rendered in prompt_toolkit layout and should not spam scrollback.
        When all todos transition to completed, append a single summary line to history.
        """
        normalized = list(todos) if todos else []
        if not normalized:
            self._current_todos = []
            self._todo_started_at_monotonic = None
            return

        all_completed = all(str(item.get("status", "")).strip().lower() == "completed" for item in normalized)
        if not all_completed:
            if not self._current_todos:
                self._todo_started_at_monotonic = time.monotonic()
            self._current_todos = normalized
            return

        # All completed: hide panel and write a summary entry once.
        started = self._todo_started_at_monotonic
        elapsed_suffix = ""
        if started is not None:
            elapsed_suffix = f" · {_format_duration(time.monotonic() - started)}"
        total = len(normalized)
        self._history.append(
            HistoryEntry(
                entry_type="tool_result",
                text=f"todo {total}/{total} completed{elapsed_suffix}",
                is_error=False,
            )
        )
        self._current_todos = []
        self._todo_started_at_monotonic = None

    def todo_renderable(self) -> RenderableType | None:
        """将当前 todo 列表渲染为 Rich 组件。

        Returns:
            Rich RenderableType 或 None（如果没有 todo）
        """
        if not self._current_todos:
            return None

        todos = self._current_todos
        total = len(todos)
        completed = sum(1 for t in todos if t.get("status") == "completed")
        in_progress = sum(1 for t in todos if t.get("status") == "in_progress")

        # 构建 Rich Text 组件
        result = Text()

        # 标题
        result.append(f"📋 任务列表 ({completed}/{total} 完成)\n")

        # 按状态分组：进行中 -> 待处理 -> 已完成
        open_items = [t for t in todos if t.get("status") != "completed"]
        done_items = [t for t in todos if t.get("status") == "completed"]

        for item in open_items:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            status = item.get("status", "pending")
            if status == "in_progress":
                # 进行中：黄色 + 进度图标
                result.append(f"  ◉ ")
                result.append(f"{content}", style="yellow")
                result.append(" ⏳\n")
            else:
                # 待处理：默认颜色
                result.append(f"  ○ {content}\n")

        if open_items and done_items:
            result.append(f"  {'─' * 15}\n")

        for item in done_items:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            # 已完成：绿色 + 删除线
            result.append(f"  ✓ ")
            result.append(f"{content}", style="green strike")
            result.append("\n")

        # 移除末尾的换行
        if result.plain.endswith("\n"):
            result = result[:-1]

        return result

    def todo_lines(self) -> list[str]:
        """返回当前 todo 列表的行列表（用于测试）。"""
        return self.todo_panel_lines(max_lines=6)

    def handle_event(self, event: Any) -> tuple[bool, list[dict[str, Any]] | None]:
        if not isinstance(event, TextEvent):
            self._flush_assistant_segment()

        match event:
            case SessionInitEvent(session_id=_):
                pass
            case ThinkingEvent(content=thinking):
                self._thinking_content = thinking
                self._history.append(HistoryEntry(entry_type="thinking", text=thinking))
            case PreCompactEvent(current_tokens=_, threshold=_, trigger=_):
                pass
            case ToolCallEvent(tool=tool_name, args=arguments, tool_call_id=tool_call_id):
                args_dict = arguments if isinstance(arguments, dict) else {"_raw": str(arguments)}
                self._thinking_content = ""
                # TodoWrite: skip normal tool tracking; updates come via TodoUpdatedEvent.
                if tool_name.lower() == "todowrite":
                    self._rebuild_loading_line()
                    return (False, None)
                self._append_tool_call(tool_name, args_dict, tool_call_id)
            case ToolResultEvent(tool=tool_name, result=result, tool_call_id=tool_call_id, is_error=is_error, metadata=metadata):
                if tool_name.lower() == "todowrite" and not is_error:
                    # Successful TodoWrite should not spam scrollback.
                    self._rebuild_loading_line()
                    return (False, None)
                self._append_tool_result(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    is_error=is_error,
                    result=result,
                    metadata=metadata,
                )
            case UsageDeltaEvent(
                source=_,
                model=_,
                level=_,
                delta_prompt_tokens=_,
                delta_prompt_cached_tokens=_,
                delta_completion_tokens=_,
                delta_total_tokens=_,
            ):
                pass
            case SubagentStartEvent(tool_call_id=_, subagent_name=_, description=_):
                pass
            case SubagentProgressEvent(
                tool_call_id=tool_call_id,
                subagent_name=subagent_name,
                description=description,
                status=status,
                elapsed_ms=elapsed_ms,
                tokens=tokens,
            ):
                state = self._running_tools.get(tool_call_id)
                if state is not None:
                    if tokens is not None:
                        state.progress_tokens = max(int(tokens), 0)
                    if elapsed_ms is not None:
                        normalized = max(float(elapsed_ms), 0.0)
                        state.started_at_monotonic = time.monotonic() - (normalized / 1000)
                    # Subagent-specific status for task tool panel.
                    state.subagent_name = str(subagent_name or "").strip()
                    state.subagent_status = str(status or "").strip()
                    state.subagent_description = str(description or "").strip()
                    # 首个有效子事件后移除 init 占位。
                    progress_status = str(status or "").strip().lower()
                    has_activity = bool(tokens and int(tokens) > 0) or bool(
                        elapsed_ms and float(elapsed_ms) > 0.0
                    )
                    if has_activity or progress_status in {"completed", "error", "timeout", "cancelled"}:
                        state.show_init = False
            case SubagentStopEvent(tool_call_id=_, subagent_name=_, status=_, duration_ms=_, error=_):
                pass
            case SubagentToolCallEvent(
                parent_tool_call_id=parent_tool_call_id,
                subagent_name=_,
                tool=tool,
                args=args,
                tool_call_id=tool_call_id,
            ):
                # 将嵌套工具调用添加到父 Task 的 nested_tools
                parent_state = self._running_tools.get(parent_tool_call_id)
                if parent_state is not None:
                    args_summary = summarize_tool_args(tool, args, self._project_root).strip()
                    nested_tool = _SubagentTool(
                        tool_name=tool,
                        args_summary=args_summary,
                        started_at_monotonic=time.monotonic(),
                        status="running",
                    )
                    parent_state.nested_tools.append((tool_call_id, nested_tool))
                    parent_state.show_init = False
            case SubagentToolResultEvent(
                parent_tool_call_id=parent_tool_call_id,
                subagent_name=_,
                tool=_,
                tool_call_id=tool_call_id,
                is_error=is_error,
                duration_ms=duration_ms,
            ):
                # 更新嵌套工具的状态
                parent_state = self._running_tools.get(parent_tool_call_id)
                if parent_state is not None:
                    for nested_id, nested_tool in parent_state.nested_tools:
                        if nested_id == tool_call_id:
                            nested_tool.status = "error" if is_error else "completed"
                            nested_tool.duration_ms = duration_ms
                            parent_state.show_init = False
                            break
            case TodoUpdatedEvent(todos=todos):
                self._update_todos(todos)
            case UserQuestionEvent(questions=questions, tool_call_id=_):
                self._append_questions(questions)
                self._rebuild_loading_line()
                return (True, questions)
            case TextEvent(content=text):
                self._thinking_content = ""
                if text:
                    self._append_assistant_text(text)
            case StopEvent(reason=reason):
                self._flush_assistant_segment()
                self._thinking_content = ""
                self._rebuild_loading_line()
                if reason == "waiting_for_input":
                    return (True, None)
                if reason == "interrupted":
                    self._history.append(
                        HistoryEntry(entry_type="system", text="当前任务已中断。")
                    )
            case _:
                logger.debug("Unhandled event type: %s", type(event).__name__)

        self._rebuild_loading_line()
        return (False, None)
