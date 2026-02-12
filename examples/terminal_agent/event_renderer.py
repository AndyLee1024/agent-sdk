from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from comate_agent_sdk.agent.events import (
    PreCompactEvent,
    SessionInitEvent,
    StopEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    TextEvent,
    ThinkingEvent,
    TodoUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageDeltaEvent,
    UserQuestionEvent,
)

from terminal_agent.models import HistoryEntry
from terminal_agent.tool_view import summarize_tool_args

logger = logging.getLogger(__name__)


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


@dataclass
class _RunningTool:
    tool_name: str
    title: str
    started_at_monotonic: float
    is_task: bool
    args_summary: str
    progress_tokens: int = 0


class EventRenderer:
    """Convert SDK events to lightweight terminal state for prompt_toolkit UI."""

    def __init__(self) -> None:
        self._history: list[HistoryEntry] = []
        self._running_tools: dict[str, _RunningTool] = {}
        self._thinking_content: str = ""
        self._assistant_buffer = ""
        self._loading_line: str = ""

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
        if not self._running_tools:
            return
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

    def loading_line(self) -> str:
        return self._loading_line

    def append_system_message(self, content: str, *, is_error: bool = False) -> None:
        normalized = content.strip()
        if not normalized:
            return
        self._flush_assistant_segment()
        self._history.append(
            HistoryEntry(entry_type="system", text=normalized, is_error=is_error)
        )

    def append_assistant_message(self, content: str) -> None:
        normalized = content.strip()
        if not normalized:
            return
        self._flush_assistant_segment()
        self._history.append(HistoryEntry(entry_type="assistant", text=normalized))

    def _flush_assistant_segment(self) -> None:
        if not self._assistant_buffer:
            return
        self._history.append(HistoryEntry(entry_type="assistant", text=self._assistant_buffer))
        self._assistant_buffer = ""

    def _append_assistant_text(self, text: str) -> None:
        self._assistant_buffer += text

    def _append_tool_call(self, tool_name: str, args: dict[str, Any], tool_call_id: str) -> None:
        title = tool_name
        is_task = tool_name.lower() == "task"
        summary = summarize_tool_args(tool_name, args).strip()
        if is_task:
            title = _extract_task_title(args)
            summary = ""
        summary_suffix = f" {summary}" if summary else ""
        self._history.append(
            HistoryEntry(entry_type="tool_call", text=f"{title}{summary_suffix}")
        )

        self._running_tools[tool_call_id] = _RunningTool(
            tool_name=tool_name,
            title=title,
            started_at_monotonic=time.monotonic(),
            is_task=is_task,
            args_summary=summary,
        )

    def _append_tool_result(
        self,
        tool_name: str,
        tool_call_id: str,
        is_error: bool,
    ) -> None:
        state = self._running_tools.pop(tool_call_id, None)
        if state is None:
            self._history.append(
                HistoryEntry(
                    entry_type="tool_result",
                    text=f"{tool_name}",
                    is_error=is_error,
                )
            )
            return

        elapsed = _format_duration(time.monotonic() - state.started_at_monotonic)
        tokens_suffix = (
            f" · {_format_tokens(state.progress_tokens)}"
            if state.is_task and state.progress_tokens > 0
            else ""
        )
        self._history.append(
            HistoryEntry(
                entry_type="tool_result",
                text=f"{state.title} · {elapsed}{tokens_suffix}",
                is_error=is_error,
            )
        )

    def _rebuild_loading_line(self) -> None:
        if self._running_tools:
            first_key = next(iter(self._running_tools))
            state = self._running_tools[first_key]
            elapsed = _format_duration(time.monotonic() - state.started_at_monotonic)
            tokens_suffix = (
                f" · {_format_tokens(state.progress_tokens)}"
                if state.is_task and state.progress_tokens > 0
                else ""
            )
            more = len(self._running_tools) - 1
            more_suffix = f" (+{more})" if more > 0 else ""
            self._loading_line = f"⏳ {state.title} · {elapsed}{tokens_suffix}{more_suffix}"
            return

        if self._thinking_content:
            self._loading_line = f"🤔 {_truncate(self._thinking_content, 90)}"
            return

        self._loading_line = ""

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

    def _append_todo_update(self, todos: list[dict[str, Any]]) -> None:
        """将 todo 列表渲染为消息并添加到历史记录中。"""
        if not todos:
            return

        total = len(todos)
        completed = sum(1 for t in todos if t.get("status") == "completed")
        in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
        pending = total - completed - in_progress

        # 构建消息
        lines: list[str] = []
        lines.append(f"📋 任务列表 ({completed}/{total} 完成)")

        # 按状态分组：进行中 -> 待处理 -> 已完成
        open_items = [t for t in todos if t.get("status") != "completed"]
        done_items = [t for t in todos if t.get("status") == "completed"]

        for item in open_items:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            status = item.get("status", "pending")
            if status == "in_progress":
                lines.append(f"  ◉ {content} ⏳")
            else:
                lines.append(f"  ○ {content}")

        if open_items and done_items:
            lines.append("  ───────────────")

        for item in done_items:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"  ✓ ~~{content}~~")

        self._history.append(
            HistoryEntry(entry_type="system", text="\n".join(lines))
        )

    def handle_event(self, event: Any) -> tuple[bool, list[dict[str, Any]] | None]:
        if not isinstance(event, TextEvent):
            self._flush_assistant_segment()

        match event:
            case SessionInitEvent(session_id=_):
                pass
            case ThinkingEvent(content=thinking):
                self._thinking_content = thinking
            case PreCompactEvent(current_tokens=_, threshold=_, trigger=_):
                pass
            case ToolCallEvent(tool=tool_name, args=arguments, tool_call_id=tool_call_id):
                args_dict = arguments if isinstance(arguments, dict) else {"_raw": str(arguments)}
                self._thinking_content = ""
                self._append_tool_call(tool_name, args_dict, tool_call_id)
            case ToolResultEvent(tool=tool_name, result=_, tool_call_id=tool_call_id, is_error=is_error):
                self._append_tool_result(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    is_error=is_error,
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
                subagent_name=_,
                description=_,
                status=_,
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
            case SubagentStopEvent(tool_call_id=_, subagent_name=_, status=_, duration_ms=_, error=_):
                pass
            case TodoUpdatedEvent(todos=todos):
                self._append_todo_update(todos)
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
