from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.text import Text

from comate_agent_sdk.agent.events import (
    PreCompactEvent,
    SessionInitEvent,
    StopEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageDeltaEvent,
    UserQuestionEvent,
)

from terminal_agent.assistant_render import AssistantStreamRenderer
from terminal_agent.todo_view import TodoDiffView
from terminal_agent.tool_view import ToolEventView

_THINKING_GLYPHS: tuple[str, ...] = ("◐", "◓", "◑", "◒")


def _truncate(content: str, max_len: int = 300) -> str:
    if len(content) <= max_len:
        return content
    return f"{content[:max_len]}..."


class EventRenderer:
    """Convert SDK stream events into structured terminal UI output."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._assistant = AssistantStreamRenderer(console)
        self._tools = ToolEventView(console)
        self._todo = TodoDiffView(console)
        self._thinking_live: Live | None = None
        self._thinking_frame = 0
        self._pending_gap_before_assistant = False
        self._overlay_active = False

    def start_turn(self) -> None:
        self._assistant.start_turn()
        self._stop_thinking()
        self._pending_gap_before_assistant = False

    def set_overlay_active(self, active: bool) -> None:
        self._overlay_active = active
        self._tools.set_live_suspended(active)
        if active:
            self._stop_thinking()

    def update_usage_tokens(
        self,
        total_tokens: int,
        source_totals: dict[str, int] | None = None,
    ) -> None:
        self._tools.update_usage_tokens(total_tokens, source_totals)

    def get_running_subagent_source_prefixes(self) -> set[str]:
        return self._tools.running_subagent_source_prefixes()

    def set_task_source_baseline(self, tool_call_id: str, source_total_tokens: int) -> None:
        self._tools.set_task_source_baseline(
            tool_call_id=tool_call_id,
            source_total_tokens=source_total_tokens,
        )

    def has_running_tasks(self) -> bool:
        return self._tools.has_running_tasks()

    def tick_progress(self) -> None:
        self._tools.tick_progress()

    def interrupt_turn(self) -> None:
        self._stop_thinking()
        self._assistant.finalize_turn()
        self._pending_gap_before_assistant = False
        self._tools.interrupt_running()

    def _update_thinking(self, content: str) -> None:
        if self._overlay_active:
            return
        if self._thinking_live is None:
            self._thinking_live = Live(
                Text(""),
                console=self._console,
                transient=True,
                refresh_per_second=12,
            )
            self._thinking_live.start()
        glyph = _THINKING_GLYPHS[self._thinking_frame % len(_THINKING_GLYPHS)]
        self._thinking_frame += 1
        self._thinking_live.update(
            Text.assemble(
                (f"{glyph} ", "dim"),
                (_truncate(content, 120), "dim italic"),
            )
        )

    def _stop_thinking(self) -> None:
        if self._thinking_live is None:
            return
        self._thinking_live.stop()
        self._thinking_live = None

    def handle_event(self, event: Any) -> tuple[bool, list[dict[str, Any]] | None]:
        if not isinstance(event, ThinkingEvent):
            self._stop_thinking()

        match event:
            case SessionInitEvent(session_id=_):
                pass
            case ThinkingEvent(content=thinking):
                self._assistant.flush_line_for_external_event()
                self._update_thinking(thinking)
                self._pending_gap_before_assistant = False
            case PreCompactEvent(current_tokens=_, threshold=_, trigger=_):
                pass
            case ToolCallEvent(tool=tool_name, args=arguments, tool_call_id=tool_call_id):
                self._assistant.flush_line_for_external_event()
                args_dict = arguments if isinstance(arguments, dict) else {"_raw": str(arguments)}
                self._tools.render_call(tool_name, args_dict, tool_call_id)
                self._todo.maybe_render_update(tool_name, args_dict)
                self._pending_gap_before_assistant = True
            case ToolResultEvent(tool=tool_name, result=result, tool_call_id=tool_call_id, is_error=is_error):
                self._assistant.flush_line_for_external_event()
                self._tools.render_result(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result=result,
                    is_error=is_error,
                )
                self._pending_gap_before_assistant = True
            case UsageDeltaEvent(source=_, model=_, level=_, delta_prompt_tokens=_, delta_prompt_cached_tokens=_, delta_completion_tokens=_, delta_total_tokens=_):
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
                self._tools.update_task_progress(
                    tool_call_id=tool_call_id,
                    tokens=tokens,
                    elapsed_ms=elapsed_ms,
                )
            case SubagentStopEvent(tool_call_id=_, subagent_name=_, status=_, duration_ms=_, error=_):
                pass
            case UserQuestionEvent(questions=questions, tool_call_id=_):
                self._assistant.finalize_turn()
                self._pending_gap_before_assistant = False
                return (True, questions)
            case TextEvent(content=text):
                if self._pending_gap_before_assistant:
                    self._assistant.insert_gap_before_next_segment()
                    self._pending_gap_before_assistant = False
                self._assistant.append_text(text)
            case StopEvent(reason=reason):
                if reason == "waiting_for_input":
                    self._assistant.finalize_turn()
                    self._pending_gap_before_assistant = False
                    return (True, None)
                self._assistant.finalize_turn()
                self._pending_gap_before_assistant = False
        return (False, None)
