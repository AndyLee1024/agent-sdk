from __future__ import annotations

from typing import Any

from rich.console import Group, Console, RenderableType
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
from terminal_agent.layout_coordinator import TerminalLayoutCoordinator
from terminal_agent.todo_view import TodoStateStore
from terminal_agent.tool_view import ToolEventView

_THINKING_GLYPHS: tuple[str, ...] = ("◐", "◓", "◑", "◒")


def _truncate(content: str, max_len: int = 300) -> str:
    if len(content) <= max_len:
        return content
    return f"{content[:max_len]}..."


class EventRenderer:
    """Convert SDK stream events into layered terminal UI output."""

    def __init__(
        self,
        console: Console,
        *,
        fancy_progress_effect: bool = True,
        target_fps: int = 8,
    ) -> None:
        self._assistant = AssistantStreamRenderer()
        self._tools = ToolEventView(
            fancy_progress_effect=fancy_progress_effect,
        )
        self._todo = TodoStateStore()
        self._layout = TerminalLayoutCoordinator(console, target_fps=target_fps)
        self._thinking_content: str | None = None
        self._thinking_frame = 0
        self._pending_gap_before_assistant = False
        self._overlay_active = False

    def start_turn(self) -> None:
        self._assistant.start_turn()
        self._tools.reset_turn()
        self._clear_thinking()
        self._pending_gap_before_assistant = False
        self._layout.start_turn()
        self._refresh_layout(force=True)

    def close(self) -> None:
        self._layout.close()

    def set_overlay_active(self, active: bool) -> None:
        self._overlay_active = active
        self._tools.set_live_suspended(active)
        if active:
            self._clear_thinking()
        self._layout.set_overlay_active(active)
        if not active:
            self._refresh_layout(force=True)

    def update_usage_tokens(
        self,
        total_tokens: int,
        source_totals: dict[str, int] | None = None,
    ) -> None:
        self._tools.update_usage_tokens(total_tokens, source_totals)
        self._refresh_layout()

    def get_running_subagent_source_prefixes(self) -> set[str]:
        return self._tools.running_subagent_source_prefixes()

    def set_task_source_baseline(self, tool_call_id: str, source_total_tokens: int) -> None:
        self._tools.set_task_source_baseline(
            tool_call_id=tool_call_id,
            source_total_tokens=source_total_tokens,
        )
        self._refresh_layout()

    def has_running_tasks(self) -> bool:
        return self._tools.has_running_tasks()

    def tick_progress(self) -> None:
        self._tools.tick_progress()
        self._refresh_layout()

    def interrupt_turn(self) -> None:
        self._clear_thinking()
        self._assistant.finalize_turn()
        self._pending_gap_before_assistant = False
        self._tools.interrupt_running()
        self._flush_tool_activities_into_message()
        self._refresh_layout(force=True)
        self._layout.stop_turn()

    def _update_thinking(self, content: str) -> None:
        if self._overlay_active:
            return
        self._thinking_content = content
        self._thinking_frame += 1

    def _clear_thinking(self) -> None:
        self._thinking_content = None

    def _thinking_renderable(self) -> RenderableType | None:
        if not self._thinking_content:
            return None
        glyph = _THINKING_GLYPHS[self._thinking_frame % len(_THINKING_GLYPHS)]
        return Text.assemble(
            (f"{glyph} ", "dim"),
            (_truncate(self._thinking_content, 120), "dim italic"),
        )

    def _loading_layer_renderable(self) -> RenderableType | None:
        thinking = self._thinking_renderable()
        tools = self._tools.renderable()
        if thinking is None and tools is None:
            return None
        if thinking is not None and tools is not None:
            return Group(thinking, Text(""), tools)
        return thinking if thinking is not None else tools

    def _flush_tool_activities_into_message(self) -> None:
        lines = self._tools.consume_activity_lines()
        if not lines:
            return
        self._assistant.append_external_lines(lines)

    def _refresh_layout(self, *, force: bool = False) -> None:
        self._layout.update_layers(
            loading=self._loading_layer_renderable(),
            message=self._assistant.renderable(),
            todo=self._todo.renderable(),
        )
        self._layout.refresh(force=force)

    def handle_event(self, event: Any) -> tuple[bool, list[dict[str, Any]] | None]:
        if not isinstance(event, ThinkingEvent):
            self._clear_thinking()

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
                self._flush_tool_activities_into_message()
                self._pending_gap_before_assistant = True
            case ToolResultEvent(tool=tool_name, result=result, tool_call_id=tool_call_id, is_error=is_error):
                self._assistant.flush_line_for_external_event()
                self._tools.render_result(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result=result,
                    is_error=is_error,
                )
                self._flush_tool_activities_into_message()
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
                self._refresh_layout(force=True)
                self._layout.stop_turn()
                return (True, questions)
            case TextEvent(content=text):
                if self._pending_gap_before_assistant:
                    self._assistant.insert_gap_before_next_segment()
                    self._pending_gap_before_assistant = False
                self._assistant.append_text(text)
            case StopEvent(reason=reason):
                self._assistant.finalize_turn()
                self._pending_gap_before_assistant = False
                self._refresh_layout(force=True)
                self._layout.stop_turn()
                if reason == "waiting_for_input":
                    return (True, None)

        self._refresh_layout()
        return (False, None)
