"""Terminal Agent TUI implementation.

This module was split out from `terminal_agent.app` to keep the entrypoint small.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.completion import (
    merge_completers,
)
from prompt_toolkit.filters import Condition, has_completions, has_focus
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout import FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.filters import Condition, has_completions, has_focus
from comate_agent_sdk.agent import ChatSession

from terminal_agent.animations import (
    DEFAULT_STATUS_PHRASES,
    StreamAnimationController,
    SubmissionAnimator,
)
from terminal_agent.env_utils import read_env_float, read_env_int
from terminal_agent.event_renderer import EventRenderer
from terminal_agent.mention_completer import LocalFileMentionCompleter
from terminal_agent.question_view import AskUserQuestionUI
from terminal_agent.selection_menu import (
    SelectionMenuUI,
)
from terminal_agent.slash_commands import (
    SLASH_COMMAND_SPECS,
    SlashCommandCompleter,
    SlashCommandSpec,
)
from terminal_agent.status_bar import StatusBar
from terminal_agent.tui_parts import (
    CommandsMixin,
    HistorySyncMixin,
    InputBehaviorMixin,
    KeyBindingsMixin,
    RenderPanelsMixin,
    UIMode,
)

logger = logging.getLogger(__name__)
logging.getLogger("comate_agent_sdk.system_tools.tools").setLevel(logging.ERROR)


class TerminalAgentTUI(
    KeyBindingsMixin,
    InputBehaviorMixin,
    CommandsMixin,
    HistorySyncMixin,
    RenderPanelsMixin,
):
    def __init__(
        self,
        session: ChatSession,
        status_bar: StatusBar,
        renderer: EventRenderer,
    ) -> None:
        self._session = session
        self._status_bar = status_bar
        self._renderer = renderer

        self._tool_panel_max_lines = read_env_int(
            "AGENT_SDK_TUI_TOOL_PANEL_MAX_LINES",
            4,
        )
        self._todo_panel_max_lines = read_env_int(
            "AGENT_SDK_TUI_TODO_PANEL_MAX_LINES",
            6,
        )

        self._busy = False
        self._waiting_for_input = False
        self._pending_questions: list[dict[str, Any]] | None = None
        self._ui_mode = UIMode.NORMAL
        self._show_thinking = False  # Ctrl+T 开关，默认关闭
        self._input_read_only = Condition(lambda: self._busy)

        self._slash_completer = SlashCommandCompleter(SLASH_COMMAND_SPECS)
        self._mention_completer = LocalFileMentionCompleter(Path.cwd())
        self._input_completer = merge_completers(
            [self._slash_completer, self._mention_completer],
            deduplicate=True,
        )
        self._slash_lookup: dict[str, SlashCommandSpec] = {}
        for spec in SLASH_COMMAND_SPECS:
            self._slash_lookup[spec.name] = spec
            for alias in spec.aliases:
                self._slash_lookup[alias] = spec
        self._slash_handlers: dict[str, Callable[[str], Any]] = {
            "help": self._slash_help,
            "model": self._slash_model,
            "session": self._slash_session,
            "usage": self._slash_usage,
            "context": self._slash_context,
            "exit": self._slash_exit,
        }
        self._loading_frame = 0
        self._fallback_loading_phrase = (
            random.choice(DEFAULT_STATUS_PHRASES)
            if DEFAULT_STATUS_PHRASES
            else "Thinking…"
        )
        self._fallback_phrase_refresh_at = 0.0

        self._tool_result_flash_seconds = read_env_float(
            "AGENT_SDK_TUI_TOOL_RESULT_FLASH_SECONDS",
            0.55,
        )
        self._tool_result_flash_gen = 0
        self._tool_result_flash_until_monotonic: float | None = None
        self._tool_result_flash_active = False

        # 初始化提交动画控制器
        self._animator = SubmissionAnimator()
        self._animation_controller = StreamAnimationController(self._animator)
        self._tool_result_animator = SubmissionAnimator()

        self._closing = False
        self._printed_history_index = 0
        self._render_dirty = True
        self._diff_panel_visible = False
        self._diff_panel_scroll = 0
        self._last_loading_line = ""

        self._app: Application[None] | None = None
        self._stream_task: asyncio.Task[Any] | None = None
        self._ui_tick_task: asyncio.Task[None] | None = None
        self._interrupt_requested_at: float | None = None
        self._interrupt_force_window_seconds = 1.5

        self._esc_last_pressed_at: float = 0.0
        self._esc_press_count: int = 0
        esc_window_ms = read_env_int("AGENT_SDK_TUI_ESC_CLEAR_WINDOW_MS", 700)
        self._esc_clear_window_seconds = esc_window_ms / 1000.0

        self._ctrl_c_last_pressed_at: float = 0.0
        self._ctrl_c_press_count: int = 0
        ctrl_c_window_ms = read_env_int("AGENT_SDK_TUI_CTRL_C_EXIT_WINDOW_MS", 700)
        self._ctrl_c_exit_window_seconds = ctrl_c_window_ms / 1000.0

        self._paste_threshold_chars = read_env_int(
            "AGENT_SDK_TUI_PASTE_PLACEHOLDER_THRESHOLD_CHARS",
            500,
        )
        self._paste_placeholder_text: str | None = None
        self._active_paste_token: str | None = None
        self._paste_payload_by_token: dict[str, str] = {}
        self._paste_token_seq = 0
        self._suppress_input_change_hook = False
        self._last_input_len = 0
        self._last_input_text = ""

        self._input_prompt_text = "> "
        self._input_prompt_width = max(1, get_cwidth(self._input_prompt_text))

        def _input_line_prefix(
            _line_number: int,
            wrap_count: int,
        ) -> list[tuple[str, str]]:
            if wrap_count <= 0:
                return [("class:input.prompt", self._input_prompt_text)]
            return [("class:input.prompt", " " * self._input_prompt_width)]

        self._input_area = TextArea(
            text="",
            multiline=True,
            prompt="",
            wrap_lines=True,
            dont_extend_height=True,
            completer=self._input_completer,
            complete_while_typing=False,  # 通过 Tab/上下键手动触发补全
            history=InMemoryHistory(),
            read_only=self._input_read_only,
            style="class:input.line",
            get_line_prefix=_input_line_prefix,
        )
        # Fill the entire input area with styled spaces to avoid VT100
        # erase-to-end-of-line resetting the background to the terminal default.
        # (prompt_toolkit renderer resets attributes before erase_end_of_line.)
        self._input_area.window.char = " "

        @self._input_area.buffer.on_text_changed.add_handler
        def _trigger_completion(_buffer) -> None:
            if self._handle_large_paste(_buffer):
                return
            if self._busy:
                return
            doc = self._input_area.buffer.document
            if self._completion_context_active(
                doc.text_before_cursor,
                doc.text_after_cursor,
            ):
                # 输入 / 或 @ 时自动弹出（不会选中第一项）
                self._input_area.buffer.start_completion(select_first=False)

        self._question_ui = AskUserQuestionUI()
        self._selection_ui = SelectionMenuUI()
        self._todo_control = FormattedTextControl(text=self._todo_text)
        self._loading_control = FormattedTextControl(text=self._loading_text)
        self._status_control = FormattedTextControl(text=self._status_text)

        self._todo_window = Window(
            content=self._todo_control,
            height=self._todo_height,
            dont_extend_height=True,
            style="class:loading",
        )
        self._todo_container = ConditionalContainer(
            content=self._todo_window,
            filter=Condition(lambda: self._renderer.has_active_todos()),
        )

        self._loading_window = Window(
            content=self._loading_control,
            height=self._loading_height,
            dont_extend_height=True,
            style="class:loading",
        )

        # Diff preview panel
        self._diff_panel_control = FormattedTextControl(text=self._diff_panel_text)
        self._diff_panel_window = Window(
            content=self._diff_panel_control,
            height=self._diff_panel_height,
            dont_extend_height=True,
            style="class:diff-panel",
            wrap_lines=False,
        )
        self._diff_panel_container = ConditionalContainer(
            content=self._diff_panel_window,
            filter=Condition(
                lambda: self._diff_panel_visible
                and self._renderer.latest_diff_lines is not None
            ),
        )

        self._status_window = Window(
            content=self._status_control,
            height=1,
            dont_extend_height=True,
            style="class:status",
        )
   
        # 补全菜单：在底部区域显示（覆盖 statusline）
        self._completion_menu = CompletionsMenu(
            max_height=8,
            scroll_offset=0,
            extra_filter=(
                Condition(lambda: self._ui_mode == UIMode.NORMAL)
                & has_focus(self._input_area)
            ),
        )

        self._completion_visible = (
            Condition(lambda: self._ui_mode == UIMode.NORMAL)
            & has_focus(self._input_area)
            & has_completions
        )
        self._bottom_container = ConditionalContainer(
            content=self._completion_menu,
            filter=self._completion_visible,
            alternative_content=self._status_window,
        )

        self._input_container = ConditionalContainer(
            content=self._input_area,
            filter=Condition(lambda: self._ui_mode == UIMode.NORMAL),
        )
        self._question_container = ConditionalContainer(
            content=self._question_ui.container,
            filter=Condition(lambda: self._ui_mode == UIMode.QUESTION),
        )
        self._selection_container = ConditionalContainer(
            content=self._selection_ui.container,
            filter=Condition(lambda: self._ui_mode == UIMode.SELECTION),
        )

        self._main_container = HSplit(
            [
                self._todo_container,
                self._loading_window,
                self._diff_panel_container,
                Window(height=1, char=" ", style="class:input.pad"),
                self._input_container,
                self._question_container,
                self._selection_container,
                Window(height=1, char=" ", style="class:input.pad"),
                self._bottom_container,
            ]
        )

        self._root = FloatContainer(
            content=self._main_container,
            floats=[],
        )

        self._layout = Layout(self._root, focused_element=self._input_area.window)
        self._bindings = self._build_key_bindings()
        self._style = PTStyle.from_dict(
            {
                "": "bg:#1f232a #e5e9f0",
                "history": "bg:#1a1e24 #d8dee9",
                "input.pad": "fg:default bg:default",
                "input.prompt": "bg:default #f2f4f8",
                "input.line": "bg:default #f2f4f8",
                "input-line": "bg:default #f2f4f8",
                "status": "bg:#2d3138 #c3ccd8",
                "git-diff.added": "#4ade80",
                "git-diff.removed": "#f87171",
                "question.tabs": "bg:#30353f #c7d2fe",
                "question.tabs.nav": "bg:#30353f #93c5fd",
                "question.tab": "bg:#374151 #cbd5e1",
                "question.tab.submit": "bg:#14532d #86efac bold",
                "question.tab.active": "bg:#1d4ed8 #eff6ff bold",
                "question.divider": "fg:#4b5563",
                "question.body": "bg:#252a33 #d8dee9",
                "question.title": "fg:#f8fafc bold",
                "question.hint": "fg:#94a3b8",
                "question.option": "fg:#dbeafe",
                "question.option.cursor": "bg:#1e3a8a #f8fafc bold",
                "question.option.selected": "fg:#93c5fd bold",
                "question.option.description": "fg:#9ca3af",
                "question.custom_input": "bg:#0f172a #f8fafc",
                "question.custom_input.border": "fg:#334155",
                "question.preview.title": "fg:#e2e8f0 bold",
                "question.preview.question": "fg:#bfdbfe",
                "question.preview.answer": "fg:#f1f5f9",
                "selection.title": "bg:#2d3138 #c3ccd8 bold",
                "selection.divider": "fg:#4b5563",
                "selection.body": "bg:#252a33 #d8dee9",
                "selection.option": "fg:#dbeafe",
                "selection.option.selected": "bg:#1e3a8a #f8fafc bold",
                "selection.description": "fg:#9ca3af",
                "selection.description.selected": "fg:#93c5fd",
                "selection.hint": "fg:#94a3b8",
                "completion-menu": "bg:#2d3441 #d8dee9",
                "completion-menu.completion.current": "bg:#5e81ac #eceff4",
                "completion-menu.meta.completion": "bg:#2d3441 #81a1c1",
                "completion-menu.meta.completion.current": "bg:#5e81ac #e5e9f0",
                "loading": "bg:#1a1e24",
            }
        )

        self._app = Application(
            layout=self._layout,
            key_bindings=self._bindings,
            style=self._style,
            full_screen=False,
            mouse_support=False,
        )

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        self._render_dirty = True
        self._invalidate()

    async def _submit_user_message(
        self,
        text: str,
        *,
        display_text: str | None = None,
    ) -> None:
        if self._busy:
            self._renderer.append_system_message("当前已有任务在运行，请稍候。", is_error=True)
            return

        if self._ui_mode == UIMode.QUESTION:
            self._exit_question_mode()

        self._session.run_controller.clear()
        self._interrupt_requested_at = None

        self._set_busy(True)
        self._waiting_for_input = False
        self._pending_questions = None
        # 本轮默认 loading 文案：避免出现 Working… 这种突兀 fallback
        self._fallback_loading_phrase = (
            random.choice(DEFAULT_STATUS_PHRASES)
            if DEFAULT_STATUS_PHRASES
            else "Thinking…"
        )
        self._fallback_phrase_refresh_at = time.monotonic() + 3.0  # 每 3 秒允许换一次

        self._renderer.start_turn()
        self._renderer.seed_user_message(
            display_text if display_text is not None else text
        )
        self._refresh_layers()

        # 启动提交动画
        await self._animation_controller.start()

        waiting_for_input = False
        questions: list[dict[str, Any]] | None = None

        stream_task = asyncio.create_task(
            self._consume_stream(text),
            name="terminal-tui-stream",
        )
        self._stream_task = stream_task
        try:
            waiting_for_input, questions = await stream_task
        except asyncio.CancelledError:
            logger.debug("stream task cancelled")
        except Exception as exc:
            logger.exception("stream failed")
            from terminal_agent.error_display import format_error

            error_msg, suggestion = format_error(exc)
            self._renderer.append_system_message(error_msg, is_error=True)
            if suggestion:
                self._renderer.append_system_message(f"💡 {suggestion}")

            # 清理 UI 状态
            self._renderer.interrupt_turn()
            await self._animation_controller.shutdown()
        finally:
            self._stream_task = None
            self._interrupt_requested_at = None
            self._set_busy(False)
            await self._status_bar.refresh()

        if waiting_for_input:
            self._waiting_for_input = True
            self._pending_questions = questions
            if questions:
                self._enter_question_mode(questions)
            else:
                self._renderer.append_system_message("请输入对上述问题的回答后回车提交。")
        else:
            self._waiting_for_input = False
            self._pending_questions = None
            self._exit_question_mode()

        self._refresh_layers()

    async def _consume_stream(
        self,
        text: str,
    ) -> tuple[bool, list[dict[str, Any]] | None]:
        waiting_for_input = False
        questions: list[dict[str, Any]] | None = None

        async for event in self._session.query_stream(text):
            self._maybe_cancel_tool_result_flash(event)
            # 将事件传递给动画控制器以控制动画生命周期
            await self._animation_controller.on_event(event)
            is_waiting, new_questions = self._renderer.handle_event(event)
            self._maybe_flash_tool_result(event)
            if is_waiting:
                waiting_for_input = True
                if new_questions is not None:
                    questions = new_questions
            self._refresh_layers()

        self._renderer.finalize_turn()
        await self._animation_controller.shutdown()
        return waiting_for_input, questions

    def _maybe_cancel_tool_result_flash(self, event: object) -> None:
        if not self._tool_result_flash_active:
            return

        from comate_agent_sdk.agent.events import StopEvent, TextEvent

        if isinstance(event, TextEvent) or isinstance(event, StopEvent):
            self._schedule_background(self._stop_tool_result_flash())

    def _maybe_flash_tool_result(self, event: object) -> None:
        from comate_agent_sdk.agent.events import ToolResultEvent

        if not isinstance(event, ToolResultEvent):
            return
        tool_name = str(event.tool or "").strip()
        if tool_name.lower() == "todowrite":
            return
        if self._renderer.has_running_tools():
            return
        self._trigger_tool_result_flash(
            tool_name=tool_name or "Tool",
            is_error=bool(event.is_error),
        )

    def _trigger_tool_result_flash(self, *, tool_name: str, is_error: bool) -> None:
        del is_error, tool_name
        phrase = (
            random.choice(DEFAULT_STATUS_PHRASES)
            if DEFAULT_STATUS_PHRASES
            else "Vibing..."
        )
        duration_seconds = max(0.15, float(self._tool_result_flash_seconds))
        self._tool_result_flash_gen += 1
        self._tool_result_flash_active = True
        self._tool_result_flash_until_monotonic = time.monotonic() + duration_seconds
        gen = self._tool_result_flash_gen
        self._schedule_background(self._start_tool_result_flash(gen=gen, hint=phrase))

    async def _start_tool_result_flash(self, *, gen: int, hint: str) -> None:
        if gen != self._tool_result_flash_gen:
            return
        await self._tool_result_animator.start()
        if gen != self._tool_result_flash_gen:
            return
        self._tool_result_animator.set_status_hint(hint)
        self._render_dirty = True
        self._invalidate()

    async def _stop_tool_result_flash(self) -> None:
        if not self._tool_result_flash_active:
            return
        self._tool_result_flash_gen += 1
        self._tool_result_flash_active = False
        self._tool_result_flash_until_monotonic = None
        self._tool_result_animator.set_status_hint(None)
        await self._tool_result_animator.stop()
        self._render_dirty = True
        self._invalidate()

    def _schedule_background(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)

        def _done(done_task: asyncio.Task[Any]) -> None:
            try:
                done_task.result()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("background task failed")

        task.add_done_callback(_done)

    def _refresh_layers(self) -> None:
        self._sync_focus_for_mode()
        self._render_dirty = True

    async def _ui_tick(self) -> None:
        try:
            while not self._closing:
                self._renderer.tick_progress()
                self._loading_frame += 2
                # busy 时每隔几秒换一个短语，让 loading 更“活”
                if self._busy and time.monotonic() >= self._fallback_phrase_refresh_at:
                    self._fallback_loading_phrase = (
                        random.choice(DEFAULT_STATUS_PHRASES)
                        if DEFAULT_STATUS_PHRASES
                        else "Thinking…"
                    )
                    self._fallback_phrase_refresh_at = time.monotonic() + 3.0
                    self._render_dirty = True

                await self._drain_history_async()

                # 检查动画器是否需要刷新
                if self._animator.consume_dirty():
                    self._render_dirty = True
                if self._tool_result_animator.consume_dirty():
                    self._render_dirty = True

                if self._tool_result_flash_active:
                    until = self._tool_result_flash_until_monotonic
                    if until is None or time.monotonic() >= until:
                        self._schedule_background(self._stop_tool_result_flash())
                    else:
                        self._render_dirty = True

                loading_line = self._renderer.loading_line().strip()
                loading_changed = loading_line != self._last_loading_line
                if loading_changed:
                    self._last_loading_line = loading_line
                    self._render_dirty = True

                if self._renderer.has_running_tools():
                    # Keep repainting for breathing animation.
                    self._render_dirty = True
                elif self._busy:
                    self._render_dirty = True


                if self._render_dirty:
                    self._invalidate()
                    self._render_dirty = False

                # 动态帧率：busy/动画时 12fps，idle 时 4fps
                fast = (
                    self._busy
                    or self._animator.is_active
                    or self._renderer.has_running_tools()
                )
                sleep_s = 1 / 12 if fast else 1 / 4
                await asyncio.sleep(sleep_s)
        except asyncio.CancelledError:
            return

    def _invalidate(self) -> None:
        if self._app is None:
            return
        self._app.invalidate()

    def _exit_app(self) -> None:
        self._closing = True
        self._session.run_controller.clear()
        if self._stream_task is not None:
            self._stream_task.cancel()
        if self._app is not None:
            self._app.exit(result=None)

    async def run(self) -> None:
        if self._app is None:
            return

        self._refresh_layers()
        self._ui_tick_task = asyncio.create_task(
            self._ui_tick(),
            name="terminal-ui-tick",
        )
        try:
            # Ensure scrollback prints don't mix with the inline (full_screen=False) UI.
            with patch_stdout(raw=True):
                await self._app.run_async()
        finally:
            self._closing = True
            if self._ui_tick_task is not None:
                self._ui_tick_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._ui_tick_task
            self._renderer.close()
