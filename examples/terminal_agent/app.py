from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.completion import (
    Completer,
    Completion,
    FuzzyCompleter,
    WordCompleter,
    merge_completers,
)
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, has_focus
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea
from rich.console import Console
from rich.text import Text

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig, ChatSession
from comate_agent_sdk.context import EnvOptions
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.tools import tool

from terminal_agent.animations import StreamAnimationController, SubmissionAnimator
from terminal_agent.event_renderer import EventRenderer
from terminal_agent.logo import print_logo
from terminal_agent.markdown_render import render_markdown_to_plain
from terminal_agent.mention_completer import LocalFileMentionCompleter
from terminal_agent.models import HistoryEntry
from terminal_agent.question_view import AskUserQuestionUI, QuestionAction
from terminal_agent.rpc_stdio import StdioRPCBridge
from terminal_agent.status_bar import StatusBar

console = Console()
logger = logging.getLogger(__name__)
logging.getLogger("comate_agent_sdk.system_tools.tools").setLevel(logging.ERROR)

@dataclass(frozen=True, slots=True)
class SlashCommandSpec:
    name: str
    description: str
    aliases: tuple[str, ...] = ()

    def slash_name(self) -> str:
        if self.aliases:
            return f"/{self.name} ({', '.join(self.aliases)})"
        return f"/{self.name}"


@dataclass(frozen=True, slots=True)
class _SlashCommandCall:
    name: str
    args: str
    raw_input: str


def _parse_slash_command_call(user_input: str) -> _SlashCommandCall | None:
    text = user_input.strip()
    if not text or not text.startswith("/"):
        return None

    match = re.match(r"^\/([a-zA-Z0-9_-]+(?::[a-zA-Z0-9_-]+)*)", text)
    if match is None:
        return None
    if len(text) > match.end() and not text[match.end()].isspace():
        return None

    return _SlashCommandCall(
        name=match.group(1),
        args=text[match.end() :].lstrip(),
        raw_input=text,
    )


SLASH_COMMAND_SPECS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec(name="help", description="Show available slash commands", aliases=("h",)),
    SlashCommandSpec(name="session", description="Show current session ID"),
    SlashCommandSpec(name="usage", description="Show token usage summary"),
    SlashCommandSpec(name="context", description="Show context usage summary"),
    SlashCommandSpec(name="exit", description="Exit terminal agent", aliases=("quit",)),
)
SLASH_COMMANDS: tuple[str, ...] = tuple(f"/{cmd.name}" for cmd in SLASH_COMMAND_SPECS)


class _SlashCommandCompleter(Completer):
    def __init__(self, commands: tuple[SlashCommandSpec, ...]) -> None:
        self._commands = commands
        self._command_lookup: dict[str, list[SlashCommandSpec]] = {}
        words: list[str] = []

        for cmd in sorted(self._commands, key=lambda item: item.name):
            if cmd.name not in self._command_lookup:
                self._command_lookup[cmd.name] = []
                words.append(cmd.name)
            self._command_lookup[cmd.name].append(cmd)
            for alias in cmd.aliases:
                if alias in self._command_lookup:
                    self._command_lookup[alias].append(cmd)
                else:
                    self._command_lookup[alias] = [cmd]
                    words.append(alias)

        self._word_pattern = re.compile(r"[^\s]+")
        self._fuzzy_pattern = r"^[^\s]*"
        self._word_completer = WordCompleter(words, WORD=False, pattern=self._word_pattern)
        self._fuzzy = FuzzyCompleter(self._word_completer, WORD=False, pattern=self._fuzzy_pattern)

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if document.text_after_cursor.strip():
            return

        last_space = text.rfind(" ")
        token = text[last_space + 1 :]
        prefix = text[: last_space + 1] if last_space != -1 else ""
        if prefix:
            return
        if not token.startswith("/"):
            return

        typed = token[1:]
        if typed and typed in self._command_lookup:
            return

        typed_doc = Document(text=typed, cursor_position=len(typed))
        candidates = list(self._fuzzy.get_completions(typed_doc, complete_event))
        seen: set[str] = set()

        for candidate in candidates:
            commands = self._command_lookup.get(candidate.text)
            if not commands:
                continue
            for cmd in commands:
                if cmd.name in seen:
                    continue
                seen.add(cmd.name)
                yield Completion(
                    text=f"/{cmd.name}",
                    start_position=-len(token),
                    display=cmd.slash_name(),
                    display_meta=cmd.description,
                )


@tool("Add two numbers 涉及到加法运算 必须使用这个工具")
async def add(a: int, b: int) -> int:
    return a + b


def _build_agent() -> Agent:
    return Agent(
        config=AgentConfig(
            role="software_engineering",
            env_options=EnvOptions(system_env=True, git_env=True),
            mcp_servers={
                 "context7": {
                    "type": "http",
                    "url": "https://mcp.context7.com/mcp",
                    "headers": {
                        "CONTEXT7_API_KEY": "ctx7sk-74909a03-11b1-47f4-bb65-011804753c9d"
                    }
                },
                "exa_search": {
                    "type": "http",
                    "url": "https://mcp.exa.ai/mcp?exaApiKey=084b86e8-c227-4ef0-9f6d-e248594839f4&tools=web_search_exa,web_search_advanced_exa,get_code_context_exa,crawling_exa",
                }
            },
        )
    )


def _extract_assistant_text(item: Any) -> str:
    content = item.content_text or ""
    if content:
        return str(content)
    message = getattr(item, "message", None)
    if message is None:
        return ""
    msg_content = getattr(message, "content", "")
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        text_parts: list[str] = []
        for part in msg_content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "".join(text_parts)
    return ""


def _fit_single_line(content: str, width: int) -> str:
    normalized = max(width, 8)
    if len(content) <= normalized:
        return content
    if normalized <= 3:
        return content[:normalized]
    return f"{content[: normalized - 3]}..."


def _lerp_rgb(
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, ratio))
    red = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * clamped)
    green = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * clamped)
    blue = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * clamped)
    return red, green, blue


def _sweep_gradient_fragments(content: str, frame: int) -> list[tuple[str, str]]:
    if not content:
        return [("", " ")]

    base_rgb = (96, 124, 156)
    mid_rgb = (118, 195, 225)
    high_rgb = (218, 246, 255)

    total = len(content)
    window = max(4, total // 6)
    cycle = max(total + window * 2, 20)
    center = (frame % cycle) - window

    fragments: list[tuple[str, str]] = []
    for idx, char in enumerate(content):
        distance = abs(idx - center)
        if distance <= window:
            glow = 1.0 - (distance / window)
            if glow >= 0.6:
                red, green, blue = _lerp_rgb(mid_rgb, high_rgb, (glow - 0.6) / 0.4)
            else:
                red, green, blue = _lerp_rgb(base_rgb, mid_rgb, glow / 0.6)
        else:
            red, green, blue = base_rgb
        fragments.append((f"fg:#{red:02x}{green:02x}{blue:02x} bold", char))
    return fragments


class UIMode(Enum):
    NORMAL = "normal"
    QUESTION = "question"


class TerminalAgentTUI:
    def __init__(self, session: ChatSession, status_bar: StatusBar, renderer: EventRenderer) -> None:
        self._session = session
        self._status_bar = status_bar
        self._renderer = renderer

        self._busy = False
        self._waiting_for_input = False
        self._pending_questions: list[dict[str, Any]] | None = None
        self._ui_mode = UIMode.NORMAL
        self._input_read_only = Condition(lambda: self._busy)

        self._slash_completer = _SlashCommandCompleter(SLASH_COMMAND_SPECS)
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
            "session": self._slash_session,
            "usage": self._slash_usage,
            "context": self._slash_context,
            "exit": self._slash_exit,
        }
        self._loading_frame = 0

        # 初始化提交动画控制器
        self._animator = SubmissionAnimator()
        self._animation_controller = StreamAnimationController(self._animator)

        self._closing = False
        self._printed_history_index = 0
        self._render_dirty = True
        self._last_loading_line = ""

        self._app: Application[None] | None = None
        self._stream_task: asyncio.Task[Any] | None = None
        self._ui_tick_task: asyncio.Task[None] | None = None
        self._interrupt_requested_at: float | None = None
        self._interrupt_force_window_seconds = 1.5

        # 历史显示区域（可滚动）
        self._history_area = TextArea(
            text="",
            multiline=True,
            scrollbar=False,
            read_only=True,
            wrap_lines=True,
            style="class:history",
            focusable=False,
        )

        self._input_area = TextArea(
            text="",
            multiline=False,
            prompt="> ",
            wrap_lines=False,
            completer=self._input_completer,
            complete_while_typing=True,
            history=InMemoryHistory(),
            read_only=self._input_read_only,
            style="class:input.line",
        )
        @self._input_area.buffer.on_text_changed.add_handler
        def _trigger_completion(buffer) -> None:
            if self._busy:
                return
            if buffer.complete_while_typing():
                buffer.start_completion(select_first=False)

        self._question_ui = AskUserQuestionUI()
        self._loading_control = FormattedTextControl(text=self._loading_text)
        self._status_control = FormattedTextControl(text=self._status_text)

        self._loading_window = Window(
            content=self._loading_control,
            height=1,
            dont_extend_height=True,
            style="class:loading",
        )
        self._status_window = Window(
            content=self._status_control,
            height=1,
            dont_extend_height=True,
            style="class:status",
        )

        self._input_container = ConditionalContainer(
            content=self._input_area,
            filter=Condition(lambda: self._ui_mode == UIMode.NORMAL),
        )
        self._question_container = ConditionalContainer(
            content=self._question_ui.container,
            filter=Condition(lambda: self._ui_mode == UIMode.QUESTION),
        )

        self._main_container = HSplit(
            [
                self._history_area,
                self._loading_window,
                Window(height=1, char=" ", style="class:input.pad"),
                self._input_container,
                self._question_container,
                Window(height=1, char=" ", style="class:input.pad"),
                self._status_window,
            ]
        )

        self._root = FloatContainer(
            content=self._main_container,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=12, scroll_offset=1),
                ),
            ],
        )

        self._layout = Layout(self._root, focused_element=self._input_area.window)
        self._bindings = self._build_key_bindings()
        self._style = PTStyle.from_dict(
            {
                "": "bg:#1f232a #e5e9f0",
                "history": "bg:#1a1e24 #d8dee9",
                "input.pad": "bg:#3a3d42",
                "input.line": "bg:#3a3d42 #f2f4f8",
                "input-line": "bg:#3a3d42 #f2f4f8",
                "status": "bg:#2d3138 #c3ccd8",
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
                "completion-menu": "bg:#2d3441 #d8dee9",
                "completion-menu.completion.current": "bg:#5e81ac #eceff4",
                "completion-menu.meta.completion": "bg:#2d3441 #81a1c1",
                "completion-menu.meta.completion.current": "bg:#5e81ac #e5e9f0",
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

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        normal_mode = Condition(lambda: self._ui_mode == UIMode.NORMAL)
        question_mode = Condition(lambda: self._ui_mode == UIMode.QUESTION)

        @bindings.add("enter", filter=question_mode)
        def _question_enter(event) -> None:
            del event
            action = self._question_ui.handle_enter()
            self._handle_question_action(action)
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("up", filter=question_mode)
        def _question_up(event) -> None:
            del event
            self._question_ui.move_option(-1)
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("down", filter=question_mode)
        def _question_down(event) -> None:
            del event
            self._question_ui.move_option(1)
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("left", filter=question_mode)
        def _question_prev(event) -> None:
            del event
            self._question_ui.prev_question()
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("right", filter=question_mode)
        def _question_next(event) -> None:
            del event
            self._question_ui.next_question()
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add(
            "space",
            filter=question_mode & ~has_focus(self._question_ui.custom_input_window),
        )
        def _question_toggle(event) -> None:
            del event
            self._question_ui.toggle_current_selection()
            self._invalidate()

        @bindings.add("tab", filter=question_mode)
        def _question_submit_tab(event) -> None:
            del event
            self._question_ui.focus_submit()
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("escape", filter=question_mode)
        def _question_cancel(event) -> None:
            del event
            self._handle_question_action(self._question_ui.handle_escape())
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("enter", filter=normal_mode)
        def _submit(event) -> None:
            del event
            if self._accept_active_completion():
                return
            self._submit_from_input()

        @bindings.add("tab", filter=normal_mode)
        def _accept_candidate(event) -> None:
            buffer = event.current_buffer
            complete_state = buffer.complete_state
            if complete_state is not None and complete_state.completions:
                self._accept_active_completion()
                return
            buffer.start_completion(select_first=False)

        @bindings.add("s-tab", filter=normal_mode & has_completions)
        def _active_prev(event) -> None:
            event.current_buffer.complete_previous()

        @bindings.add("up", filter=normal_mode)
        def _active_prev_up(event) -> None:
            buffer = event.current_buffer
            if self._move_completion_selection(buffer, backward=True):
                return
            if self._should_handle_history():
                buffer.auto_up(count=1)

        @bindings.add("down", filter=normal_mode)
        def _active_next_down(event) -> None:
            buffer = event.current_buffer
            if self._move_completion_selection(buffer, backward=False):
                return
            if self._should_handle_history():
                buffer.auto_down(count=1)

        @bindings.add("escape", filter=normal_mode)
        def _clear_active_or_refocus(event) -> None:
            buffer = event.current_buffer
            if buffer.complete_state is not None:
                buffer.cancel_completion()
                return
            if self._app is not None:
                self._app.layout.focus(self._input_area.window)

        @bindings.add("c-c")
        def _interrupt_or_exit(event) -> None:
            del event
            if self._busy and self._stream_task is not None:
                now = time.monotonic()
                interrupted_once = (
                    self._interrupt_requested_at is not None
                    and (now - self._interrupt_requested_at) <= self._interrupt_force_window_seconds
                )

                if interrupted_once:
                    self._stream_task.cancel()
                    self._session.run_controller.clear()
                    self._interrupt_requested_at = None
                    self._renderer.interrupt_turn()
                    self._renderer.append_system_message("已强制中断当前任务，可继续输入。")
                    self._set_busy(False)
                    self._waiting_for_input = False
                    self._pending_questions = None
                    self._exit_question_mode()
                    self._refresh_layers()
                    return

                self._session.run_controller.interrupt(reason="user")
                self._interrupt_requested_at = now
                self._renderer.append_system_message("已发送中断信号。再次按 Ctrl+C 将强制中断。")
                self._refresh_layers()
                return
            self._exit_app()

        @bindings.add("c-d")
        def _exit(event) -> None:
            del event
            self._exit_app()

        return bindings

    def _should_handle_history(self) -> bool:
        """检查是否应该处理历史浏览(而不是补全导航)

        Returns:
            True 表示可以浏览历史,False 表示应该优先处理补全
        """
        buffer = self._input_area.buffer

        # 如果有活动的补全菜单,不应该浏览历史
        if buffer.complete_state is not None and buffer.complete_state.completions:
            return False

        # 如果在补全上下文中(输入了 / 或 @),不应该浏览历史
        document = buffer.document
        if self._completion_context_active(document.text_before_cursor, document.text_after_cursor):
            return False

        return True

    def _completion_context_active(self, text_before_cursor: str, text_after_cursor: str) -> bool:
        if text_after_cursor.strip():
            return False
        if text_before_cursor.startswith("/") and " " not in text_before_cursor:
            return True
        return self._mention_completer.extract_context(text_before_cursor) is not None

    def _move_completion_selection(self, buffer: Any, *, backward: bool) -> bool:
        """尝试在补全菜单中移动选择项

        行为:
        - 如果补全菜单已显示,在菜单中导航
        - 如果在补全上下文中(输入 / 或 @)但菜单未显示,触发补全并选中第一项
        - 否则返回 False,允许调用者处理其他行为(如历史浏览)

        Args:
            buffer: 当前的 Buffer 对象
            backward: True 表示向上导航,False 表示向下导航

        Returns:
            True 表示已处理补全导航,False 表示未处理
        """
        complete_state = buffer.complete_state

        # 情况1: 已有补全菜单且有补全项
        if complete_state is not None and complete_state.completions:
            if backward:
                buffer.complete_previous()
            else:
                buffer.complete_next()
            return True

        # 情况2: 在补全上下文中但菜单未显示
        document = buffer.document
        if self._completion_context_active(document.text_before_cursor, document.text_after_cursor):
            # 启动补全并选择第一项(关键改动:select_first=True)
            buffer.start_completion(select_first=True)
            return True

        return False

    def _enter_question_mode(self, questions: list[dict[str, Any]]) -> None:
        if not self._question_ui.set_questions(questions):
            return
        self._ui_mode = UIMode.QUESTION
        self._sync_focus_for_mode()
        self._refresh_layers()

    def _exit_question_mode(self) -> None:
        self._ui_mode = UIMode.NORMAL
        self._question_ui.clear()
        self._sync_focus_for_mode()

    def _sync_focus_for_mode(self) -> None:
        if self._app is None:
            return
        if self._ui_mode == UIMode.QUESTION:
            self._app.layout.focus(self._question_ui.focus_target())
            return
        self._app.layout.focus(self._input_area.window)

    def _handle_question_action(self, action: QuestionAction | None) -> None:
        if action is None:
            return
        if action.kind == "cancel":
            self._submit_question_reply(action.message, cancelled=True)
            return
        if action.kind == "submit":
            self._submit_question_reply(action.message, cancelled=False)

    def _submit_question_reply(self, message: str, *, cancelled: bool) -> None:
        normalized = message.strip()
        if not normalized:
            return
        self._exit_question_mode()
        self._waiting_for_input = False
        self._pending_questions = None
        if cancelled:
            self._renderer.append_system_message("用户取消问答，已发送拒绝回答消息。")
        self._refresh_layers()
        self._schedule_background(self._submit_user_message(normalized))

    def add_resume_history(self, mode: str) -> None:
        if mode != "resume":
            return

        items = self._session._agent._context.conversation.items
        history = [
            item
            for item in items
            if item.item_type in (ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE)
        ]
        if not history:
            return

        self._renderer.append_system_message(f"history loaded: {len(history)} messages")
        for item in history:
            if item.item_type == ItemType.USER_MESSAGE:
                content = str(item.content_text or "").strip()
                if content:
                    self._renderer.seed_user_message(content)
                continue

            assistant_text = _extract_assistant_text(item).strip()
            if assistant_text:
                self._renderer.append_assistant_message(assistant_text)
            else:
                self._renderer.append_system_message("(tool call only message)")

        self._drain_history_sync()

    def _accept_active_completion(self) -> bool:
        buffer = self._input_area.buffer
        complete_state = buffer.complete_state
        if complete_state is None or not complete_state.completions:
            return False

        completion = complete_state.current_completion
        if completion is None:
            completion = complete_state.completions[0]
        buffer.apply_completion(completion)
        return True

    def _submit_from_input(self) -> None:
        if self._busy:
            return
        if self._ui_mode != UIMode.NORMAL:
            return

        raw_text = self._input_area.text
        text = raw_text.strip()
        if not text:
            return

        if self._input_area.buffer.complete_state is not None:
            self._input_area.buffer.cancel_completion()
        self._input_area.buffer.document = Document("")
        if text.startswith("/"):
            self._schedule_background(self._execute_command(text))
            return

        self._schedule_background(self._submit_user_message(text))

    async def _submit_user_message(self, text: str) -> None:
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

        self._renderer.start_turn()
        self._renderer.seed_user_message(text)
        self._refresh_layers()

        # 启动提交动画
        await self._animation_controller.start()

        waiting_for_input = False
        questions: list[dict[str, Any]] | None = None

        stream_task = asyncio.create_task(self._consume_stream(text), name="terminal-tui-stream")
        self._stream_task = stream_task
        try:
            waiting_for_input, questions = await stream_task
        except asyncio.CancelledError:
            logger.debug("stream task cancelled")
        except Exception as exc:
            logger.exception("stream failed")
            self._renderer.append_system_message(f"流式处理失败: {exc}", is_error=True)
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

    async def _consume_stream(self, text: str) -> tuple[bool, list[dict[str, Any]] | None]:
        waiting_for_input = False
        questions: list[dict[str, Any]] | None = None

        async for event in self._session.query_stream(text):
            # 将事件传递给动画控制器以控制动画生命周期
            await self._animation_controller.on_event(event)
            is_waiting, new_questions = self._renderer.handle_event(event)
            if is_waiting:
                waiting_for_input = True
                if new_questions is not None:
                    questions = new_questions
            self._refresh_layers()

        self._renderer.finalize_turn()
        await self._animation_controller.shutdown()
        return waiting_for_input, questions

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

    async def _execute_command(self, command: str) -> None:
        parsed = _parse_slash_command_call(command)
        normalized = command.strip()
        if parsed is None:
            self._renderer.append_system_message(f"Unknown command: {normalized}", is_error=True)
            self._refresh_layers()
            return

        spec = self._slash_lookup.get(parsed.name)
        if spec is None:
            self._renderer.append_system_message(f"Unknown command: {normalized}", is_error=True)
            self._refresh_layers()
            return

        handler = self._slash_handlers.get(spec.name)
        if handler is None:
            logger.error("slash handler missing for command: %s", spec.name)
            self._renderer.append_system_message(
                f"Unknown command: {parsed.raw_input}",
                is_error=True,
            )
            self._refresh_layers()
            return

        result = handler(parsed.args)
        if asyncio.iscoroutine(result):
            await result
        self._refresh_layers()

    def _slash_help(self, _args: str) -> None:
        lines = []
        for spec in SLASH_COMMAND_SPECS:
            alias_text = ""
            if spec.aliases:
                alias_text = f" ({', '.join(f'/{alias}' for alias in spec.aliases)})"
            lines.append(f"/{spec.name}{alias_text} - {spec.description}")
        self._renderer.append_system_message("\n".join(lines))

    def _slash_session(self, _args: str) -> None:
        self._renderer.append_system_message(f"Session ID: {self._session.session_id}")

    async def _slash_usage(self, _args: str) -> None:
        await self._append_usage_snapshot()

    async def _slash_context(self, _args: str) -> None:
        await self._append_context_snapshot()

    def _slash_exit(self, _args: str) -> None:
        self._exit_app()

    async def _append_usage_snapshot(self) -> None:
        usage = await self._session.get_usage()
        include_cost = bool(getattr(self._session._agent, "include_cost", False))
        prompt_new_tokens = max(usage.total_prompt_tokens - usage.total_prompt_cached_tokens, 0)

        lines = [
            "Token Usage",
            f"- total: {usage.total_tokens:,}",
            f"- entries: {usage.entry_count}",
            f"- prompt: {usage.total_prompt_tokens:,}",
            f"- prompt_cached: {usage.total_prompt_cached_tokens:,}",
            f"- prompt_new: {prompt_new_tokens:,}",
            f"- completion: {usage.total_completion_tokens:,}",
        ]

        if include_cost:
            lines.extend(
                [
                    f"- prompt_cost: ${usage.total_prompt_cost:.4f}",
                    f"- completion_cost: ${usage.total_completion_cost:.4f}",
                    f"- total_cost: ${usage.total_cost:.4f}",
                ]
            )
        self._renderer.append_system_message("\n".join(lines))

    async def _append_context_snapshot(self) -> None:
        info = await self._session.get_context_info()
        utilization = float(getattr(info, "utilization_percent", 0.0))
        context_limit = getattr(info, "context_limit_tokens", None)
        used_tokens = getattr(info, "used_tokens", None)

        lines = [
            "Context Usage",
            f"- utilization: {utilization:.1f}%",
            f"- left: {max(0.0, 100.0 - utilization):.1f}%",
        ]
        if used_tokens is not None:
            lines.append(f"- used_tokens: {used_tokens}")
        if context_limit is not None:
            lines.append(f"- context_limit_tokens: {context_limit}")

        self._renderer.append_system_message("\n".join(lines))

    def _refresh_layers(self) -> None:
        self._sync_focus_for_mode()
        self._render_dirty = True

    async def _drain_history_async(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return
        await self._print_history_entries_async(pending)

    def _drain_history_sync(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return
        self._print_history_entries_sync(pending)

    def _pending_history_entries(self) -> list[HistoryEntry]:
        entries = self._renderer.history_entries()
        if self._printed_history_index >= len(entries):
            return []
        pending = entries[self._printed_history_index :]
        self._printed_history_index = len(entries)
        return pending

    @staticmethod
    def _entry_prefix(entry: HistoryEntry) -> str:
        if entry.entry_type == "user":
            return "❯"
        if entry.entry_type == "assistant":
            return "⏺"
        if entry.entry_type == "tool_call":
            return "→"
        if entry.entry_type == "tool_result":
            return "✗" if entry.is_error else "✓"
        return "•"

    def _entry_content(self, entry: HistoryEntry) -> str:
        if entry.entry_type != "assistant":
            return entry.text
        width = max(self._terminal_width() - 6, 40)
        return render_markdown_to_plain(entry.text, width=width)

    def _format_history_entries(self, entries: list[HistoryEntry]) -> str:
        """格式化历史条目为纯文本"""
        lines: list[str] = []
        for entry in entries:
            prefix = self._entry_prefix(entry)
            content = self._entry_content(entry)
            content_lines = content.splitlines() or [""]

            # 第一行带前缀
            lines.append(f"{prefix} {content_lines[0]}")

            # 后续行缩进
            for line in content_lines[1:]:
                lines.append(f"  {line}")

            # 条目之间空行
            lines.append("")

        return "\n".join(lines)

    async def _print_history_entries_async(self, entries: list[HistoryEntry]) -> None:
        if not entries:
            return

        # 分离普通文本 entries 和 Rich renderable
        for entry in entries:
            if hasattr(entry.text, "__rich_console__"):
                # Rich Text 对象直接打印（带前缀）
                prefix = self._entry_prefix(entry)
                prefixed = Text(f"{prefix} ", style="bold")
                prefixed.append_text(entry.text)
                await run_in_terminal(lambda t=prefixed: console.print(t))
            else:
                # 普通文本按原来的方式处理
                prefix = self._entry_prefix(entry)
                content = self._entry_content(entry)
                content_lines = content.splitlines() or [""]
                lines = [f"{prefix} {content_lines[0]}"]
                for line in content_lines[1:]:
                    lines.append(f"  {line}")
                await run_in_terminal(lambda l="\n".join(lines): console.print(l))

    def _print_history_entries_sync(self, entries: list[HistoryEntry]) -> None:
        if not entries:
            return

        # 分离普通文本 entries 和 Rich renderable
        for entry in entries:
            if hasattr(entry.text, "__rich_console__"):
                # Rich Text 对象直接打印（带前缀）
                prefix = self._entry_prefix(entry)
                prefixed = Text(f"{prefix} ", style="bold")
                prefixed.append_text(entry.text)
                console.print(prefixed)
            else:
                # 普通文本按原来的方式处理
                prefix = self._entry_prefix(entry)
                content = self._entry_content(entry)
                content_lines = content.splitlines() or [""]
                lines = [f"{prefix} {content_lines[0]}"]
                for line in content_lines[1:]:
                    lines.append(f"  {line}")
                console.print("\n".join(lines))

    async def _ui_tick(self) -> None:
        try:
            while not self._closing:
                self._renderer.tick_progress()
                self._loading_frame += 2
                await self._drain_history_async()

                # 检查动画器是否需要刷新
                if self._animator.consume_dirty():
                    self._render_dirty = True

                loading_line = self._renderer.loading_line().strip()
                loading_changed = loading_line != self._last_loading_line
                if loading_changed:
                    self._last_loading_line = loading_line
                    self._render_dirty = True

                if self._busy and (loading_line or self._animator.is_active):
                    self._render_dirty = True

                if self._render_dirty:
                    self._invalidate()
                    self._render_dirty = False
                await asyncio.sleep(1 / 12)
        except asyncio.CancelledError:
            return

    def _terminal_width(self) -> int:
        if self._app is None:
            return 100
        try:
            return max(int(self._app.output.get_size().columns), 40)
        except Exception:
            return 100

    def _status_text(self) -> list[tuple[str, str]]:
        width = self._terminal_width()
        base = self._status_bar.footer_status_text()
        mode = "idle"
        if self._busy:
            mode = "running"
        elif self._waiting_for_input:
            mode = "waiting_input"
        merged = f"[{mode}] {base}"
        return [("class:status", _fit_single_line(merged, width - 1))]

    def _loading_text(self) -> list[tuple[str, str]]:
        # 优先使用动画器的渲染（流光走字 + 随机 geek 术语）
        if self._animator.is_active:
            renderable = self._animator.renderable()
            return self._rich_text_to_pt_fragments(renderable)

        # 回退到原有的 loading_line 逻辑
        text = self._renderer.loading_line().strip()
        if not text:
            return [("", " ")]
        width = self._terminal_width()
        clipped = _fit_single_line(text, width - 1)
        return _sweep_gradient_fragments(clipped, frame=self._loading_frame)

    def _rich_text_to_pt_fragments(self, renderable) -> list[tuple[str, str]]:
        """将 Rich Text 转换为 prompt_toolkit 的 fragments 格式."""
        from rich.segment import Segment

        fragments: list[tuple[str, str]] = []
        for segment in renderable.__rich_console__(console, console.options):
            if isinstance(segment, Segment):
                text = segment.text
                style = segment.style
                if style:
                    # 将 Rich style 转换为 prompt_toolkit style
                    pt_style = self._rich_style_to_pt(style)
                    fragments.append((pt_style, text))
                else:
                    fragments.append(("", text))
        return fragments if fragments else [("", " ")]

    def _rich_style_to_pt(self, rich_style) -> str:
        """将 Rich style 转换为 prompt_toolkit style 字符串."""
        parts: list[str] = []
        if rich_style.bold:
            parts.append("bold")
        if rich_style.italic:
            parts.append("italic")
        if rich_style.color:
            color = rich_style.color
            if color.type.name == "TRUECOLOR":
                # triplet.hex 已经包含 # 前缀
                parts.append(color.triplet.hex)
            elif color.type.name == "STANDARD":
                parts.append(str(color.name).lower())
            elif color.type.name == "EIGHT_BIT":
                parts.append(f"ansibright{color.number}" if color.number >= 8 else f"ansi{color.number}")
        return " ".join(parts) if parts else ""

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
        self._ui_tick_task = asyncio.create_task(self._ui_tick(), name="terminal-ui-tick")
        try:
            await self._app.run_async()
        finally:
            self._closing = True
            if self._ui_tick_task is not None:
                self._ui_tick_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._ui_tick_task
            self._renderer.close()


def _resolve_session(agent: Agent, session_id: str | None) -> tuple[ChatSession, str]:
    if session_id:
        return ChatSession.resume(agent, session_id=session_id), "resume"
    return ChatSession(agent), "new"


async def run(*, rpc_stdio: bool = False, session_id: str | None = None) -> None:
    agent = _build_agent()
    session, mode = _resolve_session(agent, session_id)

    if rpc_stdio:
        bridge = StdioRPCBridge(session)
        try:
            await bridge.run()
        finally:
            await session.close()
        return

    print_logo(console)
    status_bar = StatusBar(session)
    if mode == "resume":
        await status_bar.refresh()

    renderer = EventRenderer()
    tui = TerminalAgentTUI(session, status_bar, renderer)
    tui.add_resume_history(mode)

    try:
        await tui.run()
    finally:
        await session.close()

    console.print(
        f"[dim]Goodbye. Resume with: [bold cyan]comate resume {session.session_id}[/][/]"
    )


if __name__ == "__main__":
    argv_session_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(session_id=argv_session_id))
