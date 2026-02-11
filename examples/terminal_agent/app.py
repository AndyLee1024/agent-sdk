from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import suppress
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea
from rich.console import Console
from rich.text import Text

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig, ChatSession
from comate_agent_sdk.context import EnvOptions
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.tools import tool

from terminal_agent.event_renderer import EventRenderer
from terminal_agent.logo import print_logo
from terminal_agent.models import HistoryEntry
from terminal_agent.status_bar import StatusBar

console = Console()
logger = logging.getLogger(__name__)
logging.getLogger("comate_agent_sdk.system_tools.tools").setLevel(logging.ERROR)

SLASH_COMMANDS: tuple[str, ...] = (
    "/help",
    "/session",
    "/usage",
    "/context",
    "/exit",
)


class _SlashCommandCompleter(Completer):
    """保留用于 slash 规则单测，实际 UI 不再使用 popup completion。"""

    def __init__(self, commands: tuple[str, ...]) -> None:
        self._commands = commands

    def get_completions(self, document: Document, complete_event):
        del complete_event
        text = document.text
        if not text.startswith("/"):
            return

        for command in self._commands:
            if command.startswith(text):
                yield Completion(
                    text=command,
                    start_position=-len(text),
                    display=command,
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


class TerminalAgentTUI:
    def __init__(self, session: ChatSession, status_bar: StatusBar, renderer: EventRenderer) -> None:
        self._session = session
        self._status_bar = status_bar
        self._renderer = renderer

        self._busy = False
        self._waiting_for_input = False
        self._pending_questions: list[dict[str, Any]] | None = None
        self._input_read_only = Condition(lambda: self._busy)

        self._slash_active = False
        self._slash_candidates: list[str] = []
        self._slash_index = 0
        self._loading_frame = 0

        self._closing = False
        self._printed_history_index = 0
        self._todo_cache: list[str] = []

        self._app: Application[None] | None = None
        self._stream_task: asyncio.Task[None] | None = None
        self._ui_tick_task: asyncio.Task[None] | None = None

        self._input_area = TextArea(
            text="",
            multiline=False,
            prompt="> ",
            wrap_lines=False,
            history=InMemoryHistory(),
            read_only=self._input_read_only,
            style="class:input.line",
        )
        self._input_area.buffer.on_text_changed += self._on_input_changed

        self._todo_control = FormattedTextControl(text=self._todo_text)
        self._loading_control = FormattedTextControl(text=self._loading_text)
        self._status_control = FormattedTextControl(text=self._status_text)

        self._todo_window = Window(
            content=self._todo_control,
            wrap_lines=False,
            dont_extend_height=True,
            style="class:todo",
        )
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

        self._todo_container = ConditionalContainer(
            content=self._todo_window,
            filter=Condition(lambda: bool(self._todo_cache)),
        )

        self._root = HSplit(
            [
                self._todo_container,
                self._loading_window,
                Window(height=1, char=" ", style="class:input.pad"),
                self._input_area,
                Window(height=1, char=" ", style="class:input.pad"),
                self._status_window,
            ]
        )

        self._layout = Layout(self._root, focused_element=self._input_area.window)
        self._bindings = self._build_key_bindings()
        self._style = PTStyle.from_dict(
            {
                "": "bg:#1f232a #e5e9f0",
                "todo": "bg:#232a35 #9ecbff",
                "input.pad": "bg:#3a3d42",
                "input.line": "bg:#3a3d42 #f2f4f8",
                "input-line": "bg:#3a3d42 #f2f4f8",
                "status": "bg:#2d3138 #c3ccd8",
            }
        )

        self._app = Application(
            layout=self._layout,
            key_bindings=self._bindings,
            style=self._style,
            full_screen=False,
            mouse_support=True,
        )

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        self._invalidate()

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def _submit(event) -> None:
            del event
            self._submit_from_input()

        @bindings.add("tab")
        def _slash_next(event) -> None:
            del event
            if not self._slash_active or not self._slash_candidates:
                return
            self._slash_index = (self._slash_index + 1) % len(self._slash_candidates)
            self._invalidate()

        @bindings.add("s-tab")
        def _slash_prev(event) -> None:
            del event
            if not self._slash_active or not self._slash_candidates:
                return
            self._slash_index = (self._slash_index - 1) % len(self._slash_candidates)
            self._invalidate()

        @bindings.add("up")
        def _slash_prev_up(event) -> None:
            del event
            if not self._slash_active or not self._slash_candidates:
                return
            self._slash_index = (self._slash_index - 1) % len(self._slash_candidates)
            self._invalidate()

        @bindings.add("down")
        def _slash_next_down(event) -> None:
            del event
            if not self._slash_active or not self._slash_candidates:
                return
            self._slash_index = (self._slash_index + 1) % len(self._slash_candidates)
            self._invalidate()

        @bindings.add("escape")
        def _clear_slash(event) -> None:
            del event
            if not self._slash_active:
                return
            self._deactivate_slash()

        @bindings.add("c-c")
        def _interrupt_or_exit(event) -> None:
            del event
            if self._busy and self._stream_task is not None:
                self._stream_task.cancel()
                self._renderer.interrupt_turn()
                self._renderer.append_system_message("已中断当前任务，可继续输入。")
                self._set_busy(False)
                self._waiting_for_input = False
                self._pending_questions = None
                self._refresh_layers()
                return
            self._exit_app()

        @bindings.add("c-d")
        def _exit(event) -> None:
            del event
            self._exit_app()

        return bindings

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

    def _on_input_changed(self, _event: Any) -> None:
        if self._busy:
            return

        text = self._input_area.text
        if not text.startswith("/"):
            if self._slash_active:
                self._deactivate_slash()
            return

        candidates = [command for command in SLASH_COMMANDS if command.startswith(text)]
        if not candidates:
            self._deactivate_slash()
            return

        previous = ""
        if self._slash_active and self._slash_candidates:
            previous = self._slash_candidates[self._slash_index]

        self._slash_active = True
        self._slash_candidates = candidates
        if previous and previous in candidates:
            self._slash_index = candidates.index(previous)
        else:
            self._slash_index = 0
        self._invalidate()

    def _deactivate_slash(self) -> None:
        self._slash_active = False
        self._slash_candidates = []
        self._slash_index = 0
        self._invalidate()

    def _submit_from_input(self) -> None:
        if self._busy:
            return

        raw_text = self._input_area.text
        text = raw_text.strip()
        if not text:
            return

        self._input_area.buffer.document = Document("")

        if self._slash_active and self._slash_candidates:
            command = self._slash_candidates[self._slash_index]
            self._deactivate_slash()
            self._schedule_background(self._execute_command(command))
            return

        self._deactivate_slash()
        if text.startswith("/"):
            self._schedule_background(self._execute_command(text))
            return

        self._schedule_background(self._submit_user_message(text))

    async def _submit_user_message(self, text: str) -> None:
        if self._busy:
            self._renderer.append_system_message("当前已有任务在运行，请稍候。", is_error=True)
            return

        self._set_busy(True)
        self._waiting_for_input = False
        self._pending_questions = None

        self._renderer.start_turn()
        self._renderer.seed_user_message(text)
        self._refresh_layers()

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
            self._set_busy(False)
            await self._status_bar.refresh()

        if waiting_for_input:
            self._waiting_for_input = True
            self._pending_questions = questions
            self._renderer.append_system_message("请输入对上述问题的回答后回车提交。")
        else:
            self._waiting_for_input = False
            self._pending_questions = None

        self._refresh_layers()

    async def _consume_stream(self, text: str) -> tuple[bool, list[dict[str, Any]] | None]:
        waiting_for_input = False
        questions: list[dict[str, Any]] | None = None

        async for event in self._session.query_stream(text):
            is_waiting, new_questions = self._renderer.handle_event(event)
            if is_waiting:
                waiting_for_input = True
                if new_questions is not None:
                    questions = new_questions
            self._refresh_layers()

        self._renderer.finalize_turn()
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
        normalized = command.strip()
        if normalized == "/help":
            self._renderer.append_system_message(" ".join(SLASH_COMMANDS))
            self._refresh_layers()
            return

        if normalized == "/session":
            self._renderer.append_system_message(f"Session ID: {self._session.session_id}")
            self._refresh_layers()
            return

        if normalized == "/usage":
            await self._append_usage_snapshot()
            self._refresh_layers()
            return

        if normalized == "/context":
            await self._append_context_snapshot()
            self._refresh_layers()
            return

        if normalized == "/exit":
            self._exit_app()
            return

        self._renderer.append_system_message(f"Unknown command: {normalized}", is_error=True)
        self._refresh_layers()

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
        self._todo_cache = self._renderer.todo_lines()
        self._invalidate()

    async def _drain_history_async(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return

        def _printer() -> None:
            self._print_history_entries(pending)

        await run_in_terminal(_printer, in_executor=False)

    def _drain_history_sync(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return
        self._print_history_entries(pending)

    def _pending_history_entries(self) -> list[HistoryEntry]:
        entries = self._renderer.history_entries()
        if self._printed_history_index >= len(entries):
            return []
        pending = entries[self._printed_history_index :]
        self._printed_history_index = len(entries)
        return pending

    @staticmethod
    def _entry_prefix(entry: HistoryEntry) -> tuple[str, str]:
        if entry.entry_type == "user":
            return "❯", "bold cyan"
        if entry.entry_type == "assistant":
            return "⏺", "bold bright_cyan"
        if entry.entry_type == "tool_call":
            return "→", "bold blue"
        if entry.entry_type == "tool_result":
            return ("✗", "bold red") if entry.is_error else ("✓", "bold green")
        return "•", "dim"

    def _print_history_entries(self, entries: list[HistoryEntry]) -> None:
        for entry in entries:
            prefix, prefix_style = self._entry_prefix(entry)
            content_lines = entry.text.splitlines() or [""]

            first_line = Text()
            first_line.append(f"{prefix} ", style=prefix_style)
            first_line.append(content_lines[0])
            console.print(first_line, soft_wrap=True)

            for line in content_lines[1:]:
                continuation = Text()
                continuation.append("  ")
                continuation.append(line)
                console.print(continuation, soft_wrap=True)

            console.print()

    async def _ui_tick(self) -> None:
        try:
            while not self._closing:
                self._renderer.tick_progress()
                self._loading_frame += 2
                self._todo_cache = self._renderer.todo_lines()
                await self._drain_history_async()
                self._invalidate()
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

        if self._slash_active and self._slash_candidates:
            fragments: list[str] = []
            for idx, command in enumerate(self._slash_candidates):
                if idx == self._slash_index:
                    fragments.append(f"[{command}]")
                else:
                    fragments.append(command)
            slash_line = f"slash> {'  '.join(fragments)}"
            return [("class:status", _fit_single_line(slash_line, width - 1))]

        base = self._status_bar.footer_status_text()
        mode = "idle"
        if self._busy:
            mode = "running"
        elif self._waiting_for_input:
            mode = "waiting_input"
        merged = f"[{mode}] {base}"
        return [("class:status", _fit_single_line(merged, width - 1))]

    def _todo_text(self) -> list[tuple[str, str]]:
        if not self._todo_cache:
            return [("class:todo", "")]
        limited = self._todo_cache[:6]
        return [("class:todo", "\n".join(limited))]

    def _loading_text(self) -> list[tuple[str, str]]:
        text = self._renderer.loading_line().strip()
        if not text and self._busy:
            text = "⏳ 正在处理..."
        if not text:
            return [("", " ")]
        width = self._terminal_width()
        clipped = _fit_single_line(text, width - 1)
        return _sweep_gradient_fragments(clipped, frame=self._loading_frame)

    def _invalidate(self) -> None:
        if self._app is None:
            return
        self._app.invalidate()

    def _exit_app(self) -> None:
        self._closing = True
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


async def run() -> None:
    print_logo(console)
    agent = _build_agent()
    session_id = sys.argv[1] if len(sys.argv) > 1 else None

    if session_id:
        session = ChatSession.resume(agent, session_id=session_id)
        mode = "resume"
    else:
        session = ChatSession(agent)
        mode = "new"

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
    asyncio.run(run())
