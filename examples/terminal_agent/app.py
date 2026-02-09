from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import suppress
from typing import Any

import questionary
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig, ChatSession
from comate_agent_sdk.tools import tool

from terminal_agent.animations import AnimationPhase, StreamAnimationController, SubmissionAnimator
from terminal_agent.event_renderer import EventRenderer
from terminal_agent.session_view import (
    render_resume_timeline,
    render_session_header,
)

console = Console()
animation_console = console
logging.getLogger("comate_agent_sdk.system_tools.tools").setLevel(logging.ERROR)

pt_style = PTStyle.from_dict(
    {
        "prompt": "bold ansicyan",
    }
)
prompt_session = PromptSession(history=InMemoryHistory(), style=pt_style)


@tool("Add two numbers 涉及到加法运算 必须使用这个工具")
async def add(a: int, b: int) -> int:
    return a + b


def _build_agent() -> Agent:
    return Agent(
        config=AgentConfig(
            mcp_servers={
                "exa_search": {
                    "type": "http",
                    "url": "https://mcp.exa.ai/mcp?exaApiKey=2ac4b289-8f68-473b-8cfd-3f8cb11595b7",
                }
            }
        )
    )


def _help_text() -> None:
    console.print("[dim]/help  /session  /usage  /context  /exit[/]")


def _format_answers_as_text(answers: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for answer in answers:
        header = answer.get("header", "问题")
        selected = answer.get("selected", [])
        if not selected:
            continue
        if len(selected) == 1 and str(selected[0]).startswith("__other__:"):
            custom_text = str(selected[0]).replace("__other__:", "", 1)
            parts.append(f"关于「{header}」: 我的回答是「{custom_text}」")
            continue
        options_text = "、".join(str(item) for item in selected)
        parts.append(f"关于「{header}」: 我选择了 {options_text}")
    if not parts:
        return "我暂时没有可提交的选择。"
    return f"{'；'.join(parts)}。"


async def _interactive_question_dialog(questions: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        total = len(questions)
        answers: list[dict[str, Any]] = []
        for idx, question in enumerate(questions, 1):
            header = question.get("header", "问题")
            question_text = question.get("question", "")
            options = question.get("options", [])
            multi_select = question.get("multiSelect", False)

            choices: list[Choice] = []
            for opt in options:
                label = opt.get("label", "")
                description = opt.get("description", "")
                display = f"{label} - {description}" if description else f"{label}"
                choices.append(Choice(title=display, value=label))
            choices.append(Choice(title="Other (自定义输入)", value="__other__"))

            prompt_text = f"[{idx}/{total}] {question_text}"
            if multi_select:
                selected = await questionary.checkbox(prompt_text, choices=choices).ask_async()
            else:
                selected = await questionary.select(prompt_text, choices=choices).ask_async()

            if selected is None:
                return {"action": "reject"}
            selected_list = selected if isinstance(selected, list) else [selected]
            processed_values: list[str] = []
            for value in selected_list:
                if value == "__other__":
                    custom_text = await questionary.text(
                        f"请输入您的自定义回答 (关于「{header}」):"
                    ).ask_async()
                    if custom_text is None:
                        return {"action": "reject"}
                    if custom_text:
                        processed_values.append(f"__other__:{custom_text}")
                    continue
                processed_values.append(str(value))

            answers.append({"header": header, "selected": processed_values})

        table = Table(title="📋 您的回答", show_header=True, header_style="bold cyan")
        table.add_column("问题", style="dim", width=24)
        table.add_column("回答", min_width=60)
        for answer in answers:
            header = answer.get("header", "")
            selected = answer.get("selected", [])
            converted = [
                value.replace("__other__:", "", 1) if str(value).startswith("__other__:") else str(value)
                for value in selected
            ]
            table.add_row(str(header), "、".join(converted))
        console.print(table)

        confirm = await questionary.select(
            "请选择操作:",
            choices=[
                Choice(title="✅ 提交回答 (Submit answers)", value="submit"),
                Choice(title="❌ 取消 (Cancel)", value="cancel"),
            ],
        ).ask_async()
        if confirm is None or confirm == "cancel":
            return {"action": "reject"}
        return {"action": "submit", "answers": answers}
    except KeyboardInterrupt:
        return {"action": "reject"}


async def _show_usage(session: ChatSession) -> None:
    usage = await session.get_usage()
    include_cost = bool(getattr(session._agent, "include_cost", False))
    prompt_new_tokens = max(usage.total_prompt_tokens - usage.total_prompt_cached_tokens, 0)

    table = Table(title="📊 Token Usage", show_header=True, header_style="bold cyan")
    table.add_column("指标", style="dim", width=20)
    table.add_column("数值", justify="right", width=24)
    table.add_row("总 Tokens", f"[bold]{usage.total_tokens:,}[/]")
    table.add_row("调用次数", f"{usage.entry_count}")
    table.add_row("Prompt Tokens", f"{usage.total_prompt_tokens:,}")
    table.add_row("Prompt Cached", f"{usage.total_prompt_cached_tokens:,}")
    table.add_row("Prompt New", f"{prompt_new_tokens:,}")
    table.add_row("Completion Tokens", f"{usage.total_completion_tokens:,}")
    if include_cost:
        table.add_row("Prompt Cost", f"[green]${usage.total_prompt_cost:.4f}[/]")
        table.add_row("Completion Cost", f"[green]${usage.total_completion_cost:.4f}[/]")
        table.add_row("总成本", f"[bold green]${usage.total_cost:.4f}[/]")
    else:
        table.add_row("成本模式", "[yellow]未开启(include_cost=False)[/]")
        table.add_row("总成本", "[dim]$0.0000[/]")
    console.print(table)

    if usage.by_model:
        model_table = Table(title="按模型统计", show_header=True, header_style="bold magenta")
        model_table.add_column("模型", style="cyan")
        model_table.add_column("Tokens", justify="right")
        model_table.add_column("成本", justify="right")
        for model, stats in usage.by_model.items():
            model_table.add_row(model, f"{stats.total_tokens:,}", f"${stats.cost:.4f}")
        console.print(model_table)

    if usage.by_level:
        level_table = Table(title="按档位统计", show_header=True, header_style="bold yellow")
        level_table.add_column("档位", style="cyan")
        level_table.add_column("Tokens", justify="right")
        level_table.add_column("成本", justify="right")
        for level, stats in usage.by_level.items():
            level_table.add_row(level, f"{stats.total_tokens:,}", f"${stats.cost:.4f}")
        console.print(level_table)


async def _show_context(session: ChatSession) -> None:
    from comate_agent_sdk.context.formatter import format_context_view

    info = await session.get_context_info()
    output = format_context_view(info)
    console.print(
        Panel(
            output,
            title="📦 Context Usage",
            border_style="blue",
        )
    )


async def _stream_message(
    session: ChatSession,
    text: str,
    renderer: EventRenderer,
    animator: SubmissionAnimator,
) -> tuple[bool, list[dict[str, Any]] | None]:
    waiting_for_input = False
    questions: list[dict[str, Any]] | None = None
    min_anim_seconds = 0.35
    progress_tick_interval_seconds = 1.0
    progress_tick_stop_event = asyncio.Event()

    async def _tick_progress() -> None:
        while not progress_tick_stop_event.is_set():
            if renderer.has_running_tasks():
                renderer.tick_progress()
            try:
                await asyncio.wait_for(
                    progress_tick_stop_event.wait(),
                    timeout=progress_tick_interval_seconds,
                )
            except TimeoutError:
                continue

    renderer.start_turn()
    animation_controller = StreamAnimationController(
        animator,
        min_visible_seconds=min_anim_seconds,
    )
    renderer.set_overlay_active(True)
    progress_tick_task = asyncio.create_task(_tick_progress(), name="progress-tick")
    await animation_controller.start()
    try:
        async for event in session.query_stream(text):
            await animation_controller.on_event(event)
            renderer.set_overlay_active(
                animation_controller.phase == AnimationPhase.SUBMITTING
            )
            is_waiting, new_questions = renderer.handle_event(event)
            if is_waiting:
                waiting_for_input = True
                if new_questions is not None:
                    questions = new_questions
    finally:
        renderer.set_overlay_active(False)
        progress_tick_stop_event.set()
        progress_tick_task.cancel()
        with suppress(asyncio.CancelledError):
            await progress_tick_task
        await animation_controller.shutdown()

    return waiting_for_input, questions


async def _stream_message_with_interrupt(
    session: ChatSession,
    text: str,
    renderer: EventRenderer,
    animator: SubmissionAnimator,
) -> tuple[bool, list[dict[str, Any]] | None] | None:
    stream_task = asyncio.create_task(
        _stream_message(session, text, renderer, animator),
        name="stream-message",
    )
    try:
        return await stream_task
    except KeyboardInterrupt:
        stream_task.cancel()
        with suppress(asyncio.CancelledError):
            await stream_task
        renderer.interrupt_turn()
        console.print("[yellow]⏹ 已中断当前任务，可继续输入新问题。[/]")
        return None


async def run() -> None:
    agent = _build_agent()
    session_id = sys.argv[1] if len(sys.argv) > 1 else None

    if session_id:
        console.print(f"[yellow]⏳ Resuming session: {session_id}[/]")
        session = ChatSession.resume(agent, session_id=session_id)
        mode = "resume"
    else:
        session = ChatSession(agent)
        mode = "new"

    render_session_header(console, session.session_id, mode=mode)
    if mode == "resume":
        render_resume_timeline(console, session)
    _help_text()

    renderer = EventRenderer(console)
    animator = SubmissionAnimator(animation_console)

    while True:
        try:
            user_input = await prompt_session.prompt_async([("class:prompt", "❯ ")])
        except (EOFError, KeyboardInterrupt):
            break

        text = user_input.strip()
        if not text:
            continue

        if text == "/help":
            _help_text()
            continue
        if text == "/exit":
            break
        if text == "/session":
            console.print(f"[bold]Session ID:[/] [cyan]{session.session_id}[/]")
            continue
        if text == "/usage":
            await _show_usage(session)
            continue
        if text == "/context":
            await _show_context(session)
            continue
        if text.startswith("/"):
            console.print(f"[red]Unknown command:[/] {text}")
            continue

        console.print()
        stream_result = await _stream_message_with_interrupt(
            session,
            text,
            renderer,
            animator,
        )
        if stream_result is None:
            continue
        waiting_for_input, questions = stream_result

        while waiting_for_input:
            if questions:
                dialog_result = await _interactive_question_dialog(questions)
                if dialog_result["action"] == "reject":
                    answer_text = "用户拒绝回答问题。"
                else:
                    answer_text = _format_answers_as_text(dialog_result["answers"])
            else:
                try:
                    answer_text = await prompt_session.prompt_async([("class:prompt", "❯ ")])
                except (EOFError, KeyboardInterrupt):
                    answer_text = ""
                answer_text = answer_text.strip()
                if not answer_text:
                    continue

            console.print()
            stream_result = await _stream_message_with_interrupt(
                session,
                answer_text,
                renderer,
                animator,
            )
            if stream_result is None:
                waiting_for_input = False
                questions = None
                break
            waiting_for_input, questions = stream_result

    await session.close()
    console.print(
        f"[dim]Goodbye. Resume with: [bold cyan]comate resume {session.session_id}[/][/]"
    )


if __name__ == "__main__":
    asyncio.run(run())
