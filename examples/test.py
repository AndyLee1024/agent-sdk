from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig, ToolResultEvent, PreCompactEvent, ThinkingEvent, SessionInitEvent, ChatSession, TextEvent, StopEvent, ToolCallEvent, UserQuestionEvent
from comate_agent_sdk.llm import ChatOpenAI
from comate_agent_sdk.tools import tool
import asyncio
import sys
import json

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from rich.style import Style
from rich.table import Table

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
import logging
import questionary
from questionary import Choice

console = Console()
logger = logging.getLogger(__name__)

# prompt_toolkit 样式
pt_style = PTStyle.from_dict({
    'prompt': 'bold ansigreen',
})
prompt_session = PromptSession(history=InMemoryHistory(), style=pt_style)

@tool("Add two numbers 涉及到加法运算 必须使用这个工具")
async def add(a: int, b: int) -> int:
    return a + b

agent = Agent(
    config=AgentConfig(
        mcp_servers={
            "exa_search": {
                 "type": "http",
                "url": "https://mcp.exa.ai/mcp?exaApiKey=2ac4b289-8f68-473b-8cfd-3f8cb11595b7"
            }
        }                     
    ),
)


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."




def _format_answers_as_text(answers: list[dict]) -> str:
    """Format structured answers as natural language text.

    Args:
        answers: List of {"header": str, "selected": list[str]} dicts

    Returns:
        Formatted text like: 关于「Auth method」: 我选择了 JWT tokens；关于「Test types」: 我选择了 Unit tests、Integration tests。
    """
    parts = []
    for ans in answers:
        header = ans.get("header", "问题")
        selected = ans.get("selected", [])

        if not selected:
            continue

        # Check if it's a custom "Other" answer
        if len(selected) == 1 and selected[0].startswith("__other__:"):
            custom_text = selected[0].replace("__other__:", "", 1)
            parts.append(f"关于「{header}」: 我的回答是「{custom_text}」")
        else:
            options_text = "、".join(selected)
            parts.append(f"关于「{header}」: 我选择了 {options_text}")

    return "；".join(parts) + "。"


async def _interactive_question_dialog(questions: list[dict]) -> dict:
    """Show an interactive inline dialog for answering questions using questionary.

    Returns:
        {"action": "submit", "answers": [...]} or {"action": "reject"}
    """
    try:
        total = len(questions)
        answers = []

        # 逐个问题回答
        for idx, q in enumerate(questions, 1):
            header = q.get("header", "问题")
            question_text = q.get("question", "")
            options = q.get("options", [])
            multi_select = q.get("multiSelect", False)

            # 构建选项列表
            choices = []
            for opt in options:
                label = opt.get("label", "")
                desc = opt.get("description", "")
                display_text = f"{label} - {desc}" if desc else label
                choices.append(Choice(title=display_text, value=label))

            # 添加 Other 选项
            choices.append(Choice(title="Other (自定义输入)", value="__other__"))

            # 提示文本
            prompt_text = f"[{idx}/{total}] {question_text}"

            # 显示问题
            if multi_select:
                selected = await questionary.checkbox(
                    prompt_text,
                    choices=choices
                ).ask_async()
            else:
                selected = await questionary.select(
                    prompt_text,
                    choices=choices
                ).ask_async()

            # 处理取消 (Ctrl+C)
            if selected is None:
                return {"action": "reject"}

            # 转换为列表格式 (单选也统一为列表)
            selected_list = selected if isinstance(selected, list) else [selected]

            # 处理 Other 选项
            processed_values = []
            for val in selected_list:
                if val == "__other__":
                    custom_text = await questionary.text(
                        f"请输入您的自定义回答 (关于「{header}」):"
                    ).ask_async()

                    if custom_text is None:  # Ctrl+C
                        return {"action": "reject"}

                    if custom_text:
                        processed_values.append(f"__other__:{custom_text}")
                else:
                    processed_values.append(val)

            answers.append({
                "header": header,
                "selected": processed_values,
            })

        # Review 阶段 - 使用 rich.Table 展示
        console.print("\n[bold yellow]📋 您的回答:[/]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("问题", style="dim", width=20)
        table.add_column("回答", width=60)

        for ans in answers:
            header = ans.get("header", "")
            selected = ans.get("selected", [])

            # 格式化选项
            formatted_selected = []
            for s in selected:
                if s.startswith("__other__:"):
                    formatted_selected.append(s.replace("__other__:", "", 1))
                else:
                    formatted_selected.append(s)

            answer_text = "、".join(formatted_selected)
            table.add_row(header, answer_text)

        console.print(table)

        # 确认提交
        confirm = await questionary.select(
            "\n请选择操作:",
            choices=[
                Choice(title="✅ 提交回答 (Submit answers)", value="submit"),
                Choice(title="❌ 取消 (Cancel)", value="cancel"),
            ]
        ).ask_async()

        if confirm is None or confirm == "cancel":
            return {"action": "reject"}

        return {"action": "submit", "answers": answers}

    except KeyboardInterrupt:
        # Ctrl+C 统一处理为 reject
        return {"action": "reject"}


def _log_event(event) -> tuple[bool, list[dict] | None]:
    """Log agent events to console with rich formatting.
    
    Returns:
        (is_waiting_for_input, questions) - whether agent is waiting for user input and the questions if any
    """
    match event:
        case SessionInitEvent(session_id=se):
            console.print(Panel(
                f"[bold green]Session ID:[/] {se}",
                title="🚀 Session Started",
                border_style="green"
            ))
        case ThinkingEvent(content=thinking):
            preview = _truncate(thinking, 300)
            console.print(f"\n[dim italic]💭 {preview}[/]")
        case PreCompactEvent(current_tokens=t, threshold=th, trigger=trig):
            console.print(f"[yellow]📦 压缩前: {t} tokens (阈值: {th})[/]")
        case ToolResultEvent(tool=tool_name, result=result, tool_call_id=tcid, is_error=is_error):
            status = "[red]❌ Error[/]" if is_error else "[green]✅ Success[/]"
            preview = _truncate(result, 300)
            console.print(Panel(
                f"{status}\n[dim]{preview}[/]",
                title=f"📤 Tool Result: [cyan]{tool_name}[/]",
                border_style="blue" if not is_error else "red",
                subtitle=f"[dim]{tcid[:8]}...[/]" if len(tcid) > 8 else f"[dim]{tcid}[/]"
            ))
        case ToolCallEvent(tool=tool_name, args=arguments, tool_call_id=tcid):
            try:
                args_str = json.dumps(arguments, ensure_ascii=False, indent=2)
                args_preview = _truncate(args_str, 200)
            except:
                args_preview = str(arguments)[:200]
            console.print(Panel(
                f"[dim]{args_preview}[/]",
                title=f"🔧 Tool Call: [bold magenta]{tool_name}[/]",
                border_style="magenta",
                subtitle=f"[dim]{tcid[:8]}...[/]" if len(tcid) > 8 else f"[dim]{tcid}[/]"
            ))
        case UserQuestionEvent(questions=questions, tool_call_id=tcid):
            console.print("[yellow]正在准备问题...[/]")
            return (True, questions)
        case TextEvent(content=text):
            console.print(text, end="", style="bright_white")
        case StopEvent(reason=reason):
            if reason == "waiting_for_input":
                console.print(f"\n[yellow]── 等待用户输入 ──[/]\n")
                return (True, None)
            else:
                console.print(f"\n[dim]── Session ended: {reason} ──[/]\n")
    
    return (False, None)


def _help_text() -> None:
    """Display help with rich formatting."""
    console.print(Panel(
        "[bold cyan]/help[/]      Show this help\n"
        "[bold cyan]/session[/]   Show current session id\n"
        "[bold cyan]/exit[/]      Exit\n\n"
        "[dim]Type any message to chat with the agent.[/]",
        title="📖 Commands",
        border_style="cyan"
    ))


async def main():
    # 支持通过命令行参数恢复会话: python test.py [session_id]
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if session_id:
        # 恢复已有会话 (resume)
        console.print(f"[yellow]⏳ Resuming session: {session_id}[/]")
        session = ChatSession.resume(agent, session_id=session_id)
    else:
        # 创建新会话
        session = ChatSession(agent)
    
    console.print(Panel.fit(
        "[bold]Chat Session REPL[/]\n"
        "[dim]Powered by comate-agent-sdk[/]",
        title="🤖 Agent",
        border_style="bright_blue"
    ))
    _help_text()
    
    # 持续对话循环
    while True:
        try:
            console.print()
            user_input = await prompt_session.prompt_async([('class:prompt', '> ')])
        except (EOFError, KeyboardInterrupt):
            break
        
        text = user_input.strip()
        if not text:
            continue
        
        # 处理命令
        if text == "/help":
            _help_text()
            continue
        if text == "/exit":
            break
        if text == "/session":
            console.print(f"[bold]Session ID:[/] [cyan]{session.session_id}[/]")
            continue
        if text.startswith("/"):
            console.print(f"[red]Unknown command:[/] {text}")
            continue
        
        # 发送消息并流式处理事件
        waiting_for_input = False
        questions = None
        async for event in session.query_stream(text):
            is_waiting, new_questions = _log_event(event)
            if is_waiting:
                waiting_for_input = True
                if new_questions is not None:  # 防止 StopEvent 覆盖 UserQuestionEvent 的 questions
                    questions = new_questions

        # 如果 Agent 在等待用户输入，处理 AskUserQuestion 或普通输入
        while waiting_for_input:
            if questions:
                # AskUserQuestion - 使用交互式 Dialog
                result = await _interactive_question_dialog(questions)
                if result["action"] == "reject":
                    answer_text = "用户拒绝回答问题。"
                else:
                    answer_text = _format_answers_as_text(result["answers"])
                console.print(f"\n[dim]已回答: {answer_text}[/]\n")
            else:
                # 普通 waiting_for_input (非 AskUserQuestion)
                try:
                    answer_text = await prompt_session.prompt_async([('class:prompt', '📝 ')])
                except (EOFError, KeyboardInterrupt):
                    break
                answer_text = answer_text.strip()
                if not answer_text:
                    continue

            # 发送回答并继续
            waiting_for_input = False
            questions = None
            async for event in session.query_stream(answer_text):
                is_waiting, new_questions = _log_event(event)
                if is_waiting:
                    waiting_for_input = True
                    if new_questions is not None:  # 防止 StopEvent 覆盖 UserQuestionEvent 的 questions
                        questions = new_questions
    
    await session.close()
    console.print("[dim]👋 Goodbye![/]")


if __name__ == "__main__":
    asyncio.run(main())

