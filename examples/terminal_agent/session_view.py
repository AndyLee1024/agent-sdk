from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown

from comate_agent_sdk.agent import ChatSession
from comate_agent_sdk.context.items import ItemType

from terminal_agent.message_style import print_assistant_gap, print_assistant_prefix_line


def _truncate(content: str, max_len: int = 1400) -> str:
    if len(content) <= max_len:
        return content
    return f"{content[:max_len]}..."


def _extract_assistant_text(item) -> str:
    """从 ContextItem 提取用于显示的文本（不含 tool_calls JSON）"""
    message = getattr(item, "message", None)
    if message is None:
        return ""

    # 优先使用 message.text（纯文本，不含 tool_calls）
    if hasattr(message, "text"):
        text = message.text
        if isinstance(text, str):
            return text

    # 回退：处理非标准 content
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


def _split_first_visible_line(content: str) -> tuple[str, str]:
    lines = content.splitlines()
    if not lines:
        return "", ""

    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx >= len(lines):
        return "", ""

    first = lines[idx].strip()
    remainder = "\n".join(lines[idx + 1 :]).lstrip("\r\n")
    return first, remainder


def render_session_header(console: Console, session_id: str, mode: str) -> None:
    console.print(f"[dim]session({mode}): [cyan]{session_id}[/][/]")


def render_user_message(console: Console, content: str) -> None:
    console.print(f"[green]>[/] {_truncate(content, 1000)}")


def render_resume_timeline(console: Console, session: ChatSession) -> None:
    items = session._agent._context.get_conversation_items_snapshot()
    history = [
        item
        for item in items
        if item.item_type in (ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE)
    ]
    if not history:
        console.print("[dim]no previous messages[/]")
        return

    console.print(f"[dim]history loaded: {len(history)} messages[/]")

    for item in history:
        if item.item_type == ItemType.USER_MESSAGE:
            content = item.content_text or ""
            render_user_message(console, str(content))
            continue
        assistant_text = _extract_assistant_text(item).strip()
        if not assistant_text:
            console.print("[dim]⏺ (tool call only message)[/]")
            continue
        trimmed = _truncate(assistant_text)
        first, remainder = _split_first_visible_line(trimmed)
        if not first:
            console.print("[dim]⏺ (tool call only message)[/]")
            continue
        print_assistant_prefix_line(console, first)
        if remainder:
            console.print(Markdown(remainder, code_theme="monokai", hyperlinks=True))
        print_assistant_gap(console)
