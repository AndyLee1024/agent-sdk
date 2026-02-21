from __future__ import annotations

import logging
import sys
from typing import Any

from prompt_toolkit.application import run_in_terminal
from rich.console import Console

from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import AssistantMessage

from terminal_agent.history_printer import (
    print_history_group_sync,
    render_history_group,
)
from terminal_agent.logo import print_logo
from terminal_agent.markdown_render import render_markdown_to_plain
from terminal_agent.models import HistoryEntry

console = Console()
logger = logging.getLogger(__name__)


class HistorySyncMixin:
    async def _replay_scrollback_after_rewind(self) -> None:
        """清屏并重建 scrollback：Logo + 当前会话历史。"""
        self._renderer.reset_history_view()
        self._printed_history_index = 0

        def _clear_and_print_logo() -> None:
            out = sys.__stdout__ or sys.stdout
            out.write("\x1b[3J\x1b[2J\x1b[H")
            out.flush()
            scrollback_console = Console(
                file=out,
                force_terminal=True,
                width=self._terminal_width(),
            )
            print_logo(scrollback_console)

        try:
            await run_in_terminal(_clear_and_print_logo, in_executor=False)
        except Exception as exc:
            logger.warning(
                f"failed to clear terminal and replay logo: {exc}",
                exc_info=True,
            )

        self.add_resume_history("resume")

    def add_resume_history(self, mode: str) -> None:
        if mode != "resume":
            return

        items = self._session._agent._context.get_conversation_items_snapshot()
        history = [
            item
            for item in items
            if item.item_type in (
                ItemType.USER_MESSAGE,
                ItemType.ASSISTANT_MESSAGE,
                ItemType.TOOL_RESULT,
            )
        ]
        if not history:
            return

        self._renderer.append_system_message(f"history loaded: {len(history)} messages")

        # 维护 tool_call_id 映射，用于匹配工具调用和结果
        tool_call_info: dict[str, tuple[str, str]] = {}  # tool_call_id → (tool_name, args_summary)

        for item in history:
            if item.item_type == ItemType.USER_MESSAGE:
                content = str(item.content_text or "").strip()
                if content:
                    self._renderer.seed_user_message(content)
                continue

            # 处理 AssistantMessage - 提取 tool_calls 信息但不渲染为动态组件
            if item.item_type == ItemType.ASSISTANT_MESSAGE:
                message = getattr(item, "message", None)
                if isinstance(message, AssistantMessage) and message.tool_calls:
                    for tc in message.tool_calls:
                        tool_name = tc.function.name
                        args_str = tc.function.arguments
                        try:
                            import json

                            args_dict = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args_dict = {"_raw": args_str}
                        args_summary = self._summarize_tool_args(tool_name, args_dict)
                        # 存储映射，不调用 restore_tool_call（避免加入 _running_tools）
                        tool_call_info[tc.id] = (tool_name, args_summary)

                # 显示 assistant text
                assistant_text = self._extract_assistant_text(item).strip()
                if assistant_text:
                    self._renderer.append_assistant_message(assistant_text)
                continue

            # 处理 TOOL_RESULT - 直接输出静态文本（无计时器）
            if item.item_type == ItemType.TOOL_RESULT:
                message = getattr(item, "message", None)
                if message is None:
                    continue

                tool_call_id = getattr(message, "tool_call_id", None)
                is_error = getattr(message, "is_error", False)

                # 从映射中获取工具信息
                if tool_call_id and tool_call_id in tool_call_info:
                    tool_name, args_summary = tool_call_info[tool_call_id]
                    signature = (
                        f"{tool_name}({args_summary})" if args_summary else f"{tool_name}()"
                    )
                else:
                    # 回退：直接使用 tool_name
                    tool_name = getattr(message, "tool_name", item.tool_name or "UnknownTool")
                    signature = f"{tool_name}()"

                # Extract diff from raw_envelope for Edit/MultiEdit
                diff_lines: list[str] | None = None
                if tool_name in ("Edit", "MultiEdit") and not is_error:
                    raw_envelope = getattr(item, "metadata", {}) or {}
                    envelope = raw_envelope.get("tool_raw_envelope")
                    if isinstance(envelope, dict):
                        data = envelope.get("data", {})
                        if isinstance(data, dict):
                            diff = data.get("diff")
                            if isinstance(diff, list) and len(diff) > 0:
                                diff_lines = diff

                # 直接追加静态 HistoryEntry（无计时器）
                self._renderer.append_static_tool_result(
                    signature,
                    is_error,
                    diff_lines=diff_lines,
                )
                continue

        self._drain_history_sync()

    def _summarize_tool_args(self, tool_name: str, args: dict[str, Any]) -> str:
        """Summarize tool arguments for display in history."""
        from terminal_agent.tool_view import summarize_tool_args

        summary = summarize_tool_args(tool_name, args, self._renderer._project_root).strip()
        return summary

    @staticmethod
    def _extract_assistant_text(item: Any) -> str:
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

    async def _drain_history_async(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return
        group = render_history_group(
            console,
            pending,
            terminal_width=self._terminal_width(),
            render_markdown_to_plain=render_markdown_to_plain,
        )
        if group is None:
            return

        def _print() -> None:
            width = self._terminal_width()
            out = sys.__stdout__ or sys.stdout
            scrollback_console = Console(
                file=out,
                force_terminal=True,
                width=width,
            )
            print_history_group_sync(scrollback_console, group)

        await run_in_terminal(_print, in_executor=False)

    def _drain_history_sync(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return
        group = render_history_group(
            console,
            pending,
            terminal_width=self._terminal_width(),
            render_markdown_to_plain=render_markdown_to_plain,
        )
        if group is None:
            return
        print_history_group_sync(console, group)

    def _pending_history_entries(self) -> list[HistoryEntry]:
        entries = self._renderer.history_entries()
        if self._printed_history_index >= len(entries):
            return []
        pending = entries[self._printed_history_index :]
        self._printed_history_index = len(entries)
        # 根据 _show_thinking 开关过滤 thinking 条目
        if not getattr(self, "_show_thinking", False):
            pending = [e for e in pending if e.entry_type != "thinking"]
        return pending
