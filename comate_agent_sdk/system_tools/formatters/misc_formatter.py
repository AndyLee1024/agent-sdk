"""Formatters for TodoWrite, WebFetch, AskUserQuestion, and generic results."""
from __future__ import annotations

import json
from typing import Any

from comate_agent_sdk.context.truncation import TruncationRecord
from comate_agent_sdk.system_tools.formatters.types import FormattedToolResult, ToolExecutionMeta

from .common import _artifact_hint, _as_float, _as_int, _default_truncation, _duration_from_result, _format_hints_footer


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def format_todo_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    message = str(result.get("message") or "").strip()
    lines = [
        "# TodoWrite",
        "",
        f"Total todos: {_as_int(data.get('count'), 0)}",
        f"Active todos: {_as_int(data.get('active_count'), 0)}",
        f"Persisted: {bool(data.get('persisted'))}",
    ]
    todo_path = data.get("todo_path")
    if todo_path:
        lines.append(f"Todo path: {todo_path}")

    # 显示活跃 todos 的详细列表
    todos = data.get("todos", [])
    if todos:
        active_todos = [t for t in todos if t.get("status") in ("pending", "in_progress")]
        if active_todos:
            lines.extend(["", "Active Tasks:"])
            for todo in active_todos:
                todo_id = todo.get("id", "?")
                status = todo.get("status", "unknown")
                priority = todo.get("priority", "medium")
                content = todo.get("content", "")
                lines.append(f"- [{status}] #{todo_id} ({priority}) {content}")

    if message:
        lines.extend(["", message])

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)


def format_webfetch_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    final_url = str(data.get("final_url") or "unknown")
    status = _as_int(data.get("status"), 0)
    cached = bool(data.get("cached"))
    truncated_for_llm = bool(data.get("truncated_for_llm"))
    summary_text = _truncate_text(str(data.get("summary_text", "")), 6000)
    model_used = str(data.get("model_used") or "")

    lines = [
        f"# WebFetch: {final_url}",
        "",
        f"Status: {status}",
        f"Cached: {cached}",
    ]
    if model_used:
        lines.append(f"Model: {model_used}")
    lines.extend(["", "Summary:", summary_text or "(empty)"])

    hints: list[dict[str, Any]] = []
    artifact_hint = _artifact_hint(data, priority="high")
    if artifact_hint is not None:
        hints.append(artifact_hint)

    footer = _format_hints_footer(hints) if truncated_for_llm else ""
    if footer:
        lines.extend(["", footer])

    truncation: TruncationRecord | None = None
    if truncated_for_llm:
        truncation = TruncationRecord(
            formatter_truncated=True,
            formatter_reason="llm_input_limit",
            formatter_total_estimate=0,
        )

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        truncation=truncation,
        retrieval_hints=hints or None,
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)


def format_ask_user_question_result(
    tool_name: str,
    tool_call_id: str,
    result: dict[str, Any],
) -> FormattedToolResult:
    data = result.get("data", {})
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    lines = [
        "# AskUserQuestion",
        "",
        f"Status: {data.get('status', 'waiting_for_input')}",
        f"Question count: {len(questions)}",
        "",
    ]
    for idx, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue
        header = str(question.get("header") or "")
        text = str(question.get("question") or "")
        lines.append(f"{idx}. [{header}] {text}")

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)


def format_generic_ok(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    message = result.get("message")
    text_parts = [f"# {tool_name}", ""]
    if message:
        text_parts.extend([str(message), ""])
    text_parts.append(json.dumps(data, ensure_ascii=False, indent=2))

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        truncation=_default_truncation(data if isinstance(data, dict) else {}),
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(text_parts).strip(), meta=meta)


def format_error(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    error = result.get("error", {})
    code = str(error.get("code") or "INTERNAL")
    message = str(error.get("message") or "Unknown error")
    field_errors = error.get("field_errors", [])
    retryable = bool(error.get("retryable"))

    lines = [
        f"# {tool_name} Error",
        "",
        f"Code: {code}",
        f"Message: {message}",
        f"Retryable: {retryable}",
    ]
    first_field: str | None = None
    if isinstance(field_errors, list) and field_errors:
        lines.append("")
        lines.append("Field errors:")
        for item in field_errors[:5]:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "")
            item_msg = str(item.get("message") or "")
            if field and first_field is None:
                first_field = field
            lines.append(f"- {field}: {item_msg}")

    hints: list[dict[str, Any]] = []
    if first_field:
        hints.append(
            {
                "action": tool_name,
                "priority": "high",
                "args": {
                    first_field: "<correct_value>",
                },
            }
        )
    if code == "NOT_FOUND":
        hints.append(
            {
                "action": "LS",
                "priority": "medium",
                "args": {
                    "path": "<parent_directory>",
                    "head_limit": 200,
                },
            }
        )
    if code in {"TIMEOUT", "RATE_LIMITED", "INTERNAL"} or retryable:
        hints.append(
            {
                "action": tool_name,
                "priority": "medium",
                "args": {
                    "retry": True,
                },
            }
        )
    if code in {"INVALID_ARGUMENT", "CONFLICT"} and not first_field:
        hints.append(
            {
                "action": tool_name,
                "priority": "high",
                "args": {
                    "fix_arguments": True,
                },
            }
        )

    footer = _format_hints_footer(hints)
    if footer:
        lines.extend(["", footer])

    truncation_source = result.get("data", {})
    if (not isinstance(truncation_source, dict) or not truncation_source) and isinstance(
        result.get("meta"), dict
    ):
        truncation_source = result.get("meta", {})

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="error",
        truncation=_default_truncation(truncation_source if isinstance(truncation_source, dict) else {}),
        retrieval_hints=hints or None,
        duration_ms=_duration_from_result(result),
        error_code=code,
        error_field=first_field,
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)
