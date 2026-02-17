"""Formatter for Read tool results."""
from __future__ import annotations

from typing import Any

from comate_agent_sdk.context.truncation import TruncationRecord
from comate_agent_sdk.system_tools.formatters.types import FormattedToolResult, ToolExecutionMeta

from .common import (
    _as_int,
    _duration_from_result,
    _format_hints_footer,
)


def format_read_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    env_meta = result.get("meta", {})

    file_path = ""
    if isinstance(env_meta, dict):
        file_path = str(env_meta.get("file_path") or "")
    if not file_path:
        file_path = str(data.get("file_path") or "unknown")

    offset_line = _as_int(env_meta.get("offset_line") if isinstance(env_meta, dict) else 0, 0)
    total_lines = _as_int(data.get("total_lines"), 0)
    lines_returned = _as_int(data.get("lines_returned"), 0)
    has_more = bool(data.get("has_more"))
    truncated = bool(data.get("truncated")) or has_more

    start_line = offset_line + 1 if lines_returned > 0 else 0
    end_line = offset_line + lines_returned

    reasons: list[str] = []
    if has_more:
        reasons.append("line_limit")
    if data.get("truncated") and not reasons:
        reasons.append("line_clip")
    trunc_note = f" (TRUNCATED: {', '.join(reasons)})" if truncated else ""

    title = f"# Read: {file_path}"
    meta_line = f"Lines {start_line}-{end_line} of {total_lines}{trunc_note}"
    body = str(data.get("content", ""))

    hints: list[dict[str, Any]] = []
    if has_more:
        hints.append(
            {
                "action": "Read",
                "priority": "high",
                "args": {
                    "file_path": file_path,
                    "offset_line": _as_int(data.get("next_offset_line"), end_line),
                    "limit_lines": 500,
                },
            }
        )
    if truncated:
        hints.append(
            {
                "action": "Grep",
                "priority": "medium",
                "args": {
                    "pattern": "<keywords>",
                    "path": file_path,
                    "output_mode": "content",
                },
            }
        )
    footer = _format_hints_footer(hints) if truncated else ""
    text_parts = [title, "", meta_line, "", body]
    if footer:
        text_parts.extend(["", footer])

    truncation: TruncationRecord | None = None
    if truncated:
        truncation = TruncationRecord(
            formatter_truncated=True,
            formatter_reason=",".join(reasons) if reasons else "truncated",
            formatter_shown_range={"start_line": start_line, "end_line": end_line},
            formatter_total_estimate=total_lines,
        )

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        truncation=truncation,
        retrieval_hints=hints or None,
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(text_parts).strip(), meta=meta)
