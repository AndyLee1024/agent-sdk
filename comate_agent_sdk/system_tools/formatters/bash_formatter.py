"""Formatter for Bash tool results."""
from __future__ import annotations

from typing import Any

from comate_agent_sdk.system_tools.formatters.types import FormattedToolResult, ToolExecutionMeta

from .common import _artifact_hint, _as_float, _as_int, _default_truncation, _duration_from_result, _format_hints_footer


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def format_bash_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    env_meta = result.get("meta", {})

    args = env_meta.get("args") if isinstance(env_meta, dict) else None
    if isinstance(args, list):
        cmd = " ".join(str(part) for part in args)
    else:
        cmd = "<command>"

    exit_code = _as_int(data.get("exit_code"), 0)
    duration_ms = _as_float(data.get("duration_ms"))
    timed_out = bool(data.get("timed_out"))
    truncated = bool(data.get("truncated"))

    stdout = _truncate_text(str(data.get("stdout", "")), 6000)
    stderr = _truncate_text(str(data.get("stderr", "")), 6000)

    lines = [
        f"# Bash: {cmd}",
        "",
        f"Exit code: {exit_code}",
        f"Timed out: {timed_out}",
    ]
    if duration_ms is not None:
        lines.append(f"Duration: {duration_ms:.1f}ms")

    lines.extend(["", "stdout:", stdout or "(empty)", "", "stderr:", stderr or "(empty)"])

    hints: list[dict[str, Any]] = []
    artifact_hint = _artifact_hint(data)
    if artifact_hint is not None:
        hints.append(artifact_hint)

    if truncated:
        hints.append(
            {
                "action": "Bash",
                "priority": "medium",
                "args": {
                    "args": ["<command>", "<with narrower output>"],
                    "max_output_chars": 5000,
                },
            }
        )

    footer = _format_hints_footer(hints) if truncated or timed_out else ""
    if footer:
        lines.extend(["", footer])

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        truncation=_default_truncation(data),
        retrieval_hints=hints or None,
        duration_ms=duration_ms if duration_ms is not None else _duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)
