"""Common utility functions for formatters."""
from __future__ import annotations

import json
from typing import Any

from comate_agent_sdk.context.truncation import TruncationRecord


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def _duration_from_result(result: dict[str, Any]) -> float | None:
    meta = result.get("meta", {})
    data = result.get("data", {})
    if isinstance(meta, dict) and "duration_ms" in meta:
        return _as_float(meta.get("duration_ms"))
    if isinstance(data, dict) and "duration_ms" in data:
        return _as_float(data.get("duration_ms"))
    return None


def _format_hints_footer(hints: list[dict[str, Any]]) -> str:
    if not hints:
        return ""

    lines = ["<system-reminder>", "Recommended next step:"]
    for hint in hints[:3]:
        action = str(hint.get("action", "")).strip()
        args = hint.get("args", {})
        if not action:
            continue
        if isinstance(args, dict):
            parts = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()]
            lines.append(f"* {action}({', '.join(parts)})")
        else:
            lines.append(f"* {action}()")
    lines.append("</system-reminder>")
    return "\n".join(lines)


def _default_truncation(data: dict[str, Any]) -> TruncationRecord | None:
    truncated = bool(data.get("truncated")) or bool(data.get("has_more")) or bool(
        data.get("raw_output_truncated")
    )
    if not truncated:
        return None

    reasons: list[str] = []
    if data.get("has_more"):
        reasons.append("line_limit")
    if data.get("artifact"):
        reasons.append("output_spilled")
    if data.get("raw_output_truncated"):
        reasons.append("output_capture_limit")
    if not reasons and data.get("truncated"):
        reasons.append("truncated")

    total_estimate = (
        data.get("total_lines")
        or data.get("total_matches")
        or data.get("count")
        or len(data.get("matches", []) if isinstance(data.get("matches"), list) else [])
    )
    return TruncationRecord(
        formatter_truncated=True,
        formatter_reason=",".join(reasons) if reasons else "truncated",
        formatter_total_estimate=_as_int(total_estimate, 0),
    )


def _artifact_hint(data: dict[str, Any], *, priority: str = "low") -> dict[str, Any] | None:
    artifact = data.get("artifact")
    if not isinstance(artifact, dict):
        return None
    relpath = artifact.get("relpath")
    if not relpath:
        return None
    return {
        "action": "Read",
        "priority": priority,
        "args": {
            "file_path": relpath,
            "format": "raw",
            "offset_line": 0,
            "limit_lines": 500,
        },
    }
