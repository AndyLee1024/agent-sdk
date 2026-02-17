"""Formatter for Write/Edit/MultiEdit tool results."""
from __future__ import annotations

from typing import Any

from comate_agent_sdk.system_tools.formatters.types import FormattedToolResult, ToolExecutionMeta

from .common import _as_int, _duration_from_result


def format_write_like_result(
    tool_name: str,
    tool_call_id: str,
    result: dict[str, Any],
) -> FormattedToolResult:
    data = result.get("data", {})
    env_meta = result.get("meta", {})

    relpath = str(data.get("relpath") or "")
    file_path = ""
    operation = ""
    sha_before = None
    sha_after = None
    if isinstance(env_meta, dict):
        file_path = str(env_meta.get("file_path") or "")
        operation = str(env_meta.get("operation") or "")
        sha_before = env_meta.get("sha256_before")
        sha_after = env_meta.get("sha256_after")

    if not file_path:
        file_path = relpath or "unknown"
    if not operation:
        operation = tool_name.lower()
    if sha_after is None:
        sha_after = data.get("sha256") or data.get("after_sha256")

    title = f"# {tool_name}: {file_path}"
    lines = [title, ""]
    if tool_name == "Write":
        lines.append(f"Operation: {operation} (created={bool(data.get('created'))})")
        lines.append(f"Bytes written: {_as_int(data.get('bytes_written'), 0)}")
        lines.append(f"File bytes: {_as_int(data.get('file_bytes'), 0)}")
    elif tool_name == "Edit":
        lines.append(f"Replacements: {_as_int(data.get('replacements'), 0)}")
    elif tool_name == "MultiEdit":
        lines.append(f"Total replacements: {_as_int(data.get('total_replacements'), 0)}")
        lines.append(f"Created: {bool(data.get('created'))}")
        lines.append(f"File bytes: {_as_int(data.get('bytes'), 0)}")

    if sha_before:
        lines.append(f"SHA256 before: {sha_before}")
    if sha_after:
        lines.append(f"SHA256 after: {sha_after}")
    if relpath:
        lines.append(f"Path: {relpath}")

    verify_path = relpath or file_path
    hints = [
        {
            "action": "Read",
            "priority": "high",
            "args": {
                "file_path": verify_path,
                "offset_line": 0,
                "limit_lines": 200,
            },
        }
    ]

    file_ops = {
        "file_path": file_path,
        "operation": operation,
        "bytes_written": _as_int(data.get("bytes_written") or data.get("bytes"), 0),
        "sha256_before": sha_before,
        "sha256_after": sha_after,
    }
    file_ops = {k: v for k, v in file_ops.items() if v is not None}

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        file_ops=file_ops,
        retrieval_hints=hints,
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)
