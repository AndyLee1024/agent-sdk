from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from comate_agent_sdk.context.truncation import TruncationRecord
from comate_agent_sdk.system_tools.tool_result import is_tool_result_envelope


@dataclass
class ToolExecutionMeta:
    """工具执行元数据（存储到 ContextItem.metadata）"""

    tool_name: str
    tool_call_id: str
    status: Literal["ok", "error"]
    truncation: TruncationRecord | None = None
    file_ops: dict[str, Any] | None = None
    retrieval_hints: list[dict[str, Any]] | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    error_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "status": self.status,
            "truncation": self.truncation.to_dict() if self.truncation else None,
            "file_ops": self.file_ops,
            "retrieval_hints": self.retrieval_hints,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
            "error_field": self.error_field,
        }
        return {k: v for k, v in payload.items() if v is not None}


@dataclass
class FormattedToolResult:
    """格式化后的工具结果"""

    text: str
    meta: ToolExecutionMeta


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

    lines = ["<system-reminder>", "Recommended next step (token-efficient):"]
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


def _format_read_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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
    if data.get("artifact"):
        reasons.append("output_spilled")
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
                    "pattern": "<keyword>",
                    "path": file_path,
                    "output_mode": "content",
                },
            }
        )
    artifact_hint = _artifact_hint(data)
    if artifact_hint is not None:
        hints.append(artifact_hint)

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


def _format_write_like_result(
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


def _format_glob_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    env_meta = result.get("meta", {})

    pattern = str(env_meta.get("pattern") if isinstance(env_meta, dict) else "") or "**/*"
    search_path = str(data.get("search_path") or "")
    if not search_path and isinstance(env_meta, dict):
        search_path = str(env_meta.get("search_path") or "")
    matches = data.get("matches", [])
    if not isinstance(matches, list):
        matches = []
    count = _as_int(data.get("count"), len(matches))
    truncated = bool(data.get("truncated"))

    lines = [
        f"# Glob: {pattern}",
        "",
        f"Path: {search_path or '.'}",
        f"Matches shown: {len(matches)} of {count}" + (" (TRUNCATED)" if truncated else ""),
        "",
    ]
    if matches:
        lines.extend([f"- {m}" for m in matches[:100]])
    else:
        lines.append("- (no matches)")

    hints: list[dict[str, Any]] = []
    if truncated:
        hints.append(
            {
                "action": "Glob",
                "priority": "high",
                "args": {
                    "pattern": pattern,
                    "path": search_path,
                    "head_limit": min(max(count, 300), 1000),
                },
            }
        )
    artifact_hint = _artifact_hint(data)
    if artifact_hint is not None:
        hints.append(artifact_hint)

    footer = _format_hints_footer(hints) if truncated else ""
    if footer:
        lines.extend(["", footer])

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        truncation=_default_truncation(data),
        retrieval_hints=hints or None,
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)


def _format_ls_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    env_meta = result.get("meta", {})

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    count = _as_int(data.get("count"), len(entries))
    truncated = bool(data.get("truncated"))
    path = str(data.get("path") or "")
    if not path and isinstance(env_meta, dict):
        path = str(env_meta.get("path") or "")
    sort_by = str(env_meta.get("sort_by") if isinstance(env_meta, dict) else "") or "name"

    lines = [
        f"# LS: {path or '.'}",
        "",
        f"Entries shown: {len(entries)} of {count} (sorted by {sort_by})"
        + (" (TRUNCATED)" if truncated else ""),
        "",
    ]
    if entries:
        for row in entries[:100]:
            if not isinstance(row, dict):
                continue
            item_type = str(row.get("type") or "other").upper()
            name = str(row.get("name") or "")
            size = _as_int(row.get("size"), 0)
            lines.append(f"- [{item_type}] {name} (size={size})")
    else:
        lines.append("- (empty)")

    hints: list[dict[str, Any]] = []
    if truncated:
        hints.append(
            {
                "action": "LS",
                "priority": "high",
                "args": {
                    "path": path,
                    "head_limit": min(max(count, 300), 1000),
                    "sort_by": sort_by,
                },
            }
        )
    artifact_hint = _artifact_hint(data)
    if artifact_hint is not None:
        hints.append(artifact_hint)

    footer = _format_hints_footer(hints) if truncated else ""
    if footer:
        lines.extend(["", footer])

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        truncation=_default_truncation(data),
        retrieval_hints=hints or None,
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)


def _format_grep_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
    data = result.get("data", {})
    env_meta = result.get("meta", {})

    mode = str(env_meta.get("output_mode") if isinstance(env_meta, dict) else "") or "files_with_matches"
    pattern = str(env_meta.get("pattern") if isinstance(env_meta, dict) else "") or "<pattern>"
    search_path = str(env_meta.get("search_path") if isinstance(env_meta, dict) else "") or str(
        data.get("search_path", ".")
    )

    lines = [f"# Grep ({mode}): {pattern}", "", f"Path: {search_path}", ""]
    hints: list[dict[str, Any]] = []

    truncated = bool(data.get("truncated"))
    lower_bound = bool(data.get("total_matches_is_lower_bound"))
    if mode == "files_with_matches":
        files = data.get("files", [])
        if not isinstance(files, list):
            files = []
        count = _as_int(data.get("count"), len(files))
        lines.append(f"Files shown: {len(files)} of {count}" + (" (TRUNCATED)" if truncated else ""))
        lines.append("")
        lines.extend([f"- {f}" for f in files[:100]] or ["- (no matches)"])
    elif mode == "count":
        counts = data.get("counts", [])
        if not isinstance(counts, list):
            counts = []
        total_matches = _as_int(data.get("total_matches"), 0)
        lines.append(
            f"Count rows shown: {len(counts)}; total matches={total_matches}"
            + (" (TRUNCATED)" if truncated else "")
        )
        lines.append("")
        for row in counts[:100]:
            if isinstance(row, dict):
                lines.append(f"- {row.get('file')}: {row.get('count')}")
    else:
        matches = data.get("matches", [])
        if not isinstance(matches, list):
            matches = []
        total_matches = _as_int(data.get("total_matches"), len(matches))
        suffix = " (LOWER BOUND)" if lower_bound else ""
        lines.append(
            f"Matches shown: {len(matches)} of {total_matches}{suffix}"
            + (" (TRUNCATED)" if truncated else "")
        )
        lines.append("")
        for m in matches[:100]:
            if not isinstance(m, dict):
                continue
            file = m.get("file", "")
            ln = m.get("line_number")
            text = str(m.get("line", ""))
            loc = f"{file}:{ln}" if ln else str(file)
            lines.append(f"- {loc} | {text}")
        if not matches:
            lines.append("- (no matches)")

        if matches:
            first = matches[0]
            if isinstance(first, dict) and first.get("file"):
                line_number = _as_int(first.get("line_number"), 1)
                hints.append(
                    {
                        "action": "Read",
                        "priority": "high",
                        "args": {
                            "file_path": first.get("file"),
                            "offset_line": max(line_number - 20, 0),
                            "limit_lines": 120,
                        },
                    }
                )

    if truncated:
        hints.append(
            {
                "action": "Grep",
                "priority": "medium",
                "args": {
                    "pattern": pattern,
                    "path": search_path,
                    "output_mode": mode,
                    "head_limit": 300,
                },
            }
        )
    artifact_hint = _artifact_hint(data)
    if artifact_hint is not None:
        hints.append(artifact_hint)

    footer = _format_hints_footer(hints) if truncated else ""
    if footer:
        lines.extend(["", footer])

    meta = ToolExecutionMeta(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status="ok",
        truncation=_default_truncation(data),
        retrieval_hints=hints or None,
        duration_ms=_duration_from_result(result),
    )
    return FormattedToolResult(text="\n".join(lines).strip(), meta=meta)


def _format_bash_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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


def _format_todo_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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


def _format_webfetch_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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


def _format_ask_user_question_result(
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


def _format_generic_ok(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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


def _format_error(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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


_TOOL_FORMATTERS = {
    "Read": _format_read_result,
    "Write": _format_write_like_result,
    "Edit": _format_write_like_result,
    "MultiEdit": _format_write_like_result,
    "Glob": _format_glob_result,
    "Grep": _format_grep_result,
    "LS": _format_ls_result,
    "Bash": _format_bash_result,
    "TodoWrite": _format_todo_result,
    "WebFetch": _format_webfetch_result,
    "AskUserQuestion": _format_ask_user_question_result,
}


class OutputFormatter:
    """工具输出格式化器"""

    @staticmethod
    def format(
        *,
        tool_name: str,
        tool_call_id: str,
        result_dict: dict[str, Any],
    ) -> FormattedToolResult:
        if not is_tool_result_envelope(result_dict):
            text = (
                json.dumps(result_dict, ensure_ascii=False, indent=2)
                if isinstance(result_dict, dict)
                else str(result_dict)
            )
            return FormattedToolResult(
                text=text,
                meta=ToolExecutionMeta(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="ok",
                ),
            )

        if not bool(result_dict.get("ok", False)):
            return _format_error(tool_name, tool_call_id, result_dict)

        formatter = _TOOL_FORMATTERS.get(tool_name, _format_generic_ok)
        return formatter(tool_name, tool_call_id, result_dict)
