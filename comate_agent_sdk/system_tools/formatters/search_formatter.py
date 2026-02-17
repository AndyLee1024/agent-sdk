"""Formatters for Glob, Grep, and LS tool results."""
from __future__ import annotations

from typing import Any

from comate_agent_sdk.system_tools.formatters.types import FormattedToolResult, ToolExecutionMeta

from .common import _artifact_hint, _as_int, _default_truncation, _duration_from_result, _format_hints_footer


def _format_file_size(size_bytes: int) -> str:
    """智能单位转换：B/KB/MB/GB"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f}MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.1f}GB"


def format_glob_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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


def format_grep_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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
        # P1-3: Handle new grouped format (matches_by_file)
        matches_by_file = data.get("matches_by_file", {})
        if not isinstance(matches_by_file, dict):
            matches_by_file = {}

        total_matches = _as_int(data.get("total_matches"), 0)
        file_count = _as_int(data.get("file_count"), len(matches_by_file))
        suffix = " (LOWER BOUND)" if lower_bound else ""
        lines.append(
            f"Matches: {total_matches}{suffix} across {file_count} files"
            + (" (TRUNCATED)" if truncated else "")
        )
        lines.append("")

        match_count = 0
        first_file = None
        first_line_number = None
        for file, file_data in list(matches_by_file.items())[:20]:  # Limit to 20 files
            if not isinstance(file_data, dict):
                continue
            matches_in_file = file_data.get("matches", [])
            if not isinstance(matches_in_file, list):
                continue

            lines.append(f"## {file} ({len(matches_in_file)} matches)")
            for m in matches_in_file[:5]:  # Limit to 5 matches per file
                if not isinstance(m, dict):
                    continue
                ln = m.get("line_number")
                text = str(m.get("line", ""))
                loc = f"  {ln}" if ln else "  -"
                lines.append(f"{loc} | {text}")

                # Track first match for hint
                if first_file is None and ln is not None:
                    first_file = file
                    first_line_number = _as_int(ln, 1)

                match_count += 1
                if match_count >= 100:
                    break
            if match_count >= 100:
                break
            lines.append("")

        if not matches_by_file:
            lines.append("- (no matches)")

        if first_file is not None and first_line_number is not None:
            hints.append(
                {
                    "action": "Read",
                    "priority": "high",
                    "args": {
                        "file_path": first_file,
                        "offset_line": max(first_line_number - 20, 0),
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


def format_ls_result(tool_name: str, tool_call_id: str, result: dict[str, Any]) -> FormattedToolResult:
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
            size_display = _format_file_size(size)
            lines.append(f"- [{item_type}] {name} (size={size_display})")
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
