from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from comate_agent_sdk.llm.messages import UserMessage
from comate_agent_sdk.tools.decorator import Tool, tool
from comate_agent_sdk.tools.depends import Depends
from comate_agent_sdk.tools.system_context import SystemToolContext, get_system_tool_context
from comate_agent_sdk.system_tools.description import (
    BASH_USAGE_RULES,
    READ_USAGE_RULES,
    WRITE_USAGE_RULES,
    EDIT_USAGE_RULES,
    MULTIEDIT_USAGE_RULES,
    GLOB_USAGE_RULES,
    GREP_USAGE_RULES,
    LS_USAGE_RULES,
    TODO_USAGE_RULES,
    WEBFETCH_USAGE_RULES,
    ASKUSERQUESTION_USAGE_RULES,
)

logger = logging.getLogger(__name__)

_READ_DEFAULT_LIMIT = 500
_LINE_TRUNCATE_CHARS = 2000
_BASH_DEFAULT_TIMEOUT_MS = 120_000
_BASH_MAX_TIMEOUT_MS = 600_000
_BASH_OUTPUT_TRUNCATE_CHARS = 30_000
_WEBFETCH_TIMEOUT_SECONDS = 20
_WEBFETCH_LLM_TIMEOUT_SECONDS = 30
_WEBFETCH_CACHE_TTL_SECONDS = 15 * 60
_WEBFETCH_MARKDOWN_TRUNCATE_CHARS = 50_000


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def _truncate_line(line: str, max_chars: int) -> str:
    if len(line) <= max_chars:
        return line
    return line[:max_chars] + "...(truncated)"


def _ensure_abs_path(file_path: str) -> Path:
    p = Path(file_path)
    if not p.is_absolute():
        raise ValueError(f"file_path 必须是绝对路径：{file_path}")
    return p


def _resolve_search_path(path: str | None, project_root: Path) -> Path:
    if path is None:
        return project_root
    p = Path(path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _relpath(p: Path, project_root: Path) -> str:
    try:
        return os.path.relpath(str(p), start=str(project_root))
    except Exception:
        return str(p)


def _warn_bash_command(command: str) -> str | None:
    # 仅 warning（不阻止执行）：提示优先用专用工具
    # 注意：不要过度严格，避免误伤，比如字符串里出现 "grep"。
    # 这里以常见的命令起始/分隔符后的 token 为准。
    pattern = r'(^|[;&|()]\s*)(find|grep|rg|cat|ls|head|tail)\b'
    m = re.search(pattern, command)
    if not m:
        return None
    cmd = m.group(2)
    return (
        f"Warning: 检测到命令 `{cmd}`。通常应优先使用专用工具（搜索用 Grep/Glob，读文件用 Read），"
        f"以获得更稳定的权限与输出格式。"
    )


def _session_root_from_ctx(ctx: SystemToolContext) -> Path | None:
    if ctx.session_root is None:
        return None
    try:
        return ctx.session_root.resolve()
    except Exception:
        return ctx.session_root


def _cleanup_ttl_cache(cache: dict[str, dict[str, Any]], *, now: float, ttl: int) -> None:
    expired: list[str] = []
    for k, v in cache.items():
        ts = float(v.get("ts", 0.0))
        if now - ts > ttl:
            expired.append(k)
    for k in expired:
        cache.pop(k, None)


@tool(
    "Executes a bash command with optional timeout. Returns combined output and exit code.",
    name="Bash",
    usage_rules=BASH_USAGE_RULES,
)
async def Bash(
    command: str,
    timeout: int | None = None,
    description: str | None = None,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    if description:
        logger.info(f"Bash: {description} | {command}")
    else:
        logger.info(f"Bash: {command}")

    warn = _warn_bash_command(command)

    timeout_ms = timeout if timeout is not None else _BASH_DEFAULT_TIMEOUT_MS
    if timeout_ms > _BASH_MAX_TIMEOUT_MS:
        timeout_ms = _BASH_MAX_TIMEOUT_MS
    if timeout_ms < 0:
        timeout_ms = _BASH_DEFAULT_TIMEOUT_MS

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(ctx.project_root),
            timeout=timeout_ms / 1000.0,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if warn:
            output = f"{warn}\n\n{output}"
        output = _truncate_text(output.strip(), _BASH_OUTPUT_TRUNCATE_CHARS)
        return {"output": output, "exitCode": int(result.returncode), "killed": False}
    except subprocess.TimeoutExpired as e:
        stdout = getattr(e, "stdout", "") or ""
        stderr = getattr(e, "stderr", "") or ""
        output = stdout + stderr
        if warn:
            output = f"{warn}\n\n{output}"
        output = _truncate_text(
            output.strip() or f"Command timed out after {timeout_ms}ms",
            _BASH_OUTPUT_TRUNCATE_CHARS,
        )
        return {"output": output, "exitCode": -1, "killed": True}
    except Exception as e:
        msg = f"Error: Bash 执行失败：{e}"
        logger.error(msg, exc_info=True)
        return {"output": msg, "exitCode": -1, "killed": None}


@tool("Reads a text file with line numbers.", name="Read", usage_rules=READ_USAGE_RULES)
async def Read(
    file_path: str,
    offset: int | None = None,
    limit: int | None = None,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        path = _ensure_abs_path(file_path)
        if not path.exists():
            return {"content": f"Error: File not found: {file_path}", "total_lines": 0, "lines_returned": 0}
        if path.is_dir():
            return {"content": f"Error: Path is a directory: {file_path}", "total_lines": 0, "lines_returned": 0}

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total_lines = len(lines)

        start = int(offset) if offset is not None else 0
        if start < 0:
            start = 0
        take = int(limit) if limit is not None else _READ_DEFAULT_LIMIT
        if take < 0:
            take = _READ_DEFAULT_LIMIT

        sliced = lines[start : start + take]
        rendered: list[str] = []
        for i, line in enumerate(sliced, start=start):
            rendered.append(f"{i + 1:6d}\t{_truncate_line(line, _LINE_TRUNCATE_CHARS)}")

        return {
            "content": "\n".join(rendered),
            "total_lines": total_lines,
            "lines_returned": len(sliced),
        }
    except Exception as e:
        msg = f"Error: Read 失败：{e}"
        logger.error(msg, exc_info=True)
        return {"content": msg, "total_lines": 0, "lines_returned": 0}


@tool("Writes content to a file (overwrites).", name="Write", usage_rules=WRITE_USAGE_RULES)
async def Write(
    file_path: str,
    content: str,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        path = _ensure_abs_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        bytes_written = len(content.encode("utf-8"))
        return {
            "message": f"Success: Wrote {bytes_written} bytes to {file_path}",
            "bytes_written": bytes_written,
            "file_path": file_path,
        }
    except Exception as e:
        msg = f"Error: Write 失败：{e}"
        logger.error(msg, exc_info=True)
        return {"message": msg, "bytes_written": 0, "file_path": file_path}


@tool("Performs exact string replacement in a file.", name="Edit", usage_rules=EDIT_USAGE_RULES)
async def Edit(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool | None = None,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        path = _ensure_abs_path(file_path)
        if not path.exists():
            return {"message": f"Error: File not found: {file_path}", "replacements": 0, "file_path": file_path}
        if path.is_dir():
            return {"message": f"Error: Path is a directory: {file_path}", "replacements": 0, "file_path": file_path}
        if old_string == "":
            return {"message": "Error: old_string 不能为空", "replacements": 0, "file_path": file_path}
        if old_string == new_string:
            return {"message": "Error: new_string 必须与 old_string 不同", "replacements": 0, "file_path": file_path}

        content = path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_string)
        if count == 0:
            return {"message": f"Error: String not found in {file_path}", "replacements": 0, "file_path": file_path}

        do_replace_all = bool(replace_all) if replace_all is not None else False
        if (not do_replace_all) and count > 1:
            return {
                "message": f"Error: old_string 在文件中出现 {count} 次；请提供更精确的 old_string 或设置 replace_all=true",
                "replacements": 0,
                "file_path": file_path,
            }

        if do_replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        path.write_text(new_content, encoding="utf-8")
        return {"message": "Success: Edit applied", "replacements": replacements, "file_path": file_path}
    except Exception as e:
        msg = f"Error: Edit 失败：{e}"
        logger.error(msg, exc_info=True)
        return {"message": msg, "replacements": 0, "file_path": file_path}


class _MultiEditOp(BaseModel):
    old_string: str = Field(description="The text to replace")
    new_string: str = Field(description="The text to replace it with")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurences of old_string (default false).",
    )

    model_config = {"extra": "forbid"}


class _MultiEditInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to modify")
    edits: list[_MultiEditOp] = Field(
        min_length=1,
        description="Array of edit operations to perform sequentially on the file",
    )

    model_config = {"extra": "forbid"}


@tool(
    "Make multiple edits to a single file atomically (all succeed or none).",
    name="MultiEdit",
    usage_rules=MULTIEDIT_USAGE_RULES,
)
async def MultiEdit(
    params: _MultiEditInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    file_path = params.file_path
    edits = params.edits

    try:
        path = _ensure_abs_path(file_path)
        exists = path.exists()

        content = ""
        if exists:
            if path.is_dir():
                return {"message": f"Error: Path is a directory: {file_path}", "replacements": 0, "file_path": file_path}
            content = path.read_text(encoding="utf-8", errors="replace")
        else:
            # 创建新文件：仅当第一条 edit 的 old_string 为空
            if not edits:
                return {"message": "Error: edits 不能为空", "replacements": 0, "file_path": file_path}
            if edits[0].old_string != "":
                return {
                    "message": "Error: File not found and first edit.old_string must be empty to create a new file",
                    "replacements": 0,
                    "file_path": file_path,
                }
            # content 由第一条 new_string 初始化（后续 edits 继续对该内容做替换）
            content = edits[0].new_string
            edits = edits[1:]

        total_replacements = 0
        for op in edits:
            if op.old_string == "":
                return {"message": "Error: old_string 不能为空（创建新文件仅允许第一条）", "replacements": 0, "file_path": file_path}
            if op.old_string == op.new_string:
                return {"message": "Error: new_string 必须与 old_string 不同", "replacements": 0, "file_path": file_path}

            count = content.count(op.old_string)
            if count == 0:
                return {"message": f"Error: String not found in {file_path}", "replacements": 0, "file_path": file_path}

            if not op.replace_all and count > 1:
                return {
                    "message": f"Error: old_string 在文件中出现 {count} 次；请提供更精确的 old_string 或设置 replace_all=true",
                    "replacements": 0,
                    "file_path": file_path,
                }

            if op.replace_all:
                content = content.replace(op.old_string, op.new_string)
                total_replacements += count
            else:
                content = content.replace(op.old_string, op.new_string, 1)
                total_replacements += 1

        # 原子落盘：全部成功后才写入
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"message": "Success: MultiEdit applied", "replacements": total_replacements, "file_path": file_path}
    except Exception as e:
        msg = f"Error: MultiEdit 失败：{e}"
        logger.error(msg, exc_info=True)
        return {"message": msg, "replacements": 0, "file_path": file_path}


@tool("Find files matching a glob pattern.", name="Glob", usage_rules=GLOB_USAGE_RULES)
async def Glob(
    pattern: str,
    path: str | None = None,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        search_dir = _resolve_search_path(path, ctx.project_root)
        if not search_dir.exists():
            return {"matches": [], "count": 0, "search_path": str(search_dir)}
        if not search_dir.is_dir():
            return {"matches": [], "count": 0, "search_path": str(search_dir)}

        candidates = [p for p in search_dir.glob(pattern)]
        files = [p for p in candidates if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        matches = [_relpath(p.resolve(), ctx.project_root) for p in files]
        return {"matches": matches, "count": len(matches), "search_path": str(search_dir.resolve())}
    except Exception as e:
        msg = f"Error: Glob 失败：{e}"
        logger.error(msg, exc_info=True)
        return {"matches": [], "count": 0, "search_path": str(ctx.project_root)}


class GrepInput(BaseModel):
    pattern: str
    path: str | None = None
    glob: str | None = None
    type: str | None = None
    output_mode: Literal["content", "files_with_matches", "count"] | None = None

    B: int | None = Field(default=None, alias="-B")
    A: int | None = Field(default=None, alias="-A")
    C: int | None = Field(default=None, alias="-C")
    i: bool | None = Field(default=None, alias="-i")
    n: bool | None = Field(default=None, alias="-n")

    head_limit: int | None = None
    multiline: bool | None = None

    model_config = {"populate_by_name": True}


def _compute_context_window(params: GrepInput) -> tuple[int, int]:
    before = params.B or 0
    after = params.A or 0
    if params.C is not None and params.B is None and params.A is None:
        before = int(params.C)
        after = int(params.C)
    return max(0, int(before)), max(0, int(after))


def _rg_base_args(params: GrepInput) -> list[str]:
    args: list[str] = ["rg", "--color=never", "--no-messages"]
    if params.i:
        args.append("-i")
    if params.glob:
        args.extend(["--glob", params.glob])
    if params.type:
        args.extend(["--type", params.type])
    if params.multiline:
        args.extend(["-U", "--multiline-dotall"])
    return args


def _run_rg_lines(cmd: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.returncode), result.stdout or "", result.stderr or ""


def _grep_fallback(
    *,
    params: GrepInput,
    ctx: SystemToolContext,
    search_path: Path,
    output_mode: str,
) -> dict[str, Any]:
    import re

    before, after = _compute_context_window(params)
    ignore_case = bool(params.i)
    multiline = bool(params.multiline)

    flags = re.MULTILINE
    if ignore_case:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.DOTALL

    try:
        rx = re.compile(params.pattern, flags)
    except Exception as e:
        logger.error(f"Grep pattern 编译失败：{e}", exc_info=True)
        if output_mode == "content":
            return {"matches": [], "total_matches": 0}
        if output_mode == "count":
            return {"counts": [], "total_matches": 0}
        return {"files": [], "count": 0}

    def iter_files() -> list[Path]:
        if search_path.is_file():
            return [search_path]
        if not search_path.exists() or not search_path.is_dir():
            return []

        candidates = [p for p in search_path.rglob("*") if p.is_file()]
        if params.glob:
            pat = str(params.glob)
            filtered: list[Path] = []
            for p in candidates:
                rel = p.relative_to(search_path).as_posix()
                if fnmatch(rel, pat) or fnmatch(p.name, pat):
                    filtered.append(p)
            candidates = filtered
        return candidates

    files = iter_files()

    if output_mode == "files_with_matches":
        matched: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if rx.search(text):
                matched.append(_relpath(f.resolve(), ctx.project_root))
                if params.head_limit is not None and params.head_limit >= 0:
                    if len(matched) >= int(params.head_limit):
                        break
        return {"files": matched, "count": len(matched)}

    if output_mode == "count":
        counts: list[dict[str, Any]] = []
        total = 0
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            c = len(list(rx.finditer(text))) if multiline else sum(1 for line in text.splitlines() if rx.search(line))
            if c <= 0:
                continue
            total += c
            counts.append({"file": _relpath(f.resolve(), ctx.project_root), "count": c})
            if params.head_limit is not None and params.head_limit >= 0:
                if len(counts) >= int(params.head_limit):
                    break
        return {"counts": counts, "total_matches": total}

    # content mode
    matches: list[dict[str, Any]] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = text.splitlines()
        rel_file = _relpath(f.resolve(), ctx.project_root)

        def add_match(line_no: int) -> None:
            ln0 = max(0, int(line_no) - 1)
            line_text = lines[ln0] if ln0 < len(lines) else ""
            match_item = {
                "file": rel_file,
                "line_number": int(line_no) if params.n and line_no is not None else None,
                "line": _truncate_line(line_text, _LINE_TRUNCATE_CHARS),
                "before_context": None,
                "after_context": None,
            }
            if before > 0:
                b0 = max(0, ln0 - before)
                match_item["before_context"] = [
                    _truncate_line(s, _LINE_TRUNCATE_CHARS) for s in lines[b0:ln0]
                ]
            if after > 0:
                a1 = min(len(lines), ln0 + 1 + after)
                match_item["after_context"] = [
                    _truncate_line(s, _LINE_TRUNCATE_CHARS) for s in lines[ln0 + 1 : a1]
                ]
            matches.append(match_item)

        if multiline:
            for m in rx.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                add_match(line_no)
                if params.head_limit is not None and params.head_limit >= 0:
                    if len(matches) >= int(params.head_limit):
                        break
        else:
            for idx, line in enumerate(lines, start=1):
                if rx.search(line):
                    add_match(idx)
                    if params.head_limit is not None and params.head_limit >= 0:
                        if len(matches) >= int(params.head_limit):
                            break

        if params.head_limit is not None and params.head_limit >= 0:
            if len(matches) >= int(params.head_limit):
                break

    return {"matches": matches, "total_matches": len(matches)}


@tool("Search file contents with regex (ripgrep).", name="Grep", usage_rules=GREP_USAGE_RULES)
async def Grep(
    params: GrepInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    output_mode = params.output_mode or "files_with_matches"
    search_path = _resolve_search_path(params.path, ctx.project_root)

    rg_path = shutil.which("rg")
    if not rg_path:
        logger.warning("未找到 rg (ripgrep)，Grep 将使用 Python fallback 实现（性能较差）")
        return _grep_fallback(
            params=params,
            ctx=ctx,
            search_path=search_path,
            output_mode=output_mode,
        )

    if output_mode == "files_with_matches":
        cmd = _rg_base_args(params) + ["-l", params.pattern, str(search_path)]
        code, stdout, stderr = _run_rg_lines(cmd)
        if code == 2:
            logger.error(f"Grep 执行失败：{(stderr or stdout).strip()}")
            return {"files": [], "count": 0}
        if code == 1:
            return {"files": [], "count": 0}

        files = [line.strip() for line in stdout.splitlines() if line.strip()]
        rel_files = [_relpath(Path(f).resolve(), ctx.project_root) for f in files]
        if params.head_limit is not None and params.head_limit >= 0:
            rel_files = rel_files[: int(params.head_limit)]
        return {"files": rel_files, "count": len(rel_files)}

    if output_mode == "count":
        cmd = _rg_base_args(params) + ["--count-matches", params.pattern, str(search_path)]
        code, stdout, stderr = _run_rg_lines(cmd)
        if code == 2:
            logger.error(f"Grep 执行失败：{(stderr or stdout).strip()}")
            return {"counts": [], "total_matches": 0}
        if code == 1:
            return {"counts": [], "total_matches": 0}

        counts: list[dict[str, Any]] = []
        total = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                file_part, count_part = line.rsplit(":", 1)
                c = int(count_part)
                total += c
                counts.append({"file": _relpath(Path(file_part).resolve(), ctx.project_root), "count": c})
            except Exception:
                continue

        if params.head_limit is not None and params.head_limit >= 0:
            counts = counts[: int(params.head_limit)]
        return {"counts": counts, "total_matches": total}

    # content mode
    before, after = _compute_context_window(params)

    # 使用 rg --json 获取行号与原文，避免手工解析
    cmd = _rg_base_args(params) + ["--json", params.pattern, str(search_path)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    matches: list[dict[str, Any]] = []
    internal_line_numbers: list[int | None] = []
    by_file: dict[str, list[int]] = defaultdict(list)  # abs_path -> indices in matches
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if evt.get("type") != "match":
                continue
            data = evt.get("data", {})
            path_text = ((data.get("path") or {}).get("text") or "").strip()
            line_no = data.get("line_number")
            line_text = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
            line_text = _truncate_line(line_text, _LINE_TRUNCATE_CHARS)

            abs_path = str(Path(path_text).resolve()) if path_text else ""
            match_item = {
                "file": _relpath(Path(abs_path), ctx.project_root) if abs_path else "",
                "line_number": int(line_no) if params.n and line_no is not None else None,
                "line": line_text,
                "before_context": None,
                "after_context": None,
            }
            matches.append(match_item)
            internal_line_numbers.append(int(line_no) if line_no is not None else None)
            if abs_path:
                by_file[abs_path].append(len(matches) - 1)

            if params.head_limit is not None and params.head_limit >= 0:
                if len(matches) >= int(params.head_limit):
                    break
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass

    # 读取文件补充上下文
    if (before > 0 or after > 0) and by_file:
        for abs_path, idxs in by_file.items():
            try:
                file_lines = Path(abs_path).read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            total = len(file_lines)
            for i in idxs:
                item = matches[i]
                real_line_no = internal_line_numbers[i]
                if real_line_no is None:
                    continue
                ln0 = int(real_line_no) - 1
                b0 = max(0, ln0 - before)
                a1 = min(total, ln0 + 1 + after)
                before_ctx = [_truncate_line(s, _LINE_TRUNCATE_CHARS) for s in file_lines[b0:ln0]]
                after_ctx = [_truncate_line(s, _LINE_TRUNCATE_CHARS) for s in file_lines[ln0 + 1 : a1]]
                item["before_context"] = before_ctx
                item["after_context"] = after_ctx

    return {"matches": matches, "total_matches": len(matches)}


class _LSInput(BaseModel):
    path: str = Field(description="The absolute path to the directory to list (must be absolute, not relative)")
    ignore: list[str] | None = Field(default=None, description="List of glob patterns to ignore")

    model_config = {"extra": "forbid"}


@tool("Lists files and directories in a given path.", name="LS", usage_rules=LS_USAGE_RULES)
async def LS(
    params: _LSInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        p = _ensure_abs_path(params.path)
        if not p.exists():
            return {"entries": [], "count": 0}
        if not p.is_dir():
            return {"entries": [], "count": 0}

        ignore = params.ignore or []
        entries: list[dict[str, Any]] = []
        for child in sorted(p.iterdir(), key=lambda x: x.name):
            name = child.name
            if any(fnmatch(name, pat) for pat in ignore):
                continue
            item_type = "dir" if child.is_dir() else "file" if child.is_file() else "other"
            size = int(child.stat().st_size) if child.is_file() else 0
            entries.append({"name": name, "type": item_type, "size": size})

        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        msg = f"Error: LS 失败：{e}"
        logger.error(msg, exc_info=True)
        return {"entries": [], "count": 0, "error": msg}


class _TodoItem(BaseModel):
    id: str = Field(description="Unique identifier for the todo item")
    content: str = Field(min_length=1, description="Description of the task")
    status: Literal["pending", "in_progress", "completed"] = Field(description="Current status of the task")
    priority: Literal["high", "medium", "low"] = Field(default="medium", description="Priority level of the task")

    model_config = {"extra": "forbid"}


class _TodoWriteInput(BaseModel):
    todos: list[_TodoItem] = Field(description="The updated todo list")

    model_config = {"extra": "forbid"}


@tool("Create and manage a structured task list for the current session.", name="TodoWrite", usage_rules=TODO_USAGE_RULES)
async def TodoWrite(
    params: _TodoWriteInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> str:
    """创建和管理结构化任务列表

    工具返回时会附加固定提醒文本，强化 Agent 继续执行 TODO 的行为
    """
    try:
        # 1. 序列化 todos
        todos_data = [t.model_dump(mode="json") for t in params.todos]

        has_active = any(t.get("status") in ("pending", "in_progress") for t in todos_data)

        # 2. 更新 ContextIR 的 todo 状态（通过 SystemToolContext 中的 agent_context）
        if ctx.agent_context is not None:
            if has_active:
                ctx.agent_context.set_todo_state(todos_data)
            else:
                ctx.agent_context.set_todo_state([])

        # 3. 可选：持久化到文件（如果有 session_root）
        root = _session_root_from_ctx(ctx)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            todo_path = root / "todos.json"
            if not has_active:
                try:
                    todo_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                data = {
                    "schema_version": 2,
                    "todos": todos_data,
                    "turn_number_at_update": (
                        ctx.agent_context.get_todo_persist_turn_number_at_update()
                        if ctx.agent_context is not None
                        else 0
                    ),
                }
                todo_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 4. 返回成功消息 + 固定提醒
        # 注意：不返回 todos_json，因为会占用 token；TODO 状态通过 ContextIR 管理（必要时会注入温和提醒）
        reminder_text = (
            "\n\n"
            "Remember to keep using the TODO list to keep track of your work "
            "and to now follow the next task on the list. "
            "Mark tasks as 'in_progress' when you start them and 'completed' when done."
        )

        return f"TODO list updated successfully: {len(todos_data)} items.{reminder_text}"

    except Exception as e:
        msg = f"Error: TodoWrite 失败：{e}"
        logger.error(msg, exc_info=True)
        return msg


_WEBFETCH_CACHE: dict[str, dict[str, Any]] = {}


class _WebFetchInput(BaseModel):
    url: str = Field(
        description="The URL to fetch content from",
        json_schema_extra={"format": "uri"},
    )
    prompt: str = Field(description="The prompt to run on the fetched content")

    model_config = {"extra": "forbid"}


@tool("Fetch a URL, convert to markdown, and process with a small model.", name="WebFetch", usage_rules=WEBFETCH_USAGE_RULES)
async def WebFetch(
    params: _WebFetchInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> str:
    url = params.url.strip()
    prompt = params.prompt

    ## make http protocol to https
    if url.startswith("http://"):
        url = "https://" + url[7:]

    now = time.time()
    _cleanup_ttl_cache(_WEBFETCH_CACHE, now=now, ttl=_WEBFETCH_CACHE_TTL_SECONDS)

    cached = _WEBFETCH_CACHE.get(url)
    if cached and (now - float(cached.get("ts", 0.0)) <= _WEBFETCH_CACHE_TTL_SECONDS):
        markdown = str(cached.get("markdown", ""))
        final_url = str(cached.get("final_url", url))
    else:
        try:
            from curl_cffi import requests as crequests
        except Exception as e:
            return f"Error: WebFetch 缺少依赖 curl_cffi：{e}"

        try:
            r = crequests.get(
                url,
                timeout=_WEBFETCH_TIMEOUT_SECONDS,
                impersonate="chrome",
            )
        except Exception as e:
            return f"Error: WebFetch 请求失败：{e}"

        status = int(getattr(r, "status_code", 0) or 0)
        final_url = str(getattr(r, "url", "") or url)
        text = str(getattr(r, "text", "") or "")

        if status != 200:
            preview = _truncate_text(text, 2000)
            return f"Error: HTTP {status} fetching {final_url}\n\n{preview}"

        try:
            from markdownify import markdownify as to_markdown
        except Exception as e:
            return f"Error: WebFetch 缺少依赖 markdownify：{e}"

        markdown = to_markdown(text)
        _WEBFETCH_CACHE[url] = {"ts": now, "markdown": markdown, "final_url": final_url}

    if ctx.llm_levels is None or "LOW" not in ctx.llm_levels:
        return "Error: WebFetch 需要 LOW 级别 LLM（llm_levels['LOW'] 未配置）"
    if ctx.token_cost is None:
        return "Error: WebFetch 需要 token_cost 注入以记录 usage"

    markdown_for_llm = markdown
    truncated_note = ""
    if len(markdown_for_llm) > _WEBFETCH_MARKDOWN_TRUNCATE_CHARS:
        markdown_for_llm = markdown_for_llm[:_WEBFETCH_MARKDOWN_TRUNCATE_CHARS]
        truncated_note = "\n\n(Note: Page content was truncated for context budget.)"

    llm = ctx.llm_levels["LOW"]
    try:
        completion = await asyncio.wait_for(
            llm.ainvoke(
                messages=[
                    UserMessage(
                        content=(
                            f"{prompt}\n\n<url>{final_url}</url>\n\n<content>\n{markdown_for_llm}\n</content>{truncated_note}"
                        )
                    )
                ],
                tools=None,
                tool_choice=None,
            ),
            timeout=_WEBFETCH_LLM_TIMEOUT_SECONDS,
        )
        if completion.usage:
            ctx.token_cost.add_usage(
                str(llm.model),
                completion.usage,
                level="LOW",
                source="webfetch",
            )
        return completion.text
    except asyncio.TimeoutError:
        return f"Error: WebFetch LLM timeout after {_WEBFETCH_LLM_TIMEOUT_SECONDS}s"
    except Exception as e:
        logger.error(f"WebFetch LLM 调用失败：{e}", exc_info=True)
        return f"Error: WebFetch LLM 调用失败：{e}"


class _QuestionOption(BaseModel):
    label: str = Field(max_length=50, description="Display text for this option (1-5 words)")
    description: str = Field(description="Explanation of what this option means or what will happen if chosen")

    model_config = {"extra": "forbid"}


class _Question(BaseModel):
    question: str = Field(description="The complete question to ask the user")
    header: str = Field(max_length=12, description="Short label displayed as a chip/tag (max 12 chars)")
    options: list[_QuestionOption] = Field(min_length=2, max_length=4, description="The available choices (2-4 options)")
    multiSelect: bool = Field(
        default=False,
        description="Set to true to allow the user to select multiple options instead of just one",
    )

    model_config = {"extra": "forbid"}


class _AskUserQuestionInput(BaseModel):
    questions: list[_Question] = Field(min_length=1, max_length=4, description="Questions to ask the user (1-4 questions)")

    model_config = {"extra": "forbid"}


@tool(
    "Ask the user questions during execution to gather input and clarify requirements.",
    name="AskUserQuestion",
    usage_rules=ASKUSERQUESTION_USAGE_RULES,
)
async def AskUserQuestion(
    params: _AskUserQuestionInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """向用户询问问题以收集输入和澄清需求

    此工具会触发特殊的执行流程：
    1. 返回包含问题的 ToolResult
    2. runner_stream 检测到此工具后会 yield UserQuestionEvent
    3. 然后 yield StopEvent(reason='waiting_for_input') 暂停执行
    4. 外部 UI 展示问题，用户回答后通过新的 UserMessage 发送
    """
    try:
        # 序列化问题列表
        questions_data = [q.model_dump(mode="json") for q in params.questions]

        logger.info(f"AskUserQuestion: {len(questions_data)} question(s) prepared for user")

        # 返回状态和问题数据
        # runner_stream 会检测这个返回值并触发特殊流程
        return {
            "questions": questions_data,
            "status": "waiting_for_input",
            "message": f"Prepared {len(questions_data)} question(s) for user. Waiting for user response.",
        }
    except Exception as e:
        msg = f"Error: AskUserQuestion 失败：{e}"
        logger.error(msg, exc_info=True)
        return {"status": "error", "message": msg, "questions": []}


SYSTEM_TOOLS: list[Tool] = [
    Bash,
    Read,
    Write,
    Edit,
    MultiEdit,
    Glob,
    Grep,
    LS,
    TodoWrite,
    WebFetch,
    AskUserQuestion,
]
