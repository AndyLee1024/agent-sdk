from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from bu_agent_sdk.tools.decorator import Tool, tool
from bu_agent_sdk.tools.depends import Depends
from bu_agent_sdk.tools.system_context import SystemToolContext, get_system_tool_context

logger = logging.getLogger(__name__)

_READ_DEFAULT_LIMIT = 500
_LINE_TRUNCATE_CHARS = 2000
_BASH_DEFAULT_TIMEOUT_MS = 120_000
_BASH_MAX_TIMEOUT_MS = 600_000
_BASH_OUTPUT_TRUNCATE_CHARS = 30_000


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


@tool(
    "Executes a bash command with optional timeout. Returns combined output and exit code.",
    name="Bash",
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


@tool("Reads a text file with line numbers.", name="Read")
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


@tool("Writes content to a file (overwrites).", name="Write")
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


@tool("Performs exact string replacement in a file.", name="Edit")
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


@tool("Find files matching a glob pattern.", name="Glob")
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


@tool("Search file contents with regex (ripgrep).", name="Grep")
async def Grep(
    params: GrepInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    output_mode = params.output_mode or "files_with_matches"
    search_path = _resolve_search_path(params.path, ctx.project_root)

    rg_path = shutil.which("rg")
    if not rg_path:
        logger.error("未找到 rg (ripgrep)，Grep 工具不可用")
        if output_mode == "content":
            return {"matches": [], "total_matches": 0}
        if output_mode == "count":
            return {"counts": [], "total_matches": 0}
        return {"files": [], "count": 0}

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


SYSTEM_TOOLS: list[Tool] = [Bash, Read, Write, Edit, Glob, Grep]
