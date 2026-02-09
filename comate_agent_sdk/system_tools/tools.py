from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field

from comate_agent_sdk.llm.messages import UserMessage
from comate_agent_sdk.system_tools.artifact_store import ArtifactStore
from comate_agent_sdk.system_tools.description import (
    BASH_USAGE_RULES,
    EDIT_USAGE_RULES,
    GLOB_USAGE_RULES,
    GREP_USAGE_RULES,
    LS_USAGE_RULES,
    MULTIEDIT_USAGE_RULES,
    READ_USAGE_RULES,
    TODO_USAGE_RULES,
    WEBFETCH_USAGE_RULES,
    WRITE_USAGE_RULES,
)
from comate_agent_sdk.system_tools.policy_engine import (
    BASH_COMMAND_POLICY,
    READ_REGISTRY_POLICY,
)
from comate_agent_sdk.system_tools.path_guard import (
    PathGuardError,
    resolve_for_read,
    resolve_for_search,
    resolve_for_write,
    to_safe_relpath,
)
from comate_agent_sdk.system_tools.tool_result import err, ok
from comate_agent_sdk.tools.decorator import Tool, tool
from comate_agent_sdk.tools.depends import Depends
from comate_agent_sdk.tools.system_context import SystemToolContext, get_system_tool_context

logger = logging.getLogger(__name__)

_READ_MIN_LIMIT = 1
_READ_DEFAULT_LIMIT = 500
_READ_MAX_LIMIT = 5000
_LINE_TRUNCATE_CHARS = 2000

_BASH_DEFAULT_TIMEOUT_MS = 120_000
_BASH_MAX_TIMEOUT_MS = 600_000
_BASH_OUTPUT_DEFAULT_MAX_CHARS = 30_000

_WEBFETCH_TIMEOUT_SECONDS = 20
_WEBFETCH_LLM_TIMEOUT_SECONDS = 30
_WEBFETCH_CACHE_TTL_SECONDS = 15 * 60
_WEBFETCH_MARKDOWN_TRUNCATE_CHARS = 50_000

_TEXT_SPILL_CHARS = 10_000
_BYTES_SPILL_LIMIT = 16 * 1024
_LIST_SPILL_LIMIT = 100
_ARTIFACT_TTL_SECONDS = 3600
_GREP_CONTENT_PER_FILE_MAX_COUNT = 1000
_GREP_CONTENT_PARSE_MAX_CHARS = 200_000
_GREP_CONTENT_MATCH_EVENT_LIMIT = 5000

_EXCLUDED_WALK_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".agent"}

_BASH_SEMAPHORES: dict[int, asyncio.Semaphore] = {}
_FILE_LOCKS_BY_LOOP: dict[int, dict[str, asyncio.Lock]] = defaultdict(dict)

_WEBFETCH_CACHE: dict[str, dict[str, Any]] = {}
# Backward-compatible alias for existing tests/introspection.
_READ_REGISTRY_MEMORY = READ_REGISTRY_POLICY.memory


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def _truncate_line(line: str, max_chars: int) -> str:
    if len(line) <= max_chars:
        return line
    return line[:max_chars] + "...(truncated)"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cleanup_ttl_cache(cache: dict[str, dict[str, Any]], *, now: float, ttl: int) -> None:
    expired: list[str] = []
    for k, v in cache.items():
        ts = float(v.get("ts", 0.0))
        if now - ts > ttl:
            expired.append(k)
    for k in expired:
        cache.pop(k, None)


def _project_root(ctx: SystemToolContext) -> Path:
    return ctx.project_root.resolve()


def _workspace_root(ctx: SystemToolContext) -> Path:
    if ctx.workspace_root is not None:
        return ctx.workspace_root.resolve()
    return (_project_root(ctx) / ".agent_workspace").resolve()


def _session_root_from_ctx(ctx: SystemToolContext) -> Path | None:
    if ctx.session_root is None:
        return None
    return ctx.session_root.resolve()


def _allowed_roots(ctx: SystemToolContext) -> list[Path]:
    roots = [_project_root(ctx)]
    roots.append(_workspace_root(ctx))
    return roots


def _artifact_store(ctx: SystemToolContext) -> ArtifactStore:
    return ArtifactStore(_workspace_root(ctx))


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _loop_id() -> int:
    return id(asyncio.get_running_loop())


def _bash_semaphore() -> asyncio.Semaphore:
    lid = _loop_id()
    sem = _BASH_SEMAPHORES.get(lid)
    if sem is None:
        sem = asyncio.Semaphore(4)
        _BASH_SEMAPHORES[lid] = sem
    return sem


def _file_lock(path: Path) -> asyncio.Lock:
    lid = _loop_id()
    key = str(path.resolve())
    lock = _FILE_LOCKS_BY_LOOP[lid].get(key)
    if lock is None:
        lock = asyncio.Lock()
        _FILE_LOCKS_BY_LOOP[lid][key] = lock
    return lock


async def _io_call(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _mark_read(ctx: SystemToolContext, path: Path) -> None:
    await READ_REGISTRY_POLICY.mark_read(
        session_root=_session_root_from_ctx(ctx),
        session_id=ctx.session_id,
        path=path,
    )


async def _require_read(ctx: SystemToolContext, path: Path) -> bool:
    return await READ_REGISTRY_POLICY.has_read(
        session_root=_session_root_from_ctx(ctx),
        session_id=ctx.session_id,
        path=path,
    )


async def _read_text(path: Path, *, encoding: str = "utf-8") -> str:
    return await _io_call(path.read_text, encoding=encoding, errors="replace")


async def _exists(path: Path) -> bool:
    return await _io_call(path.exists)


async def _is_dir(path: Path) -> bool:
    return await _io_call(path.is_dir)


async def _is_file(path: Path) -> bool:
    return await _io_call(path.is_file)


async def _stat(path: Path):
    return await _io_call(path.stat)


async def _resolve_write_path(
    *,
    user_path: str,
    ctx: SystemToolContext,
) -> Path:
    """Resolve write target.

    Policy:
    - New files are created under workspace root.
    - Existing files under project root are writable (for repository edits).
    - Existing files under workspace remain writable.
    """
    workspace_root = _workspace_root(ctx)
    project_root = _project_root(ctx)

    try:
        workspace_target = resolve_for_write(user_path=user_path, workspace_root=workspace_root)
    except PathGuardError as ws_exc:
        # May be an absolute project-root path for existing file edits.
        project_candidate = resolve_for_read(
            user_path=user_path,
            project_root=project_root,
            workspace_root=workspace_root,
        )
        if await _exists(project_candidate):
            return project_candidate
        raise ws_exc

    # Direct workspace target exists: use it.
    if await _exists(workspace_target):
        return workspace_target

    # For relative paths, prefer existing project file so Read/Edit target the same file.
    if not Path(user_path).is_absolute():
        project_candidate = resolve_for_read(
            user_path=user_path,
            project_root=project_root,
            workspace_root=workspace_root,
        )
        if _is_under(project_candidate, project_root) and await _exists(project_candidate):
            return project_candidate

    return workspace_target


def _atomic_write_bytes_sync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        except Exception:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except FileNotFoundError:
                pass


async def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    payload = content.encode(encoding)
    await _io_call(_atomic_write_bytes_sync, path, payload)


def _path_error_result(exc: PathGuardError) -> dict[str, Any]:
    return err(exc.code, exc.message)


def _invalid_argument(message: str, *, field: str | None = None) -> dict[str, Any]:
    field_errors: list[dict[str, Any]] = []
    if field:
        field_errors.append({"field": field, "message": message})
    return err("INVALID_ARGUMENT", message, field_errors=field_errors)


def _validate_bash_command(
    params: BashInput,
    *,
    cwd: Path,
    ctx: SystemToolContext,
) -> dict[str, Any] | None:
    violation = BASH_COMMAND_POLICY.validate(
        args=params.args,
        cwd=cwd,
        allowed_roots=_allowed_roots(ctx),
    )
    if violation is None:
        return None
    return err(violation.code, violation.message)


async def _spill_text(
    *,
    ctx: SystemToolContext,
    namespace: str,
    text: str,
    mime: str,
) -> dict[str, Any]:
    return await _artifact_store(ctx).put_text(
        namespace=namespace,
        text=text,
        mime=mime,
        ttl_seconds=_ARTIFACT_TTL_SECONDS,
    )


async def _spill_json(
    *,
    ctx: SystemToolContext,
    namespace: str,
    data: Any,
) -> dict[str, Any]:
    return await _artifact_store(ctx).put_json(
        namespace=namespace,
        data=data,
        ttl_seconds=_ARTIFACT_TTL_SECONDS,
    )


class BashInput(BaseModel):
    args: list[str] = Field(min_length=1, description="Executable and arguments")
    timeout_ms: int = Field(default=_BASH_DEFAULT_TIMEOUT_MS, ge=1, le=_BASH_MAX_TIMEOUT_MS)
    cwd: str | None = Field(default=None, description="Working directory for command")
    env: dict[str, str] | None = Field(default=None, description="Optional environment variable overrides")
    max_output_chars: int = Field(default=_BASH_OUTPUT_DEFAULT_MAX_CHARS, ge=200, le=200_000)

    model_config = {"extra": "forbid"}


class ReadInput(BaseModel):
    file_path: str = Field(description="Path to text file")
    offset_line: int = Field(
        default=0,
        ge=0,
        description="Line number to start reading from (0-based)",
    )
    limit_lines: int = Field(
        default=_READ_DEFAULT_LIMIT,
        ge=_READ_MIN_LIMIT,
        le=_READ_MAX_LIMIT,
        description="Number of lines to read",
    )
    format: Literal["line_numbers", "raw"] = Field(default="line_numbers")
    max_line_chars: int = Field(default=_LINE_TRUNCATE_CHARS, ge=100, le=20_000)

    model_config = {"extra": "forbid"}


class WriteInput(BaseModel):
    file_path: str = Field(description="Target file path")
    content: str = Field(description="Content to write")
    mode: Literal["overwrite", "append"] = Field(default="overwrite")
    encoding: str = Field(default="utf-8")

    model_config = {"extra": "forbid"}


class EditInput(BaseModel):
    file_path: str = Field(description="Target file path")
    old_string: str = Field(min_length=1, description="Text to replace")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False)

    model_config = {"extra": "forbid"}


class MultiEditOp(BaseModel):
    old_string: str = Field(description="Text to replace")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False)

    model_config = {"extra": "forbid"}


class MultiEditInput(BaseModel):
    file_path: str = Field(description="Target file path")
    edits: list[MultiEditOp] = Field(min_length=1)

    model_config = {"extra": "forbid"}


class GlobInput(BaseModel):
    pattern: str
    path: str | None = None
    head_limit: int = Field(default=_LIST_SPILL_LIMIT, ge=1, le=300)
    include_dirs: bool = False

    model_config = {"extra": "forbid"}


class GrepInput(BaseModel):
    pattern: str
    path: str | None = None
    glob: str | None = None
    type: str | None = None
    output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches"

    B: int | None = Field(default=None, alias="-B", ge=0)
    A: int | None = Field(default=None, alias="-A", ge=0)
    C: int | None = Field(default=None, alias="-C", ge=0)
    i: bool | None = Field(default=None, alias="-i")
    n: bool | None = Field(default=True, alias="-n")

    head_limit: int = Field(default=_LIST_SPILL_LIMIT, ge=1, le=300)
    multiline: bool = False
    max_files: int = Field(default=2000, ge=1, le=100_000)

    model_config = {"populate_by_name": True, "extra": "forbid"}


class LSInput(BaseModel):
    path: str | None = None
    ignore: list[str] | None = None
    head_limit: int = Field(default=_LIST_SPILL_LIMIT, ge=1, le=300)
    include_hidden: bool = False
    sort_by: Literal["name", "mtime", "size"] = "name"

    model_config = {"extra": "forbid"}


class TodoItem(BaseModel):
    id: str = Field(description="Unique identifier for the todo item")
    content: str = Field(min_length=1, description="Description of the task")
    status: Literal["pending", "in_progress", "completed"] = Field(description="Current status of the task")
    priority: Literal["high", "medium", "low"] = Field(default="medium", description="Priority level of the task")

    model_config = {"extra": "forbid"}


class TodoWriteInput(BaseModel):
    todos: list[TodoItem] = Field(description="The updated todo list")

    model_config = {"extra": "forbid"}


class WebFetchInput(BaseModel):
    url: str = Field(
        description="The URL to fetch content from",
        json_schema_extra={"format": "uri"},
    )
    prompt: str = Field(description="The prompt to run on the fetched content")

    model_config = {"extra": "forbid"}


class QuestionOption(BaseModel):
    label: str = Field(max_length=50, description="Display text for this option (1-5 words)")
    description: str = Field(description="Explanation of this option")

    model_config = {"extra": "forbid"}


class Question(BaseModel):
    question: str = Field(description="The complete question to ask the user")
    header: str = Field(max_length=12, description="Short label displayed as a chip/tag (max 12 chars)")
    options: list[QuestionOption] = Field(min_length=2, max_length=4, description="Available options (2-4)")
    multiSelect: bool = Field(default=False)

    model_config = {"extra": "forbid"}


class AskUserQuestionInput(BaseModel):
    questions: list[Question] = Field(min_length=1, max_length=4, description="Questions to ask")

    model_config = {"extra": "forbid"}


def _compute_context_window(params: GrepInput) -> tuple[int, int]:
    before = params.B or 0
    after = params.A or 0
    if params.C is not None and params.B is None and params.A is None:
        before = int(params.C)
        after = int(params.C)
    return max(0, int(before)), max(0, int(after))


def _build_rg_base_args(params: GrepInput) -> list[str]:
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


async def _run_subprocess(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return await _io_call(
        subprocess.run,
        args,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def _append_bounded_text(
    chunks: list[str],
    text: str,
    *,
    current_chars: int,
    max_chars: int,
) -> tuple[int, bool]:
    if max_chars <= 0:
        return current_chars, True
    if current_chars >= max_chars:
        return current_chars, True

    remain = max_chars - current_chars
    if len(text) <= remain:
        chunks.append(text)
        return current_chars + len(text), False

    chunks.append(text[:remain])
    return max_chars, True


@dataclass
class _StreamingSubprocessResult:
    returncode: int
    stdout_capture: str
    stderr_capture: str
    stdout_capture_truncated: bool
    stderr_capture_truncated: bool
    stopped_early: bool


async def _run_subprocess_streaming_lines(
    args: list[str],
    *,
    cwd: Path | None = None,
    on_stdout_line: Callable[[str], bool] | None = None,
    stdout_capture_max_chars: int = _GREP_CONTENT_PARSE_MAX_CHARS,
    stderr_capture_max_chars: int = _TEXT_SPILL_CHARS,
) -> _StreamingSubprocessResult:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_chars = 0
    stderr_chars = 0
    stdout_capture_truncated = False
    stderr_capture_truncated = False
    stopped_early = False

    async def _consume_stdout() -> None:
        nonlocal stdout_chars, stdout_capture_truncated, stopped_early

        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace")

            stdout_chars, clipped = _append_bounded_text(
                stdout_chunks,
                text,
                current_chars=stdout_chars,
                max_chars=stdout_capture_max_chars,
            )
            stdout_capture_truncated = stdout_capture_truncated or clipped

            if stopped_early:
                continue
            if on_stdout_line is None:
                continue

            row = text.rstrip("\n")
            should_continue = bool(on_stdout_line(row))
            if not should_continue:
                stopped_early = True
                if proc.returncode is None:
                    try:
                        proc.terminate()
                    except ProcessLookupError:
                        pass

    async def _consume_stderr() -> None:
        nonlocal stderr_chars, stderr_capture_truncated
        while True:
            raw = await proc.stderr.read(4096)
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace")
            stderr_chars, clipped = _append_bounded_text(
                stderr_chunks,
                text,
                current_chars=stderr_chars,
                max_chars=stderr_capture_max_chars,
            )
            stderr_capture_truncated = stderr_capture_truncated or clipped

    await asyncio.gather(_consume_stdout(), _consume_stderr())
    returncode = await proc.wait()
    return _StreamingSubprocessResult(
        returncode=int(returncode),
        stdout_capture="".join(stdout_chunks),
        stderr_capture="".join(stderr_chunks),
        stdout_capture_truncated=stdout_capture_truncated,
        stderr_capture_truncated=stderr_capture_truncated,
        stopped_early=stopped_early,
    )


def _walk_candidates(
    *,
    search_path: Path,
    include_dirs: bool,
    glob_pattern: str | None,
    max_files: int | None,
) -> list[Path]:
    if not search_path.exists():
        return []

    if search_path.is_file():
        return [search_path]

    candidates: list[Path] = []
    visited_files = 0
    for root, dirs, files in os.walk(search_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_WALK_DIRS]

        root_path = Path(root)
        if include_dirs:
            for d in dirs:
                p = root_path / d
                rel = p.relative_to(search_path).as_posix()
                if glob_pattern and not (fnmatch(rel, glob_pattern) or fnmatch(d, glob_pattern)):
                    continue
                candidates.append(p)

        for f in files:
            p = root_path / f
            rel = p.relative_to(search_path).as_posix()
            if glob_pattern and not (fnmatch(rel, glob_pattern) or fnmatch(f, glob_pattern)):
                continue
            candidates.append(p)
            visited_files += 1
            if max_files is not None and visited_files >= max_files:
                return candidates

    return candidates


def _sort_ls(entries: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "mtime":
        return sorted(entries, key=lambda x: (x["mtime"], x["name"]), reverse=True)
    if sort_by == "size":
        return sorted(entries, key=lambda x: (x["size"], x["name"]), reverse=True)
    return sorted(entries, key=lambda x: x["name"])


@tool(
    "Executes a command using argv args with optional timeout. Returns stdout/stderr and exit code.",
    name="Bash",
    usage_rules=BASH_USAGE_RULES,
)
async def Bash(
    params: BashInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    start = time.monotonic()

    try:
        if params.cwd is not None:
            cwd = resolve_for_read(
                user_path=params.cwd,
                project_root=_project_root(ctx),
                workspace_root=_workspace_root(ctx),
            )
            if not await _exists(cwd):
                return err("NOT_FOUND", f"cwd not found: {params.cwd}")
            if not await _is_dir(cwd):
                return err("INVALID_ARGUMENT", f"cwd is not a directory: {params.cwd}")
        else:
            cwd = _project_root(ctx)
    except PathGuardError as exc:
        return _path_error_result(exc)

    validation_err = _validate_bash_command(params, cwd=cwd, ctx=ctx)
    if validation_err is not None:
        return validation_err

    env = os.environ.copy()
    if params.env:
        env.update(params.env)

    timed_out = False
    cp: subprocess.CompletedProcess[str] | None = None
    stdout = ""
    stderr = ""
    exit_code = -1

    async with _bash_semaphore():
        try:
            cp = await _io_call(
                subprocess.run,
                params.args,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                env=env,
                timeout=params.timeout_ms / 1000.0,
            )
            stdout = cp.stdout or ""
            stderr = cp.stderr or ""
            exit_code = int(cp.returncode)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
            exit_code = -1

    duration_ms = int((time.monotonic() - start) * 1000)

    combined = stdout + stderr
    should_spill = len(combined) > params.max_output_chars or len(combined.encode("utf-8")) > _BYTES_SPILL_LIMIT

    artifact: dict[str, Any] | None = None
    if should_spill:
        artifact = await _spill_json(
            ctx=ctx,
            namespace="bash",
            data={
                "args": params.args,
                "cwd": str(cwd),
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "timed_out": timed_out,
            },
        )

    data = {
        "stdout": _truncate_text(stdout, params.max_output_chars),
        "stderr": _truncate_text(stderr, params.max_output_chars),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "killed": timed_out,
        "duration_ms": duration_ms,
        "truncated": bool(should_spill),
    }
    if artifact is not None:
        data["artifact"] = artifact
        data["read_hint"] = (
            f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
        )

    if timed_out:
        return err(
            "TIMEOUT",
            f"Command timed out after {params.timeout_ms}ms",
            retryable=True,
            meta=data,
        )

    return ok(data=data)


@tool("Reads a text file with line numbers.", name="Read", usage_rules=READ_USAGE_RULES)
async def Read(
    params: ReadInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        path = resolve_for_read(
            user_path=params.file_path,
            project_root=_project_root(ctx),
            workspace_root=_workspace_root(ctx),
        )
    except PathGuardError as exc:
        return _path_error_result(exc)

    if not await _exists(path):
        return err("NOT_FOUND", f"File not found: {params.file_path}")
    if await _is_dir(path):
        return err("IS_DIRECTORY", f"Path is a directory: {params.file_path}")
    if not await _is_file(path):
        return err("PERMISSION_DENIED", f"Path is not a regular file: {params.file_path}")

    text = await _read_text(path)
    await _mark_read(ctx, path)
    lines = text.splitlines()
    total_lines = len(lines)

    start = params.offset_line
    end = min(total_lines, start + params.limit_lines)
    sliced = lines[start:end]

    line_truncated = False
    if params.format == "line_numbers":
        rendered_lines: list[str] = []
        for i, line in enumerate(sliced, start=start):
            clipped = _truncate_line(line, params.max_line_chars)
            if clipped != line:
                line_truncated = True
            rendered_lines.append(f"{i + 1:6d}\t{clipped}")
        content = "\n".join(rendered_lines)
    else:
        clipped_lines = []
        for line in sliced:
            clipped = _truncate_line(line, params.max_line_chars)
            if clipped != line:
                line_truncated = True
            clipped_lines.append(clipped)
        content = "\n".join(clipped_lines)

    has_more = end < total_lines
    output_too_large = len(content) > _TEXT_SPILL_CHARS or len(content.encode("utf-8")) > _BYTES_SPILL_LIMIT

    data: dict[str, Any] = {
        "content": _truncate_text(content, _TEXT_SPILL_CHARS),
        "total_lines": total_lines,
        "lines_returned": len(sliced),
        "has_more": has_more,
        "truncated": bool(line_truncated or output_too_large),
        "meta": {
            "encoding": "utf-8",
            "file_bytes": int((await _stat(path)).st_size),
        },
    }

    if line_truncated or output_too_large:
        artifact = await _spill_text(
            ctx=ctx,
            namespace="read",
            text=content,
            mime="text/plain",
        )
        data["artifact"] = artifact
        data["read_hint"] = (
            f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
        )

    if has_more:
        data["next_offset_line"] = end

    return ok(data=data)


@tool("Writes content to a file.", name="Write", usage_rules=WRITE_USAGE_RULES)
async def Write(
    params: WriteInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        path = await _resolve_write_path(user_path=params.file_path, ctx=ctx)
    except PathGuardError as exc:
        return _path_error_result(exc)

    root_refs = _allowed_roots(ctx)
    async with _file_lock(path):
        existed = await _exists(path)
        if existed and await _is_dir(path):
            return err("IS_DIRECTORY", f"Path is a directory: {params.file_path}")
        if existed and not await _is_file(path):
            return err("PERMISSION_DENIED", f"Path is not a regular file: {params.file_path}")
        if existed and not await _require_read(ctx, path):
            return err("PRECONDITION_FAILED", f"Must Read file before modifying it: {params.file_path}")

        if params.mode == "append" and existed:
            original = await _read_text(path, encoding=params.encoding)
            final_content = original + params.content
        elif params.mode == "append" and not existed:
            final_content = params.content
        else:
            final_content = params.content

        await _atomic_write_text(path, final_content, encoding=params.encoding)

    final_bytes = len(final_content.encode(params.encoding, errors="replace"))
    op_bytes = len(params.content.encode(params.encoding, errors="replace"))
    relpath = to_safe_relpath(path, roots=root_refs)
    return ok(
        data={
            "bytes_written": op_bytes,
            "file_bytes": final_bytes,
            "created": not existed,
            "sha256": _sha256_text(final_content),
            "relpath": relpath,
        }
    )


@tool("Performs exact string replacement in a file.", name="Edit", usage_rules=EDIT_USAGE_RULES)
async def Edit(
    params: EditInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    if params.old_string == params.new_string:
        return _invalid_argument("new_string must differ from old_string", field="new_string")

    try:
        path = await _resolve_write_path(user_path=params.file_path, ctx=ctx)
    except PathGuardError as exc:
        return _path_error_result(exc)

    root_refs = _allowed_roots(ctx)
    async with _file_lock(path):
        if not await _exists(path):
            return err("NOT_FOUND", f"File not found: {params.file_path}")
        if await _is_dir(path):
            return err("IS_DIRECTORY", f"Path is a directory: {params.file_path}")
        if not await _is_file(path):
            return err("PERMISSION_DENIED", f"Path is not a regular file: {params.file_path}")
        if not await _require_read(ctx, path):
            return err("PRECONDITION_FAILED", f"Must Read file before modifying it: {params.file_path}")

        before = await _read_text(path)
        count = before.count(params.old_string)
        if count == 0:
            return err("CONFLICT", f"old_string not found in file: {params.file_path}")

        if not params.replace_all and count > 1:
            return err(
                "CONFLICT",
                f"old_string appears {count} times; provide a more specific old_string or set replace_all=true",
            )

        if params.replace_all:
            after = before.replace(params.old_string, params.new_string)
            replacements = count
        else:
            after = before.replace(params.old_string, params.new_string, 1)
            replacements = 1

        await _atomic_write_text(path, after)

    return ok(
        data={
            "replacements": replacements,
            "before_sha256": _sha256_text(before),
            "after_sha256": _sha256_text(after),
            "relpath": to_safe_relpath(path, roots=root_refs),
        }
    )


@tool(
    "Make multiple edits to a single file atomically (all succeed or none).",
    name="MultiEdit",
    usage_rules=MULTIEDIT_USAGE_RULES,
)
async def MultiEdit(
    params: MultiEditInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        path = await _resolve_write_path(user_path=params.file_path, ctx=ctx)
    except PathGuardError as exc:
        return _path_error_result(exc)

    root_refs = _allowed_roots(ctx)
    async with _file_lock(path):
        exists = await _exists(path)

        if exists:
            if await _is_dir(path):
                return err("IS_DIRECTORY", f"Path is a directory: {params.file_path}")
            if not await _is_file(path):
                return err("PERMISSION_DENIED", f"Path is not a regular file: {params.file_path}")
            if not await _require_read(ctx, path):
                return err("PRECONDITION_FAILED", f"Must Read file before modifying it: {params.file_path}")
            content = await _read_text(path)
            before_hash = _sha256_text(content)
            edits = params.edits
            created = False
        else:
            edits = list(params.edits)
            if edits[0].old_string != "":
                return err(
                    "NOT_FOUND",
                    "File does not exist; first edit.old_string must be empty to create a file",
                )
            content = edits[0].new_string
            before_hash = None
            edits = edits[1:]
            created = True

        total_replacements = 0
        for idx, op in enumerate(edits):
            if op.old_string == "":
                return err(
                    "INVALID_ARGUMENT",
                    f"edits[{idx}] old_string cannot be empty",
                )
            if op.old_string == op.new_string:
                return err(
                    "INVALID_ARGUMENT",
                    f"edits[{idx}] new_string must differ from old_string",
                )

            count = content.count(op.old_string)
            if count == 0:
                return err(
                    "CONFLICT",
                    f"edits[{idx}] old_string not found in file",
                )

            if not op.replace_all and count > 1:
                return err(
                    "CONFLICT",
                    f"edits[{idx}] old_string appears {count} times; use replace_all=true or provide more context",
                )

            if op.replace_all:
                content = content.replace(op.old_string, op.new_string)
                total_replacements += count
            else:
                content = content.replace(op.old_string, op.new_string, 1)
                total_replacements += 1

        await _atomic_write_text(path, content)

    return ok(
        data={
            "total_replacements": total_replacements,
            "created": created,
            "before_sha256": before_hash,
            "after_sha256": _sha256_text(content),
            "bytes": len(content.encode("utf-8")),
            "relpath": to_safe_relpath(path, roots=root_refs),
        }
    )


@tool("Find files matching a glob pattern.", name="Glob", usage_rules=GLOB_USAGE_RULES)
async def Glob(
    params: GlobInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        search_dir = resolve_for_search(
            user_path=params.path,
            project_root=_project_root(ctx),
            workspace_root=_workspace_root(ctx),
        )
    except PathGuardError as exc:
        return _path_error_result(exc)

    if not await _exists(search_dir):
        return err("NOT_FOUND", f"Search path not found: {params.path or str(search_dir)}")

    candidates = await _io_call(
        _walk_candidates,
        search_path=search_dir,
        include_dirs=params.include_dirs,
        glob_pattern=params.pattern,
        max_files=None,
    )

    root_refs = _allowed_roots(ctx)
    matches = [
        to_safe_relpath(p.resolve(), roots=root_refs)
        for p in candidates
        if params.include_dirs or p.is_file()
    ]
    matches = sorted(matches)

    total = len(matches)
    preview = matches[: params.head_limit]
    truncated = total > len(preview)

    data: dict[str, Any] = {
        "matches": preview,
        "count": total,
        "search_path": to_safe_relpath(search_dir, roots=root_refs),
        "truncated": truncated,
    }

    if truncated:
        artifact = await _spill_json(
            ctx=ctx,
            namespace="glob",
            data={"matches": matches, "count": total},
        )
        data["artifact"] = artifact
        data["read_hint"] = (
            f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
        )

    return ok(data=data)


def _grep_fallback(
    *,
    params: GrepInput,
    search_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    import re

    before, after = _compute_context_window(params)
    flags = re.MULTILINE
    if params.i:
        flags |= re.IGNORECASE
    if params.multiline:
        flags |= re.DOTALL

    try:
        rx = re.compile(params.pattern, flags)
    except Exception as exc:
        return err("INVALID_ARGUMENT", f"invalid regex pattern: {exc}")

    files = _walk_candidates(
        search_path=search_path,
        include_dirs=False,
        glob_pattern=params.glob,
        max_files=params.max_files,
    )
    files = [p for p in files if p.is_file()]

    if params.output_mode == "files_with_matches":
        matched: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if rx.search(text):
                matched.append(to_safe_relpath(f.resolve(), roots=[project_root]))
        total = len(matched)
        preview = matched[: params.head_limit]
        return ok(data={"files": preview, "count": total, "truncated": total > len(preview)})

    if params.output_mode == "count":
        counts: list[dict[str, Any]] = []
        total_matches = 0
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            c = len(list(rx.finditer(text))) if params.multiline else sum(1 for line in text.splitlines() if rx.search(line))
            if c <= 0:
                continue
            total_matches += c
            counts.append({"file": to_safe_relpath(f.resolve(), roots=[project_root]), "count": c})

        total = len(counts)
        preview = counts[: params.head_limit]
        return ok(
            data={
                "counts": preview,
                "total_matches": total_matches,
                "truncated": total > len(preview),
            }
        )

    matches: list[dict[str, Any]] = []
    total_matches = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = text.splitlines()
        rel_file = to_safe_relpath(f.resolve(), roots=[project_root])
        for idx, line in enumerate(lines, start=1):
            if not rx.search(line):
                continue
            total_matches += 1
            if len(matches) >= params.head_limit:
                continue

            item: dict[str, Any] = {
                "file": rel_file,
                "line_number": idx if params.n else None,
                "line": _truncate_line(line, _LINE_TRUNCATE_CHARS),
                "before_context": None,
                "after_context": None,
            }
            if before > 0:
                b0 = max(0, idx - 1 - before)
                item["before_context"] = [_truncate_line(s, _LINE_TRUNCATE_CHARS) for s in lines[b0 : idx - 1]]
            if after > 0:
                a1 = min(len(lines), idx + after)
                item["after_context"] = [_truncate_line(s, _LINE_TRUNCATE_CHARS) for s in lines[idx:a1]]

            matches.append(item)

    return ok(
        data={
            "matches": matches,
            "total_matches": total_matches,
            "truncated": total_matches > len(matches),
        }
    )


@tool("Search file contents with regex (ripgrep).", name="Grep", usage_rules=GREP_USAGE_RULES)
async def Grep(
    params: GrepInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        search_path = resolve_for_search(
            user_path=params.path,
            project_root=_project_root(ctx),
            workspace_root=_workspace_root(ctx),
        )
    except PathGuardError as exc:
        return _path_error_result(exc)

    if not await _exists(search_path):
        return err("NOT_FOUND", f"Search path not found: {params.path or str(search_path)}")

    rg_path = shutil.which("rg")
    if not rg_path:
        logger.warning("ripgrep not found; Grep will use Python fallback")
        return _grep_fallback(
            params=params,
            search_path=search_path,
            project_root=_project_root(ctx),
        )

    root_refs = _allowed_roots(ctx)

    if params.output_mode == "files_with_matches":
        cmd = _build_rg_base_args(params) + ["-l", params.pattern, str(search_path)]
        result = await _run_subprocess(cmd)
        if result.returncode == 2:
            return err("INVALID_ARGUMENT", (result.stderr or result.stdout).strip() or "rg failed")
        if result.returncode == 1:
            return ok(data={"files": [], "count": 0, "truncated": False})

        rel_files = [
            to_safe_relpath(Path(line.strip()).resolve(), roots=root_refs)
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        total = len(rel_files)
        preview = rel_files[: params.head_limit]
        data: dict[str, Any] = {
            "files": preview,
            "count": total,
            "truncated": total > len(preview),
        }
        if total > len(preview):
            artifact = await _spill_json(
                ctx=ctx,
                namespace="grep_files",
                data={"files": rel_files, "count": total},
            )
            data["artifact"] = artifact
            data["read_hint"] = (
                f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
            )
        return ok(data=data)

    if params.output_mode == "count":
        cmd = _build_rg_base_args(params) + ["--count-matches", params.pattern, str(search_path)]
        result = await _run_subprocess(cmd)
        if result.returncode == 2:
            return err("INVALID_ARGUMENT", (result.stderr or result.stdout).strip() or "rg failed")
        if result.returncode == 1:
            return ok(data={"counts": [], "total_matches": 0, "truncated": False})

        counts: list[dict[str, Any]] = []
        total_matches = 0
        for line in result.stdout.splitlines():
            row = line.strip()
            if not row:
                continue
            try:
                file_part, count_part = row.rsplit(":", 1)
                c = int(count_part)
            except Exception:
                continue
            total_matches += c
            counts.append(
                {
                    "file": to_safe_relpath(Path(file_part).resolve(), roots=root_refs),
                    "count": c,
                }
            )

        total = len(counts)
        preview = counts[: params.head_limit]
        data = {
            "counts": preview,
            "total_matches": total_matches,
            "truncated": total > len(preview),
        }
        if total > len(preview):
            artifact = await _spill_json(
                ctx=ctx,
                namespace="grep_count",
                data={"counts": counts, "total_matches": total_matches},
            )
            data["artifact"] = artifact
            data["read_hint"] = (
                f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
            )
        return ok(data=data)

    before, after = _compute_context_window(params)
    cmd = _build_rg_base_args(params) + [
        "--json",
        "--max-count",
        str(_GREP_CONTENT_PER_FILE_MAX_COUNT),
        params.pattern,
        str(search_path),
    ]
    matches: list[dict[str, Any]] = []
    line_refs: list[tuple[str, int]] = []
    total_matches = 0
    lower_bound = False

    def _on_rg_json_line(raw_line: str) -> bool:
        nonlocal total_matches, lower_bound
        row = raw_line.strip()
        if not row:
            return True

        try:
            evt = json.loads(row)
        except json.JSONDecodeError:
            return True
        if evt.get("type") != "match":
            return True

        data = evt.get("data", {})
        path_text = ((data.get("path") or {}).get("text") or "").strip()
        line_no = data.get("line_number")
        line_text = ((data.get("lines") or {}).get("text") or "").rstrip("\n")

        total_matches += 1
        if total_matches >= _GREP_CONTENT_MATCH_EVENT_LIMIT:
            lower_bound = True
            return False
        if len(matches) >= params.head_limit:
            return True

        abs_path = str(Path(path_text).resolve()) if path_text else ""
        rel_file = to_safe_relpath(Path(abs_path), roots=root_refs) if abs_path else ""
        ln = int(line_no) if line_no is not None else 0

        matches.append(
            {
                "file": rel_file,
                "line_number": ln if params.n and line_no is not None else None,
                "line": _truncate_line(line_text, _LINE_TRUNCATE_CHARS),
                "before_context": None,
                "after_context": None,
            }
        )
        line_refs.append((abs_path, ln))
        return True

    stream = await _run_subprocess_streaming_lines(
        cmd,
        on_stdout_line=_on_rg_json_line,
        stdout_capture_max_chars=_GREP_CONTENT_PARSE_MAX_CHARS,
        stderr_capture_max_chars=_TEXT_SPILL_CHARS,
    )

    if stream.returncode == 2:
        return err("INVALID_ARGUMENT", (stream.stderr_capture or stream.stdout_capture).strip() or "rg failed")
    if stream.returncode == 1 and total_matches == 0:
        return ok(data={"matches": [], "total_matches": 0, "truncated": False})
    if stream.returncode not in {0, 1} and not stream.stopped_early:
        return err("INTERNAL", (stream.stderr_capture or stream.stdout_capture).strip() or "rg failed", retryable=True)

    if stream.stopped_early:
        lower_bound = True

    if (before > 0 or after > 0) and matches:
        by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, (abs_path, line_no) in enumerate(line_refs):
            if not abs_path or line_no <= 0:
                continue
            by_file[abs_path].append((idx, line_no))

        for abs_path, refs in by_file.items():
            try:
                file_lines = await _read_text(Path(abs_path))
            except Exception:
                continue
            lines = file_lines.splitlines()
            total_lines = len(lines)
            for idx, line_no in refs:
                ln0 = line_no - 1
                b0 = max(0, ln0 - before)
                a1 = min(total_lines, ln0 + 1 + after)
                matches[idx]["before_context"] = [_truncate_line(s, _LINE_TRUNCATE_CHARS) for s in lines[b0:ln0]]
                matches[idx]["after_context"] = [_truncate_line(s, _LINE_TRUNCATE_CHARS) for s in lines[ln0 + 1 : a1]]

    truncated = total_matches > len(matches) or lower_bound
    data = {
        "matches": matches,
        "total_matches": total_matches,
        "truncated": truncated,
        "total_matches_is_lower_bound": lower_bound,
    }

    raw_capture = stream.stdout_capture
    if truncated or stream.stdout_capture_truncated or len(raw_capture.encode("utf-8")) > _BYTES_SPILL_LIMIT:
        artifact = await _spill_text(
            ctx=ctx,
            namespace="grep_content",
            text=raw_capture,
            mime="text/plain",
        )
        data["artifact"] = artifact
        data["read_hint"] = (
            f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
        )
        if stream.stdout_capture_truncated:
            data["raw_output_truncated"] = True

    return ok(data=data)


@tool("Lists files and directories in a given path.", name="LS", usage_rules=LS_USAGE_RULES)
async def LS(
    params: LSInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        p = resolve_for_search(
            user_path=params.path,
            project_root=_project_root(ctx),
            workspace_root=_workspace_root(ctx),
        )
    except PathGuardError as exc:
        return _path_error_result(exc)

    if not await _exists(p):
        return err("NOT_FOUND", f"Path not found: {params.path or str(p)}")
    if not await _is_dir(p):
        return err("INVALID_ARGUMENT", f"Path is not a directory: {params.path or str(p)}")

    ignore = params.ignore or []

    def _collect_entries(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for child in path.iterdir():
            name = child.name
            if not params.include_hidden and name.startswith("."):
                continue
            if any(fnmatch(name, pat) for pat in ignore):
                continue

            item_type = "dir" if child.is_dir() else "file" if child.is_file() else "other"
            try:
                st = child.stat()
                size = int(st.st_size) if child.is_file() else 0
                mtime = int(st.st_mtime)
            except Exception:
                size = 0
                mtime = 0

            rows.append({
                "name": name,
                "type": item_type,
                "size": size,
                "mtime": mtime,
            })
        return rows

    entries = await _io_call(_collect_entries, p)
    entries = _sort_ls(entries, params.sort_by)

    total = len(entries)
    preview = entries[: params.head_limit]
    data: dict[str, Any] = {
        "entries": preview,
        "count": total,
        "truncated": total > len(preview),
        "path": to_safe_relpath(p, roots=_allowed_roots(ctx)),
    }

    if total > len(preview):
        artifact = await _spill_json(
            ctx=ctx,
            namespace="ls",
            data={"entries": entries, "count": total},
        )
        data["artifact"] = artifact
        data["read_hint"] = (
            f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
        )

    return ok(data=data)


@tool("Create and manage a structured task list for the current session.", name="TodoWrite", usage_rules=TODO_USAGE_RULES)
async def TodoWrite(
    params: TodoWriteInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    reminder_text = (
        "Remember to keep using the TODO list to keep track of your work and "
        "to now follow the next task on the list. Mark tasks as in_progress "
        "when you start them and completed when done."
    )

    try:
        todos_data = [t.model_dump(mode="json") for t in params.todos]
        active_count = sum(1 for t in todos_data if t.get("status") in ("pending", "in_progress"))
        has_active = active_count > 0

        if ctx.agent_context is not None:
            if has_active:
                ctx.agent_context.set_todo_state(todos_data)
            else:
                ctx.agent_context.set_todo_state([])

        persisted = False
        todo_path_rel: str | None = None

        root = _session_root_from_ctx(ctx)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            todo_path = root / "todos.json"
            todo_path_rel = "todos.json"

            if not has_active:
                def _remove_todo() -> None:
                    try:
                        todo_path.unlink()
                    except FileNotFoundError:
                        pass

                await _io_call(_remove_todo)
            else:
                payload = {
                    "schema_version": 2,
                    "todos": todos_data,
                    "turn_number_at_update": (
                        ctx.agent_context.get_todo_persist_turn_number_at_update()
                        if ctx.agent_context is not None
                        else 0
                    ),
                }
                await _atomic_write_text(
                    todo_path,
                    json.dumps(payload, ensure_ascii=False, indent=2),
                )
                persisted = True

        return ok(
            data={
                "count": len(todos_data),
                "active_count": active_count,
                "persisted": persisted,
                "todo_path": todo_path_rel,
            },
            message=reminder_text,
        )
    except Exception as exc:
        logger.error(f"TodoWrite failed: {exc}", exc_info=True)
        return err("INTERNAL", f"TodoWrite failed: {exc}")


@tool("Fetch a URL, convert to markdown, and process with a small model.", name="WebFetch", usage_rules=WEBFETCH_USAGE_RULES)
async def WebFetch(
    params: WebFetchInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    url = params.url.strip()
    prompt = params.prompt

    if not url:
        return _invalid_argument("url cannot be empty", field="url")
    if url.startswith("http://"):
        url = f"https://{url[7:]}"

    now = time.time()
    _cleanup_ttl_cache(_WEBFETCH_CACHE, now=now, ttl=_WEBFETCH_CACHE_TTL_SECONDS)

    cached = _WEBFETCH_CACHE.get(url)
    if cached and (now - float(cached.get("ts", 0.0)) <= _WEBFETCH_CACHE_TTL_SECONDS):
        markdown = str(cached.get("markdown", ""))
        final_url = str(cached.get("final_url", url))
        status = int(cached.get("status", 200))
        cached_hit = True
    else:
        cached_hit = False
        try:
            from curl_cffi import requests as crequests
        except Exception as exc:
            return err("TOOL_NOT_AVAILABLE", f"WebFetch missing dependency curl_cffi: {exc}")

        try:
            response = await _io_call(
                crequests.get,
                url,
                timeout=_WEBFETCH_TIMEOUT_SECONDS,
                impersonate="chrome",
            )
        except Exception as exc:
            return err("INTERNAL", f"WebFetch request failed: {exc}", retryable=True)

        status = int(getattr(response, "status_code", 0) or 0)
        final_url = str(getattr(response, "url", "") or url)
        html_text = str(getattr(response, "text", "") or "")

        if status != 200:
            preview = _truncate_text(html_text, 2000)
            code = "NOT_FOUND" if status == 404 else "INTERNAL"
            return err(
                code,
                f"HTTP {status} fetching {final_url}",
                meta={"status": status, "preview": preview},
            )

        try:
            from markdownify import markdownify as to_markdown
        except Exception as exc:
            return err("TOOL_NOT_AVAILABLE", f"WebFetch missing dependency markdownify: {exc}")

        markdown = await _io_call(to_markdown, html_text)
        _WEBFETCH_CACHE[url] = {
            "ts": now,
            "markdown": markdown,
            "final_url": final_url,
            "status": status,
        }

    artifact = await _spill_text(
        ctx=ctx,
        namespace="webfetch",
        text=markdown,
        mime="text/markdown",
    )

    if ctx.llm_levels is None or "LOW" not in ctx.llm_levels:
        return err("TOOL_NOT_AVAILABLE", "WebFetch requires llm_levels['LOW']")

    markdown_for_llm = markdown
    truncated_for_llm = False
    if len(markdown_for_llm) > _WEBFETCH_MARKDOWN_TRUNCATE_CHARS:
        markdown_for_llm = markdown_for_llm[:_WEBFETCH_MARKDOWN_TRUNCATE_CHARS]
        truncated_for_llm = True

    llm = ctx.llm_levels["LOW"]
    try:
        completion = await asyncio.wait_for(
            llm.ainvoke(
                messages=[
                    UserMessage(
                        content=(
                            f"{prompt}\n\n<url>{final_url}</url>\n\n"
                            f"<content>\n{markdown_for_llm}\n</content>"
                        )
                    )
                ],
                tools=None,
                tool_choice=None,
            ),
            timeout=_WEBFETCH_LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return err(
            "TIMEOUT",
            f"WebFetch LLM timeout after {_WEBFETCH_LLM_TIMEOUT_SECONDS}s",
            retryable=True,
        )
    except Exception as exc:
        logger.error(f"WebFetch LLM call failed: {exc}", exc_info=True)
        return err("INTERNAL", f"WebFetch LLM call failed: {exc}", retryable=True)

    if completion.usage and ctx.token_cost is not None:
        source = "webfetch"
        if ctx.subagent_name:
            source = f"subagent:{ctx.subagent_name}:webfetch"
        ctx.token_cost.add_usage(
            str(llm.model),
            completion.usage,
            level="LOW",
            source=source,
        )

    return ok(
        data={
            "final_url": final_url,
            "status": status,
            "cached": cached_hit,
            "artifact": artifact,
            "truncated_for_llm": truncated_for_llm,
            "summary_text": completion.text,
            "model_used": str(llm.model),
            "read_hint": (
                f"Use Read with file_path='{artifact['relpath']}', format='raw', offset_line=0, limit_lines=500"
            ),
        }
    )


@tool(
    "Use this tool when you need to ask the user questions during execution.",
    name="AskUserQuestion",
)
async def AskUserQuestion(
    params: AskUserQuestionInput,
    ctx: Annotated[SystemToolContext, Depends(get_system_tool_context)] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    try:
        questions_data = [q.model_dump(mode="json") for q in params.questions]
        logger.info(f"AskUserQuestion prepared {len(questions_data)} question(s)")
        return ok(
            data={
                "status": "waiting_for_input",
                "questions": questions_data,
            },
            message=f"Prepared {len(questions_data)} question(s) for user",
        )
    except Exception as exc:
        logger.error(f"AskUserQuestion failed: {exc}", exc_info=True)
        return err("INTERNAL", f"AskUserQuestion failed: {exc}")


# Backward-compatible aliases for tests/importers.
_TodoItem = TodoItem
_TodoWriteInput = TodoWriteInput


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
