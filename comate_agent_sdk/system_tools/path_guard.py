from __future__ import annotations

from pathlib import Path
from typing import Iterable

from comate_agent_sdk.system_tools.tool_result import ToolErrorCode

_SENSITIVE_PATHS: tuple[Path, ...] = (
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/etc"),
)


class PathGuardError(Exception):
    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ToolErrorCode = code
        self.message = message


def _ensure_not_sensitive(path: Path) -> None:
    for prefix in _SENSITIVE_PATHS:
        try:
            path.relative_to(prefix)
            raise PathGuardError(
                "PERMISSION_DENIED",
                f"path {path} is under restricted system directory {prefix}",
            )
        except ValueError:
            continue


def _ensure_under_roots(path: Path, roots: Iterable[Path]) -> None:
    resolved_roots = [r.resolve() for r in roots]
    for root in resolved_roots:
        try:
            path.relative_to(root)
            return
        except ValueError:
            continue
    roots_text = ", ".join(str(r) for r in resolved_roots)
    raise PathGuardError(
        "PATH_ESCAPE",
        f"path {path} is outside allowed roots: {roots_text}",
    )


def _ensure_no_symlink_traversal(path: Path, root: Path) -> None:
    root_resolved = root.resolve()
    try:
        rel_parts = path.relative_to(root_resolved).parts
    except ValueError as exc:
        raise PathGuardError("PATH_ESCAPE", f"path {path} is outside root {root_resolved}") from exc

    cursor = root_resolved
    for part in rel_parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise PathGuardError(
                "PATH_ESCAPE",
                f"path segment {cursor} is a symbolic link and is not allowed",
            )


def resolve_for_read(*, user_path: str, project_root: Path, workspace_root: Path | None) -> Path:
    if not user_path or not user_path.strip():
        raise PathGuardError("INVALID_ARGUMENT", "file_path cannot be empty")

    project_root_resolved = project_root.resolve()
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = project_root_resolved / candidate
    resolved = candidate.resolve()

    allowed_roots: list[Path] = [project_root_resolved]
    if workspace_root is not None:
        allowed_roots.append(workspace_root.resolve())

    _ensure_not_sensitive(resolved)
    _ensure_under_roots(resolved, allowed_roots)
    return resolved


def resolve_for_search(
    *,
    user_path: str | None,
    project_root: Path,
    workspace_root: Path | None,
) -> Path:
    if user_path is None:
        return project_root.resolve()
    return resolve_for_read(
        user_path=user_path,
        project_root=project_root,
        workspace_root=workspace_root,
    )


def resolve_for_write(*, user_path: str, workspace_root: Path) -> Path:
    if not user_path or not user_path.strip():
        raise PathGuardError("INVALID_ARGUMENT", "file_path cannot be empty")

    root = workspace_root.resolve()
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    _ensure_not_sensitive(resolved)
    _ensure_under_roots(resolved, [root])
    _ensure_no_symlink_traversal(resolved, root)
    return resolved


def to_safe_relpath(path: Path, *, roots: Iterable[Path]) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            rel = resolved.relative_to(root.resolve())
            return rel.as_posix()
        except Exception:
            continue
    return str(resolved.name)
