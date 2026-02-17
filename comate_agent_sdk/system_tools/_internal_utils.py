"""Internal utilities for system tools.

This module contains low-level helper functions extracted from tools.py
to keep the main module size under control (<700 LOC guideline).
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from comate_agent_sdk.system_tools.artifact_store import ArtifactStore


# ────────────────────────────────────────────────────────────────────────────
# Global state (loop-local locks and semaphores)
# ────────────────────────────────────────────────────────────────────────────
_FILE_LOCKS_BY_LOOP: dict[int, dict[str, asyncio.Lock]] = defaultdict(dict)
_BASH_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def loop_id() -> int:
    """Get the ID of the current running event loop."""
    return id(asyncio.get_running_loop())


def file_lock(path: Path) -> asyncio.Lock:
    """Get or create a per-file lock for the current event loop."""
    lid = loop_id()
    key = str(path.resolve())
    lock = _FILE_LOCKS_BY_LOOP[lid].get(key)
    if lock is None:
        lock = asyncio.Lock()
        _FILE_LOCKS_BY_LOOP[lid][key] = lock
    return lock


def bash_semaphore() -> asyncio.Semaphore:
    """Get or create a bash semaphore (max 4 concurrent) for the current event loop."""
    lid = loop_id()
    sem = _BASH_SEMAPHORES.get(lid)
    if sem is None:
        sem = asyncio.Semaphore(4)
        _BASH_SEMAPHORES[lid] = sem
    return sem


# ────────────────────────────────────────────────────────────────────────────
# Hash utilities
# ────────────────────────────────────────────────────────────────────────────
def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Text truncation
# ────────────────────────────────────────────────────────────────────────────
def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending '...(truncated)' if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def truncate_line(line: str, max_chars: int) -> str:
    """Truncate a single line to max_chars, appending '...(truncated)' if needed."""
    if len(line) <= max_chars:
        return line
    return line[:max_chars] + "...(truncated)"


# ────────────────────────────────────────────────────────────────────────────
# TTL cache cleanup
# ────────────────────────────────────────────────────────────────────────────
def cleanup_ttl_cache(cache: dict[str, dict[str, Any]], *, now: float, ttl: int) -> None:
    """Remove expired entries from a TTL cache.

    Args:
        cache: Dict mapping keys to dicts with 'ts' field
        now: Current timestamp (seconds since epoch)
        ttl: Time-to-live in seconds
    """
    expired: list[str] = []
    for k, v in cache.items():
        ts = float(v.get("ts", 0.0))
        if now - ts > ttl:
            expired.append(k)
    for k in expired:
        cache.pop(k, None)


def cleanup_loop_local_dicts() -> None:
    """Clean up loop-local dictionaries, keeping current + 10 most recent loops.

    Prevents unbounded memory growth in long-running processes.
    Cleans both _FILE_LOCKS_BY_LOOP and _BASH_SEMAPHORES.
    """
    try:
        current_lid = loop_id()
    except RuntimeError:
        # No running loop (shouldn't happen in normal usage)
        return

    # Collect all loop IDs from both dicts
    all_lids = sorted(set(_FILE_LOCKS_BY_LOOP.keys()) | set(_BASH_SEMAPHORES.keys()))
    if current_lid in all_lids:
        all_lids.remove(current_lid)

    # Remove oldest loops beyond the retention limit (keep 10 most recent)
    retention_limit = 10
    if len(all_lids) > retention_limit:
        to_remove = all_lids[:-retention_limit]
        for lid in to_remove:
            _FILE_LOCKS_BY_LOOP.pop(lid, None)
            _BASH_SEMAPHORES.pop(lid, None)


# ────────────────────────────────────────────────────────────────────────────
# Async I/O wrapper
# ────────────────────────────────────────────────────────────────────────────
async def io_call(func: Callable, /, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking I/O function in a thread pool executor."""
    return await asyncio.to_thread(func, *args, **kwargs)


# ────────────────────────────────────────────────────────────────────────────
# Unified diff generation
# ────────────────────────────────────────────────────────────────────────────
def generate_unified_diff(
    before: str,
    after: str,
    fromfile: str = "",
    tofile: str = "",
) -> list[str]:
    """Generate unified diff lines between two strings.

    Args:
        before: Original text
        after: Modified text
        fromfile: Label for 'before' (e.g., filename)
        tofile: Label for 'after' (e.g., filename)

    Returns:
        List of diff lines (without trailing newlines)
    """
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=fromfile,
        tofile=tofile,
    )
    return [line.rstrip("\n") for line in diff]


# ────────────────────────────────────────────────────────────────────────────
# Artifact spilling (high-level wrappers, still depends on ctx)
# ────────────────────────────────────────────────────────────────────────────
async def spill_text(
    *,
    artifact_store: ArtifactStore,
    namespace: str,
    text: str,
    mime: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Spill large text to artifact store.

    Args:
        artifact_store: ArtifactStore instance
        namespace: Artifact namespace (e.g., 'read', 'bash')
        text: Text content to spill
        mime: MIME type (e.g., 'text/plain')
        ttl_seconds: Time-to-live in seconds

    Returns:
        Artifact metadata dict (relpath, size, etc.)
    """
    return await artifact_store.put_text(
        namespace=namespace,
        text=text,
        mime=mime,
        ttl_seconds=ttl_seconds,
    )


async def spill_json(
    *,
    artifact_store: ArtifactStore,
    namespace: str,
    data: Any,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Spill JSON-serializable data to artifact store.

    Args:
        artifact_store: ArtifactStore instance
        namespace: Artifact namespace (e.g., 'grep_files')
        data: JSON-serializable data
        ttl_seconds: Time-to-live in seconds

    Returns:
        Artifact metadata dict (relpath, size, etc.)
    """
    return await artifact_store.put_json(
        namespace=namespace,
        data=data,
        ttl_seconds=ttl_seconds,
    )


def format_file_size(size_bytes: int) -> str:
    """智能单位转换：B/KB/MB/GB

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        带单位的可读字符串（如 "1.5KB", "512B", "2.3MB"）
    """
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f}MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.1f}GB"
