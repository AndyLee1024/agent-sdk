from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyViolation:
    code: str
    message: str


class BashCommandPolicy:
    def __init__(
        self,
        *,
        banned_commands: set[str] | None = None,
        rg_required_flags: set[str] | None = None,
        rg_flags_with_value: set[str] | None = None,
    ) -> None:
        self._banned_commands = banned_commands or {"grep", "find", "cat", "head", "tail", "ls"}
        self._rg_required_flags = rg_required_flags or {"--line-number", "--no-heading", "--color=never"}
        self._rg_flags_with_value = rg_flags_with_value or {
            "--glob",
            "--type",
            "--max-count",
            "-g",
            "-t",
            "-m",
            "-e",
            "-f",
        }

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _has_flag(args: list[str], flag: str) -> bool:
        return flag in args

    @staticmethod
    def _has_flag_prefix(args: list[str], flag: str) -> bool:
        return any(a == flag or a.startswith(f"{flag}=") for a in args)

    def _get_rg_scopes(self, args: list[str]) -> list[str]:
        """
        Best-effort parse:
        rg [opts...] PATTERN [PATH...]

        Policy requires explicit PATH scopes.
        """
        if not args:
            return []
        if Path(args[0]).name != "rg":
            return []

        i = 1
        while i < len(args):
            token = args[i]
            if token == "--":
                i += 1
                break
            if not token.startswith("-"):
                break
            if token in self._rg_flags_with_value:
                i += 2
                continue
            if token.startswith("--max-count="):
                i += 1
                continue
            i += 1

        if i >= len(args):
            return []

        # Pattern
        i += 1
        if i >= len(args):
            return []
        return args[i:]

    def validate(
        self,
        *,
        args: list[str],
        cwd: Path,
        allowed_roots: list[Path],
    ) -> PolicyViolation | None:
        if not args:
            return PolicyViolation("INVALID_ARGUMENT", "Command arguments cannot be empty")

        exe_name = Path(args[0]).name
        if exe_name in self._banned_commands:
            return PolicyViolation(
                "POLICY_DENIED",
                f"Bash command '{exe_name}' is not allowed; use specialized tools instead.",
            )

        if exe_name != "rg":
            return None

        missing = [f for f in self._rg_required_flags if not self._has_flag(args, f)]
        if missing:
            return PolicyViolation(
                "POLICY_DENIED",
                f"rg missing required flags: {', '.join(sorted(missing))}",
            )

        if not self._has_flag_prefix(args, "--max-count"):
            return PolicyViolation("POLICY_DENIED", "rg must include --max-count to bound output")

        scopes = self._get_rg_scopes(args)
        if not scopes:
            return PolicyViolation(
                "POLICY_DENIED",
                "rg must include explicit scoped PATH(s); implicit cwd search is not allowed",
            )

        if any(s in {".", "/"} for s in scopes):
            return PolicyViolation("POLICY_DENIED", "rg scope cannot be '.' or '/'")

        for scope in scopes:
            p = Path(scope)
            abs_p = p.resolve() if p.is_absolute() else (cwd / p).resolve()
            if not any(self._is_under(abs_p, r) for r in allowed_roots):
                return PolicyViolation("POLICY_DENIED", f"rg scope outside allowed roots: {scope}")

        return None


class ReadRegistryPolicy:
    def __init__(self, *, filename: str = "read_index.json") -> None:
        self._filename = filename
        self._memory: dict[str, set[str]] = defaultdict(set)
        self._locks_by_loop: dict[int, dict[str, asyncio.Lock]] = defaultdict(dict)

    @property
    def memory(self) -> dict[str, set[str]]:
        return self._memory

    def clear_memory(self) -> None:
        self._memory.clear()

    @staticmethod
    def _loop_id() -> int:
        return id(asyncio.get_running_loop())

    def _lock(self, key: str) -> asyncio.Lock:
        lid = self._loop_id()
        lock = self._locks_by_loop[lid].get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks_by_loop[lid][key] = lock
        return lock

    @staticmethod
    async def _io_call(func, /, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    @staticmethod
    def _atomic_write_text_sync(path: Path, content: str) -> None:
        payload = content.encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except FileNotFoundError:
                    pass

    def _registry_key(self, *, session_root: Path | None, session_id: str) -> str:
        if session_root is not None:
            return str(session_root.resolve())
        return f"session:{session_id}"

    def _registry_path(self, session_root: Path | None) -> Path | None:
        if session_root is None:
            return None
        return session_root.resolve() / self._filename

    async def _load(self, *, session_root: Path | None, session_id: str) -> set[str]:
        path = self._registry_path(session_root)
        if path is None:
            return set(self._memory.get(self._registry_key(session_root=None, session_id=session_id), set()))
        exists = await self._io_call(path.exists)
        if not exists:
            return set()
        try:
            raw = await self._io_call(path.read_text, encoding="utf-8", errors="replace")
            data = json.loads(raw) if raw.strip() else {}
            files = data.get("files", [])
            if not isinstance(files, list):
                return set()
            return {str(x) for x in files if isinstance(x, str)}
        except Exception:
            return set()

    async def _save(self, *, session_root: Path | None, session_id: str, reg: set[str]) -> None:
        path = self._registry_path(session_root)
        if path is None:
            self._memory[self._registry_key(session_root=None, session_id=session_id)] = set(reg)
            return
        payload = {
            "schema_version": 1,
            "files": sorted(reg),
        }
        await self._io_call(
            self._atomic_write_text_sync,
            path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    async def mark_read(self, *, session_root: Path | None, session_id: str, path: Path) -> None:
        key = self._registry_key(session_root=session_root, session_id=session_id)
        async with self._lock(key):
            reg = await self._load(session_root=session_root, session_id=session_id)
            reg.add(str(path.resolve()))
            await self._save(session_root=session_root, session_id=session_id, reg=reg)

    async def has_read(self, *, session_root: Path | None, session_id: str, path: Path) -> bool:
        key = self._registry_key(session_root=session_root, session_id=session_id)
        async with self._lock(key):
            reg = await self._load(session_root=session_root, session_id=session_id)
            return str(path.resolve()) in reg


BASH_COMMAND_POLICY = BashCommandPolicy()
READ_REGISTRY_POLICY = ReadRegistryPolicy()
