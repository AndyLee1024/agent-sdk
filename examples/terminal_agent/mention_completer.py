from __future__ import annotations

import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import logging
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MentionContext:
    marker_index: int
    fragment: str


class LocalFileMentionCompleter(Completer):
    """Offer fuzzy `@` path suggestions backed by workspace files."""

    _TRIGGER_GUARDS = frozenset((".", "-", "_", "`", "'", '"', ":", "@", "#", "~"))
    _IGNORED_NAME_GROUPS: dict[str, tuple[str, ...]] = {
        "vcs_metadata": (".DS_Store", ".bzr", ".git", ".hg", ".svn"),
        "tooling_caches": (
            ".build",
            ".cache",
            ".coverage",
            ".fleet",
            ".gradle",
            ".idea",
            ".ipynb_checkpoints",
            ".pnpm-store",
            ".pytest_cache",
            ".pub-cache",
            ".ruff_cache",
            ".swiftpm",
            ".tox",
            ".venv",
            ".vs",
            ".vscode",
            ".yarn",
            ".yarn-cache",
        ),
        "js_frontend": (
            ".next",
            ".nuxt",
            ".parcel-cache",
            ".svelte-kit",
            ".turbo",
            ".vercel",
            "node_modules",
        ),
        "python_packaging": (
            "__pycache__",
            "build",
            "coverage",
            "dist",
            "htmlcov",
            "pip-wheel-metadata",
            "venv",
        ),
        "java_jvm": (".mvn", "out", "target"),
        "dotnet_native": ("bin", "cmake-build-debug", "cmake-build-release", "obj"),
        "bazel_buck": ("bazel-bin", "bazel-out", "bazel-testlogs", "buck-out"),
        "misc_artifacts": (
            ".dart_tool",
            ".serverless",
            ".stack-work",
            ".terraform",
            ".terragrunt-cache",
            "DerivedData",
            "Pods",
            "deps",
            "tmp",
            "vendor",
        ),
    }
    _IGNORED_NAMES = frozenset(name for group in _IGNORED_NAME_GROUPS.values() for name in group)
    _IGNORED_PATTERN_PARTS: tuple[str, ...] = (
        r".*_cache$",
        r".*-cache$",
        r".*\.egg-info$",
        r".*\.dist-info$",
        r".*\.py[co]$",
        r".*\.class$",
        r".*\.sw[po]$",
        r".*~$",
        r".*\.(?:tmp|bak)$",
    )
    _IGNORED_PATTERNS = re.compile(
        "|".join(f"(?:{part})" for part in _IGNORED_PATTERN_PARTS),
        re.IGNORECASE,
    )

    def __init__(
        self,
        root: Path,
        *,
        refresh_interval: float = 2.0,
        limit: int = 1000,
    ) -> None:
        self._root = root
        self._refresh_interval = refresh_interval
        self._limit = limit

        self._top_cache_time: float = 0.0
        self._top_cached_paths: list[str] = []

        self._deep_cache_time: float = 0.0
        self._deep_cached_paths: list[str] = []

    @classmethod
    def _is_ignored(cls, name: str) -> bool:
        if not name:
            return True
        if name in cls._IGNORED_NAMES:
            return True
        return bool(cls._IGNORED_PATTERNS.fullmatch(name))

    def extract_context(self, text_before_cursor: str) -> MentionContext | None:
        marker_index = text_before_cursor.rfind("@")
        if marker_index == -1:
            return None

        if marker_index > 0:
            previous = text_before_cursor[marker_index - 1]
            if previous.isalnum() or previous in self._TRIGGER_GUARDS:
                return None

        fragment = text_before_cursor[marker_index + 1 :]
        if any(char.isspace() for char in fragment):
            return None

        return MentionContext(marker_index=marker_index, fragment=fragment)

    def suggest(self, text_before_cursor: str, *, max_items: int = 12) -> tuple[MentionContext, list[str]] | None:
        context = self.extract_context(text_before_cursor)
        if context is None:
            return None
        if self._is_completed_file(context.fragment):
            return None

        fragment_lower = context.fragment.lower()
        source = self._candidate_paths(context.fragment)
        if not source:
            return None

        matched: list[str] = []
        for path in source:
            if not fragment_lower:
                matched.append(path)
                continue
            if fragment_lower in path.lower():
                matched.append(path)

        if not matched:
            return None

        def _rank(path: str) -> tuple[int, str]:
            basename = path.rstrip("/").split("/")[-1].lower()
            if basename.startswith(fragment_lower):
                category = 0
            elif fragment_lower in basename:
                category = 1
            else:
                category = 2
            return (category, path.lower())

        matched.sort(key=_rank)
        return context, matched[: max(max_items, 1)]

    def apply_completion(
        self,
        *,
        full_text: str,
        cursor_position: int,
        context: MentionContext,
        completion: str,
        append_space: bool,
    ) -> tuple[str, int]:
        before_cursor = full_text[:cursor_position]
        after_cursor = full_text[cursor_position:]
        fragment_start = context.marker_index + 1
        replacement_before = f"{before_cursor[:fragment_start]}{completion}"
        new_cursor = len(replacement_before)
        replacement_after = after_cursor

        if append_space:
            should_insert_space = not replacement_after.startswith(" ")
            if should_insert_space:
                replacement_before = f"{replacement_before} "
                new_cursor += 1

        return f"{replacement_before}{replacement_after}", new_cursor

    def _candidate_paths(self, fragment: str) -> list[str]:
        if "/" not in fragment and len(fragment) < 3:
            return self._get_top_level_paths()
        return self._get_deep_paths()

    def _get_top_level_paths(self) -> list[str]:
        now = time.monotonic()
        if now - self._top_cache_time <= self._refresh_interval:
            return self._top_cached_paths

        entries: list[str] = []
        try:
            for entry in sorted(self._root.iterdir(), key=lambda path: path.name):
                name = entry.name
                if self._is_ignored(name):
                    continue
                entries.append(f"{name}/" if entry.is_dir() else name)
                if len(entries) >= self._limit:
                    break
        except OSError as exc:
            logger.debug("top-level mention scan failed: %s", exc)
            return self._top_cached_paths

        self._top_cached_paths = entries
        self._top_cache_time = now
        return self._top_cached_paths

    def _get_deep_paths(self) -> list[str]:
        now = time.monotonic()
        if now - self._deep_cache_time <= self._refresh_interval:
            return self._deep_cached_paths

        paths: list[str] = []
        try:
            for current_root, dirs, files in os.walk(self._root):
                relative_root = Path(current_root).relative_to(self._root)
                dirs[:] = sorted(directory for directory in dirs if not self._is_ignored(directory))

                if relative_root.parts and any(self._is_ignored(part) for part in relative_root.parts):
                    dirs[:] = []
                    continue

                if relative_root.parts:
                    paths.append(relative_root.as_posix() + "/")
                    if len(paths) >= self._limit:
                        break

                for file_name in sorted(files):
                    if self._is_ignored(file_name):
                        continue
                    relative = (relative_root / file_name).as_posix()
                    if not relative:
                        continue
                    paths.append(relative)
                    if len(paths) >= self._limit:
                        break

                if len(paths) >= self._limit:
                    break
        except OSError as exc:
            logger.debug("deep mention scan failed: %s", exc)
            return self._deep_cached_paths

        self._deep_cached_paths = paths
        self._deep_cache_time = now
        return self._deep_cached_paths

    def _is_completed_file(self, fragment: str) -> bool:
        candidate = fragment.rstrip("/")
        if not candidate:
            return False
        try:
            return (self._root / candidate).is_file()
        except OSError:
            return False

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        del complete_event
        result = self.suggest(document.text_before_cursor, max_items=12)
        if result is None:
            return

        context, candidates = result
        for candidate in candidates:
            append_space = not candidate.endswith("/") and not document.text_after_cursor.startswith(" ")
            insert_text = f"{candidate} " if append_space else candidate
            yield Completion(
                text=insert_text,
                start_position=-len(context.fragment),
                display=candidate,
            )
