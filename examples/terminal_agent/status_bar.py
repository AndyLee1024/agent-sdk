from __future__ import annotations

import logging
import subprocess
import time
from typing import NamedTuple

from prompt_toolkit.application.current import get_app_or_none

from comate_agent_sdk.agent import ChatSession

logger = logging.getLogger(__name__)


class GitDiffStats(NamedTuple):
    added: int
    removed: int


class StatusBar:
    _DEFAULT_TERMINAL_WIDTH: int = 100
    _MIN_TERMINAL_WIDTH: int = 40
    _GIT_DIFF_CACHE_SECONDS: float = 5.0

    def __init__(self, session: ChatSession):
        self._session = session
        self._model_name: str = self._resolve_model_name(session)
        self._mode: str = "act"
        self._git_branch: str = self._resolve_git_branch()
        self._context_used_pct: float = 0.0
        self._context_left_pct: float = 100.0
        self._git_diff_stats: GitDiffStats | None = None
        self._git_diff_cache_time: float = 0.0

    @staticmethod
    def _resolve_model_name(session: ChatSession) -> str:
        agent = getattr(session, "_agent", None)
        llm = getattr(agent, "llm", None)
        model = getattr(llm, "model", "")
        normalized = str(model).strip()
        return normalized or "unknown-model"

    def set_model_name(self, model_name: str) -> None:
        normalized = str(model_name).strip()
        self._model_name = normalized or "unknown-model"

    def set_session(self, session: ChatSession) -> None:
        self._session = session
        self._model_name = self._resolve_model_name(session)
        try:
            self._mode = str(session.get_mode()).strip().lower() or "act"
        except Exception:
            self._mode = "act"

    def set_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        self._mode = normalized or "act"

    @staticmethod
    def _resolve_git_branch() -> str:
        try:
            completed = subprocess.run(
                ["git", "branch", "--show-current"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return "N/A"

        if completed.returncode != 0:
            return "N/A"
        branch = completed.stdout.strip()
        return branch or "N/A"

    @staticmethod
    def _resolve_git_diff_stats() -> GitDiffStats | None:
        try:
            completed = subprocess.run(
                ["git", "diff", "--shortstat", "--no-color"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None

        if completed.returncode != 0:
            return None

        output = completed.stdout.strip()
        if not output:
            return None

        # Parse output like "3 files changed, 12 insertions(+), 5 deletions(-)"
        # or just "12 insertions(+), 5 deletions(-)"
        added, removed = 0, 0
        for part in output.split(","):
            part = part.strip()
            if "+" in part and "insertion" in part:
                # e.g., "12 insertions(+)"
                num_str = part.split()[0]
                added = int(num_str)
            elif "-" in part and "deletion" in part:
                # e.g., "5 deletions(-)"
                num_str = part.split()[0]
                removed = int(num_str)

        return GitDiffStats(added=added, removed=removed)

    def _ensure_git_diff_stats(self) -> None:
        now = time.monotonic()
        if (
            self._git_diff_stats is not None
            and now - self._git_diff_cache_time < self._GIT_DIFF_CACHE_SECONDS
        ):
            return
        self._git_diff_stats = self._resolve_git_diff_stats()
        self._git_diff_cache_time = now
        logger.debug(
            f"Git diff stats refreshed: {self._git_diff_stats}"
        )

    async def refresh(self) -> None:
        try:
            ctx_info = await self._session.get_context_info()
            utilization = float(getattr(ctx_info, "utilization_percent", 0.0))
        except Exception:
            return

        try:
            self._mode = str(self._session.get_mode()).strip().lower() or "act"
        except Exception:
            pass

        normalized = max(0.0, min(utilization, 100.0))
        self._context_used_pct = normalized
        self._context_left_pct = max(0.0, 100.0 - normalized)

        # Refresh git diff stats to keep them up-to-date
        self._ensure_git_diff_stats()

    @classmethod
    def _resolve_terminal_width(cls) -> int:
        app = get_app_or_none()
        if app is None:
            return cls._DEFAULT_TERMINAL_WIDTH

        try:
            width = int(app.output.get_size().columns)
        except Exception:
            return cls._DEFAULT_TERMINAL_WIDTH
        return max(width, cls._MIN_TERMINAL_WIDTH)

    @staticmethod
    def _truncate_text(text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return f"{text[: max_len - 3]}..."

    def _right_prompt_budget(self) -> int:
        width = self._resolve_terminal_width()
        return max(24, min(width - 6, 72))

    def context_left_text(self) -> str:
        return f"{self._context_left_pct:.0f}% context left"

    def _status_text_for_width(self, width: int) -> str:
        mode_text = f"[{self._mode}]"
        branch_text = f"~{self._git_branch}"
        context_text = self.context_left_text()
        full_text = f"{mode_text} {self._model_name} | {branch_text} / {context_text}"
        budget = max(len(context_text), width)
        if len(full_text) <= budget:
            return full_text

        prefix = f"{mode_text} {self._model_name} | {branch_text} / "
        prefix_budget = max(0, budget - len(context_text))
        trimmed_prefix = self._truncate_text(prefix, prefix_budget)
        return f"{trimmed_prefix}{context_text}"

    def right_prompt_text(self) -> str:
        return self._status_text_for_width(self._right_prompt_budget())

    def right_prompt_fragments(self) -> list[tuple[str, str]]:
        return [("class:prompt.rprompt", self.right_prompt_text())]

    def footer_status_text(self) -> str:
        width = self._resolve_terminal_width()
        content_budget = max(16, width - 2)
        return self._status_text_for_width(content_budget)

    def _git_diff_fragments(self) -> list[tuple[str, str]]:
        self._ensure_git_diff_stats()
        if self._git_diff_stats is None or (
            self._git_diff_stats.added == 0 and self._git_diff_stats.removed == 0
        ):
            return []

        parts: list[tuple[str, str]] = []
        if self._git_diff_stats.added > 0:
            parts.append(("class:git-diff.added", f"+{self._git_diff_stats.added}"))
        if self._git_diff_stats.removed > 0:
            parts.append(("class:git-diff.removed", f"-{self._git_diff_stats.removed}"))
        return parts

    def git_diff_fragments(self) -> list[tuple[str, str]]:
        """Prompt-toolkit fragments for git diff stats.

        Working tree only (unstaged): `git diff --shortstat`.
        """
        return self._git_diff_fragments()

    def footer_toolbar(self) -> list[tuple[str, str]]:
        width = self._resolve_terminal_width()
        status_text = self.footer_status_text()
        git_fragments = self._git_diff_fragments()

        # Calculate total length: status_text + git diff parts
        git_len = sum(len(text) + 1 for _, text in git_fragments)  # +1 for space
        left_padding = max(0, width - len(status_text) - git_len - 1)

        fragments: list[tuple[str, str]] = [
            ("", " " * left_padding),
            ("", status_text),
        ]

        # Add git diff stats with colors
        if git_fragments:
            fragments.append(("", " "))  # Separator
            for class_name, text in git_fragments:
                fragments.append((class_name, text))

        fragments.append(("", " "))
        return fragments

    def helper_toolbar(self) -> list[tuple[str, str]]:
        return []
