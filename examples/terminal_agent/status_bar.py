from __future__ import annotations

import subprocess

from prompt_toolkit.application.current import get_app_or_none

from comate_agent_sdk.agent import ChatSession


class StatusBar:
    _DEFAULT_TERMINAL_WIDTH: int = 100
    _MIN_TERMINAL_WIDTH: int = 40

    def __init__(self, session: ChatSession):
        self._session = session
        self._model_name: str = self._resolve_model_name(session)
        self._git_branch: str = self._resolve_git_branch()
        self._context_used_pct: float = 0.0
        self._context_left_pct: float = 100.0

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

    async def refresh(self) -> None:
        try:
            ctx_info = await self._session.get_context_info()
            utilization = float(getattr(ctx_info, "utilization_percent", 0.0))
        except Exception:
            return

        normalized = max(0.0, min(utilization, 100.0))
        self._context_used_pct = normalized
        self._context_left_pct = max(0.0, 100.0 - normalized)

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
        branch_text = f"~{self._git_branch}"
        context_text = self.context_left_text()
        full_text = f"{self._model_name} | {branch_text} / {context_text}"
        budget = max(len(context_text), width)
        if len(full_text) <= budget:
            return full_text

        prefix = f"{self._model_name} | {branch_text} / "
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

    def footer_toolbar(self) -> list[tuple[str, str]]:
        width = self._resolve_terminal_width()
        status_text = self.footer_status_text()
        left_padding = max(0, width - len(status_text) - 1)
        return [
        
            ("", " " * left_padding),
            ("", status_text),
            ("", " "),
        ]

    def helper_toolbar(self) -> list[tuple[str, str]]:
        return []
