from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from terminal_agent.status_bar import StatusBar


class _FakeSession:
    def __init__(self, model: str = "claude-sonnet-4-5", utilization_percent: float = 0.0) -> None:
        self._agent = SimpleNamespace(llm=SimpleNamespace(model=model))
        self._utilization_percent = utilization_percent
        self._raise_on_refresh = False

    async def get_context_info(self):
        if self._raise_on_refresh:
            raise RuntimeError("context failed")
        return SimpleNamespace(utilization_percent=self._utilization_percent)


def _join_fragments(fragments: list[tuple[str, str]]) -> str:
    return "".join(part for _, part in fragments)


class TestStatusBar(unittest.IsolatedAsyncioTestCase):
    @patch("terminal_agent.status_bar.subprocess.run")
    async def test_refresh_updates_cached_context_percent(self, mock_run) -> None:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="main\n")
        session = _FakeSession(utilization_percent=42.0)
        status_bar = StatusBar(session)

        await status_bar.refresh()
        footer = _join_fragments(status_bar.footer_toolbar())

        self.assertIn("claude-sonnet-4-5", footer)
        self.assertIn("~main", footer)
        self.assertIn("58% context left", footer)

    @patch("terminal_agent.status_bar.subprocess.run")
    async def test_refresh_failure_keeps_previous_cached_value(self, mock_run) -> None:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="dev\n")
        session = _FakeSession(utilization_percent=10.0)
        status_bar = StatusBar(session)

        await status_bar.refresh()
        self.assertIn("90% context left", status_bar.footer_status_text())

        session._raise_on_refresh = True
        await status_bar.refresh()
        self.assertIn("90% context left", status_bar.footer_status_text())

    @patch("terminal_agent.status_bar.subprocess.run")
    async def test_git_branch_failure_falls_back_to_na(self, mock_run) -> None:
        mock_run.side_effect = OSError("git not found")
        session = _FakeSession(model="", utilization_percent=0.0)
        status_bar = StatusBar(session)

        await status_bar.refresh()
        footer = status_bar.footer_status_text()
        self.assertIn("unknown-model", footer)
        self.assertIn("~N/A", footer)

    @patch("terminal_agent.status_bar.subprocess.run")
    @patch.object(StatusBar, "_resolve_terminal_width")
    async def test_right_prompt_keeps_context_when_terminal_is_narrow(self, mock_width, mock_run) -> None:
        mock_width.return_value = 40
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="main\n")
        session = _FakeSession(utilization_percent=15.0)
        status_bar = StatusBar(session)

        await status_bar.refresh()
        footer = _join_fragments(status_bar.footer_toolbar())

        self.assertIn("85% context left", footer)

    @patch("terminal_agent.status_bar.subprocess.run")
    async def test_helper_toolbar_is_removed(self, mock_run) -> None:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="main\n")
        session = _FakeSession(utilization_percent=0.0)
        status_bar = StatusBar(session)

        self.assertEqual(status_bar.helper_toolbar(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
