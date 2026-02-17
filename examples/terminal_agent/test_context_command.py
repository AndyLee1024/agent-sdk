from __future__ import annotations

import unittest
from types import SimpleNamespace

from terminal_agent.tui_parts.commands import CommandsMixin


class _FakeRenderer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    def append_system_message(self, text: str, is_error: bool = False) -> None:
        self.messages.append((text, is_error))


class _Harness(CommandsMixin):
    def __init__(self, *, session: object, renderer: _FakeRenderer) -> None:
        self._session = session
        self._renderer = renderer


class TestContextCommand(unittest.IsolatedAsyncioTestCase):
    def _build_session(self) -> object:
        info = SimpleNamespace(
            context_limit=204800,
            next_step_estimated_tokens=35635,
            last_step_reported_tokens=32096,
            used_tokens=32610,
        )
        usage = SimpleNamespace(
            prompt_tokens=32068,
            prompt_cached_tokens=0,
            prompt_cache_creation_tokens=9701,
            completion_tokens=28,
            total_tokens=32096,
        )
        usage_entry = SimpleNamespace(usage=usage)
        token_cost = SimpleNamespace(usage_history=[usage_entry])
        agent = SimpleNamespace(_token_cost=token_cost)
        return SimpleNamespace(
            _agent=agent,
            get_context_info=self._make_async_return(info),
        )

    @staticmethod
    def _make_async_return(value):
        async def _inner():
            return value

        return _inner

    async def test_context_default_shows_only_est_and_actual(self) -> None:
        renderer = _FakeRenderer()
        session = self._build_session()
        app = _Harness(session=session, renderer=renderer)

        await app._slash_context("")

        message, is_error = renderer.messages[-1]
        self.assertFalse(is_error)
        self.assertIn("Headroom (est):", message)
        self.assertIn("Last call (actual):", message)
        self.assertNotIn("Context Details", message)
        self.assertNotIn("breakdown(last call)", message)

    async def test_context_details_shows_breakdown_and_delta(self) -> None:
        renderer = _FakeRenderer()
        session = self._build_session()
        app = _Harness(session=session, renderer=renderer)

        await app._slash_context("--details")

        message, is_error = renderer.messages[-1]
        self.assertFalse(is_error)
        self.assertIn("Context Details", message)
        self.assertIn("breakdown(last call):", message)
        self.assertIn("I=22,367", message)
        self.assertIn("delta_ir_vs_actual: +514", message)

    async def test_context_invalid_args_show_usage(self) -> None:
        renderer = _FakeRenderer()
        session = self._build_session()
        app = _Harness(session=session, renderer=renderer)

        await app._slash_context("--unknown")

        message, is_error = renderer.messages[-1]
        self.assertTrue(is_error)
        self.assertEqual(message, "Usage: /context [--details]")


if __name__ == "__main__":
    unittest.main()
