import asyncio
import time
import unittest
from pathlib import Path

from comate_agent_sdk.agent.hooks.engine import HookEngine
from comate_agent_sdk.agent.hooks.models import HookConfig, HookHandlerSpec, HookInput, HookMatcherGroup, HookResult


def _build_hook_input(event_name: str) -> HookInput:
    payload = HookInput(
        session_id="s-timeout",
        cwd=str(Path.cwd()),
        permission_mode="default",
        hook_event_name=event_name,
    )
    if event_name == "Stop":
        payload.stop_reason = "completed"
    return payload


class TestHooksPythonTimeout(unittest.IsolatedAsyncioTestCase):
    async def test_sync_python_hook_timeout_is_non_blocking(self) -> None:
        def _slow_sync(_: HookInput) -> HookResult:
            time.sleep(1.2)
            return HookResult(decision="block", reason="late-sync")

        engine = HookEngine(
            config=HookConfig(
                events={
                    "Stop": [
                        HookMatcherGroup(
                            hooks=[HookHandlerSpec(type="python", callback=_slow_sync, timeout=1)]
                        )
                    ]
                }
            ),
            project_root=Path.cwd(),
            session_id="s-timeout",
        )

        outcome = await engine.run_event("Stop", _build_hook_input("Stop"))
        self.assertFalse(outcome.should_block_stop)
        self.assertIsNone(outcome.reason)

    async def test_async_python_hook_timeout_is_non_blocking(self) -> None:
        async def _slow_async(_: HookInput) -> HookResult:
            await asyncio.sleep(1.2)
            return HookResult(decision="block", reason="late-async")

        engine = HookEngine(
            config=HookConfig(
                events={
                    "Stop": [
                        HookMatcherGroup(
                            hooks=[HookHandlerSpec(type="python", callback=_slow_async, timeout=1)]
                        )
                    ]
                }
            ),
            project_root=Path.cwd(),
            session_id="s-timeout",
        )

        outcome = await engine.run_event("Stop", _build_hook_input("Stop"))
        self.assertFalse(outcome.should_block_stop)
        self.assertIsNone(outcome.reason)

    async def test_python_hook_within_timeout_can_block_stop(self) -> None:
        def _fast_sync(_: HookInput) -> HookResult:
            return HookResult(decision="block", reason="ok")

        engine = HookEngine(
            config=HookConfig(
                events={
                    "Stop": [
                        HookMatcherGroup(
                            hooks=[HookHandlerSpec(type="python", callback=_fast_sync, timeout=2)]
                        )
                    ]
                }
            ),
            project_root=Path.cwd(),
            session_id="s-timeout",
        )

        outcome = await engine.run_event("Stop", _build_hook_input("Stop"))
        self.assertTrue(outcome.should_block_stop)
        self.assertEqual(outcome.reason, "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
