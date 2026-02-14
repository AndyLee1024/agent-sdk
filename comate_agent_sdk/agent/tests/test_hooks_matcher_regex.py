import unittest
from pathlib import Path

from comate_agent_sdk.agent.hooks.engine import HookEngine
from comate_agent_sdk.agent.hooks.models import HookConfig, HookHandlerSpec, HookInput, HookMatcherGroup, HookResult


class TestHooksMatcherRegex(unittest.IsolatedAsyncioTestCase):
    async def test_matcher_default_star_and_regex(self) -> None:
        config = HookConfig(
            events={
                "PreToolUse": [
                    HookMatcherGroup(
                        matcher="*",
                        hooks=[
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(permission_decision="allow"),
                            )
                        ],
                    ),
                    HookMatcherGroup(
                        matcher="^Read$",
                        hooks=[
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(permission_decision="ask"),
                            )
                        ],
                    ),
                ]
            }
        )
        engine = HookEngine(config=config, project_root=Path.cwd(), session_id="s1")

        read_outcome = await engine.run_event(
            "PreToolUse",
            HookInput(
                session_id="s1",
                cwd=str(Path.cwd()),
                permission_mode="default",
                hook_event_name="PreToolUse",
                tool_name="Read",
                tool_input={},
            ),
        )
        self.assertEqual(read_outcome.permission_decision, "ask")

        write_outcome = await engine.run_event(
            "PreToolUse",
            HookInput(
                session_id="s1",
                cwd=str(Path.cwd()),
                permission_mode="default",
                hook_event_name="PreToolUse",
                tool_name="Write",
                tool_input={},
            ),
        )
        self.assertEqual(write_outcome.permission_decision, "allow")

    async def test_matcher_is_ignored_for_non_tool_events(self) -> None:
        config = HookConfig(
            events={
                "Stop": [
                    HookMatcherGroup(
                        matcher="^NeverMatch$",
                        hooks=[
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(decision="block", reason="session-started"),
                            )
                        ],
                    )
                ]
            }
        )
        engine = HookEngine(config=config, project_root=Path.cwd(), session_id="s1")

        outcome = await engine.run_event(
            "Stop",
            HookInput(
                session_id="s1",
                cwd=str(Path.cwd()),
                permission_mode="default",
                hook_event_name="Stop",
                stop_reason="completed",
            ),
        )
        self.assertEqual(outcome.decision, "block")
        self.assertEqual(outcome.reason, "session-started")
