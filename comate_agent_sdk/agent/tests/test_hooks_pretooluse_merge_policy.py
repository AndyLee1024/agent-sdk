import unittest
from pathlib import Path

from comate_agent_sdk.agent.hooks.engine import HookEngine
from comate_agent_sdk.agent.hooks.models import HookConfig, HookHandlerSpec, HookInput, HookMatcherGroup, HookResult


class TestHooksPreToolUseMergePolicy(unittest.IsolatedAsyncioTestCase):
    async def test_deny_has_highest_priority_and_short_circuits(self) -> None:
        visited = {"tail": 0}

        def _tail(_: HookInput) -> HookResult:
            visited["tail"] += 1
            return HookResult(permission_decision="allow")

        config = HookConfig(
            events={
                "PreToolUse": [
                    HookMatcherGroup(
                        hooks=[
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(
                                    permission_decision="allow",
                                    updated_input={"value": 1},
                                ),
                            ),
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(permission_decision="ask"),
                            ),
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(
                                    permission_decision="deny",
                                    reason="hard-block",
                                ),
                            ),
                            HookHandlerSpec(type="python", callback=_tail),
                        ]
                    )
                ]
            }
        )

        engine = HookEngine(config=config, project_root=Path.cwd(), session_id="s1")
        outcome = await engine.run_event(
            "PreToolUse",
            HookInput(
                session_id="s1",
                cwd=str(Path.cwd()),
                permission_mode="default",
                hook_event_name="PreToolUse",
                tool_name="Read",
                tool_input={"value": 0},
            ),
        )

        self.assertEqual(outcome.permission_decision, "deny")
        self.assertEqual(outcome.reason, "hard-block")
        self.assertIsNone(outcome.updated_input)
        self.assertEqual(visited["tail"], 0)

    async def test_ask_overrides_allow_and_updated_input_last_wins(self) -> None:
        config = HookConfig(
            events={
                "PreToolUse": [
                    HookMatcherGroup(
                        hooks=[
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(
                                    permission_decision="allow",
                                    updated_input={"value": 1},
                                ),
                            ),
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(permission_decision="ask", reason="need-approval"),
                            ),
                            HookHandlerSpec(
                                type="python",
                                callback=lambda _: HookResult(
                                    permission_decision="allow",
                                    updated_input={"value": 2},
                                ),
                            ),
                        ]
                    )
                ]
            }
        )

        engine = HookEngine(config=config, project_root=Path.cwd(), session_id="s1")
        outcome = await engine.run_event(
            "PreToolUse",
            HookInput(
                session_id="s1",
                cwd=str(Path.cwd()),
                permission_mode="default",
                hook_event_name="PreToolUse",
                tool_name="Write",
                tool_input={"value": 0},
            ),
        )

        self.assertEqual(outcome.permission_decision, "ask")
        self.assertEqual(outcome.reason, "need-approval")
        self.assertEqual(outcome.updated_input, {"value": 2})
