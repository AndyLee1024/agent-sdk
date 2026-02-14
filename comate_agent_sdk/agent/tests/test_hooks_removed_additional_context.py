import unittest
from pathlib import Path

from comate_agent_sdk.agent.hooks.engine import HookEngine
from comate_agent_sdk.agent.hooks.models import HookConfig, HookHandlerSpec, HookInput, HookMatcherGroup, HookResult


class TestHooksRemovedAdditionalContext(unittest.IsolatedAsyncioTestCase):
    def test_from_mapping_rejects_additional_context_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "additionalContext"):
            HookResult.from_mapping({"additionalContext": "legacy"})

        with self.assertRaisesRegex(ValueError, "additionalContext"):
            HookResult.from_mapping({"additional_context": "legacy"})

    async def test_engine_raises_for_legacy_additional_context(self) -> None:
        engine = HookEngine(
            config=HookConfig(
                events={
                    "SessionStart": [
                        HookMatcherGroup(
                            hooks=[
                                HookHandlerSpec(
                                    type="python",
                                    callback=lambda _: {"additionalContext": "legacy"},
                                )
                            ]
                        )
                    ]
                }
            ),
            project_root=Path.cwd(),
            session_id="s-legacy",
        )

        with self.assertRaisesRegex(ValueError, "additionalContext"):
            await engine.run_event(
                "SessionStart",
                HookInput(
                    session_id="s-legacy",
                    cwd=str(Path.cwd()),
                    permission_mode="default",
                    hook_event_name="SessionStart",
                ),
            )
