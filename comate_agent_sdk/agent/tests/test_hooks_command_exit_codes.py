import tempfile
import unittest
from pathlib import Path

from comate_agent_sdk.agent.hooks.engine import HookEngine
from comate_agent_sdk.agent.hooks.models import HookConfig, HookHandlerSpec, HookInput, HookMatcherGroup


class TestHooksCommandExitCodes(unittest.IsolatedAsyncioTestCase):
    async def test_exit_zero_valid_json_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "ok.sh"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "echo '{\"permissionDecision\":\"allow\"}'\n",
                encoding="utf-8",
            )
            script.chmod(0o755)

            engine = HookEngine(
                config=HookConfig(
                    events={
                        "PreToolUse": [
                            HookMatcherGroup(
                                hooks=[
                                    HookHandlerSpec(type="command", command=str(script), timeout=3),
                                ]
                            )
                        ]
                    }
                ),
                project_root=root,
                session_id="s1",
            )
            outcome = await engine.run_event(
                "PreToolUse",
                HookInput(
                    session_id="s1",
                    cwd=str(root),
                    permission_mode="default",
                    hook_event_name="PreToolUse",
                    tool_name="Read",
                    tool_input={},
                ),
            )
            self.assertEqual(outcome.permission_decision, "allow")

    async def test_exit_zero_invalid_json_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "bad_json.sh"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'not-json'\n",
                encoding="utf-8",
            )
            script.chmod(0o755)

            engine = HookEngine(
                config=HookConfig(
                    events={
                        "PreToolUse": [
                            HookMatcherGroup(
                                hooks=[HookHandlerSpec(type="command", command=str(script), timeout=3)]
                            )
                        ]
                    }
                ),
                project_root=root,
                session_id="s1",
            )
            outcome = await engine.run_event(
                "PreToolUse",
                HookInput(
                    session_id="s1",
                    cwd=str(root),
                    permission_mode="default",
                    hook_event_name="PreToolUse",
                    tool_name="Read",
                    tool_input={},
                ),
            )
            self.assertIsNone(outcome.permission_decision)

    async def test_exit_two_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "block.sh"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'blocked-by-script' >&2\n"
                "exit 2\n",
                encoding="utf-8",
            )
            script.chmod(0o755)

            engine = HookEngine(
                config=HookConfig(
                    events={
                        "PreToolUse": [
                            HookMatcherGroup(
                                hooks=[HookHandlerSpec(type="command", command=str(script), timeout=3)]
                            )
                        ],
                        "Stop": [
                            HookMatcherGroup(
                                hooks=[HookHandlerSpec(type="command", command=str(script), timeout=3)]
                            )
                        ],
                    }
                ),
                project_root=root,
                session_id="s1",
            )

            pre_outcome = await engine.run_event(
                "PreToolUse",
                HookInput(
                    session_id="s1",
                    cwd=str(root),
                    permission_mode="default",
                    hook_event_name="PreToolUse",
                    tool_name="Read",
                    tool_input={},
                ),
            )
            self.assertEqual(pre_outcome.permission_decision, "deny")
            self.assertIn("blocked-by-script", pre_outcome.reason or "")

            stop_outcome = await engine.run_event(
                "Stop",
                HookInput(
                    session_id="s1",
                    cwd=str(root),
                    permission_mode="default",
                    hook_event_name="Stop",
                    stop_reason="completed",
                ),
            )
            self.assertEqual(stop_outcome.decision, "block")
            self.assertIn("blocked-by-script", stop_outcome.reason or "")

    async def test_exit_other_is_non_blocking_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "error.sh"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'oops' >&2\n"
                "exit 3\n",
                encoding="utf-8",
            )
            script.chmod(0o755)

            engine = HookEngine(
                config=HookConfig(
                    events={
                        "PreToolUse": [
                            HookMatcherGroup(
                                hooks=[HookHandlerSpec(type="command", command=str(script), timeout=3)]
                            )
                        ]
                    }
                ),
                project_root=root,
                session_id="s1",
            )
            outcome = await engine.run_event(
                "PreToolUse",
                HookInput(
                    session_id="s1",
                    cwd=str(root),
                    permission_mode="default",
                    hook_event_name="PreToolUse",
                    tool_name="Read",
                    tool_input={},
                ),
            )
            self.assertIsNone(outcome.permission_decision)
