"""测试 ExitPlanMode 工具。"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from comate_agent_sdk.system_tools.tools import ExitPlanMode
from comate_agent_sdk.tools.system_context import bind_system_tool_context


class TestExitPlanMode(unittest.TestCase):
    def test_exit_plan_mode_writes_artifact_and_returns_approval_payload(self) -> None:
        async def _run() -> None:
            with patch("comate_agent_sdk.system_tools.tools.Path.home", return_value=Path("/tmp")):
                with bind_system_tool_context(project_root=Path("/tmp")):
                    result = await ExitPlanMode.execute(
                        plan_markdown="# Plan\n\n- step 1",
                        summary="ready for approval",
                        title="my task",
                        execution_prompt="execute now",
                    )

            payload = json.loads(result)
            self.assertTrue(payload["ok"])
            data = payload["data"]
            self.assertEqual(data["status"], "waiting_for_plan_approval")
            self.assertEqual(data["summary"], "ready for approval")
            self.assertEqual(data["execution_prompt"], "execute now")

            artifact_path = Path(data["plan_path"])
            self.assertTrue(artifact_path.exists())
            self.assertIn("my-task", artifact_path.name)
            content = artifact_path.read_text(encoding="utf-8")
            self.assertIn("# Plan", content)

            # cleanup
            artifact_path.unlink(missing_ok=True)

        asyncio.run(_run())
