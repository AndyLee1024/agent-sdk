import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comate_agent_sdk.agent.hooks.loader import load_hook_config_from_sources


class TestHooksLoaderMergePrecedence(unittest.TestCase):
    def test_global_project_local_are_merged_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            user_dir = Path(tmp) / "home" / ".agent"
            project_agent_dir = root / ".agent"
            root.mkdir(parents=True)
            user_dir.mkdir(parents=True)
            project_agent_dir.mkdir(parents=True)

            user_settings = user_dir / "settings.json"
            project_settings = project_agent_dir / "settings.json"
            local_settings = project_agent_dir / "settings.local.json"

            user_settings.write_text(
                json.dumps(
                    {
                        "hooks": {
                            "PreToolUse": [
                                {
                                    "matcher": "Read",
                                    "hooks": [
                                        {"type": "command", "command": "echo user", "timeout": 5}
                                    ],
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            project_settings.write_text(
                json.dumps(
                    {
                        "hooks": {
                            "PreToolUse": [
                                {
                                    "matcher": "Write",
                                    "hooks": [
                                        {"type": "command", "command": "echo project", "timeout": 5}
                                    ],
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            local_settings.write_text(
                json.dumps(
                    {
                        "hooks": {
                            "PreToolUse": [
                                {
                                    "matcher": "Edit",
                                    "hooks": [
                                        {"type": "command", "command": "echo local", "timeout": 5}
                                    ],
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("comate_agent_sdk.agent.hooks.loader.USER_SETTINGS_PATH", user_settings):
                config = load_hook_config_from_sources(
                    project_root=root,
                    sources=("user", "project", "local"),
                )

        groups = config.groups_for("PreToolUse")
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0].matcher, "Read")
        self.assertEqual(groups[1].matcher, "Write")
        self.assertEqual(groups[2].matcher, "Edit")
