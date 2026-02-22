from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from comate_agent_sdk.agent.mode_policy import evaluate_tool_permission


class _FakeAgent:
    def __init__(
        self,
        *,
        mode: str = "act",
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._active_mode_snapshot = mode
        self.options = SimpleNamespace(
            project_root=project_root or Path.cwd(),
            permission_rules_allow=allow,
            permission_rules_deny=deny,
        )

    def get_mode(self) -> str:
        return str(self._active_mode_snapshot)


def test_plan_mode_denies_bash() -> None:
    agent = _FakeAgent(mode="plan")
    allowed, reason = evaluate_tool_permission(
        agent,
        tool_name="Bash",
        args={"args": ["pwd"]},
    )
    assert not allowed
    assert reason is not None


def test_plan_mode_denies_edit_outside_plan_root() -> None:
    agent = _FakeAgent(mode="plan", project_root=Path("/tmp/proj"))
    allowed, reason = evaluate_tool_permission(
        agent,
        tool_name="Edit",
        args={"file_path": "src/main.py"},
    )
    assert not allowed
    assert reason is not None


def test_permissions_deny_rule_blocks_matching_call() -> None:
    agent = _FakeAgent(
        mode="act",
        deny=["Read(./.env)"],
        project_root=Path("/tmp/proj"),
    )
    allowed, reason = evaluate_tool_permission(
        agent,
        tool_name="Read",
        args={"file_path": ".env"},
    )
    assert not allowed
    assert reason is not None
