import asyncio
import json
import tempfile
from pathlib import Path

from comate_agent_sdk.system_tools.policy_engine import (
    BASH_COMMAND_POLICY,
    READ_REGISTRY_POLICY,
)


def test_bash_policy_denies_banned_command() -> None:
    violation = BASH_COMMAND_POLICY.validate(
        args=["cat", "a.txt"],
        cwd=Path.cwd(),
        allowed_roots=[Path.cwd()],
    )
    assert violation is not None
    assert violation.code == "POLICY_DENIED"


def test_bash_policy_requires_rg_max_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        violation = BASH_COMMAND_POLICY.validate(
            args=["rg", "--line-number", "--no-heading", "--color=never", "foo", str(root)],
            cwd=root,
            allowed_roots=[root],
        )
        assert violation is not None
        assert violation.code == "POLICY_DENIED"


def test_bash_policy_accepts_valid_rg_scope() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        violation = BASH_COMMAND_POLICY.validate(
            args=[
                "rg",
                "--line-number",
                "--no-heading",
                "--color=never",
                "--max-count",
                "10",
                "foo",
                str(root),
            ],
            cwd=root,
            allowed_roots=[root],
        )
        assert violation is None


def test_read_registry_policy_persists_to_session_root() -> None:
    READ_REGISTRY_POLICY.clear_memory()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        session_root = root / "sessions" / "s1"
        target = root / "a.txt"
        target.write_text("x\n", encoding="utf-8")

        async def _run() -> bool:
            await READ_REGISTRY_POLICY.mark_read(
                session_root=session_root,
                session_id="s1",
                path=target,
            )
            return await READ_REGISTRY_POLICY.has_read(
                session_root=session_root,
                session_id="s1",
                path=target,
            )

        assert asyncio.run(_run())
        idx = session_root / "read_index.json"
        assert idx.exists()
        payload = json.loads(idx.read_text(encoding="utf-8"))
        assert payload.get("schema_version") == 1
        assert str(target.resolve()) in payload.get("files", [])
