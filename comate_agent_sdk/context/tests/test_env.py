import subprocess
from pathlib import Path

import pytest

from comate_agent_sdk.context.env import EnvProvider
from comate_agent_sdk.context.info import _build_categories
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )


def _init_repo(repo_dir: Path) -> None:
    _run_git(["init"], repo_dir)
    _run_git(["config", "user.email", "test@example.com"], repo_dir)
    _run_git(["config", "user.name", "test"], repo_dir)


def _commit_file(repo_dir: Path, name: str, content: str, message: str) -> None:
    p = repo_dir / name
    p.write_text(content, encoding="utf-8")
    _run_git(["add", name], repo_dir)
    _run_git(["commit", "-m", message], repo_dir)


def test_env_provider_system_env_and_git_env_limits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    for i in range(7):
        _commit_file(repo, f"f{i}.txt", f"v{i}", f"c{i}")

    for i in range(12):
        (repo / f"u{i}.txt").write_text("x", encoding="utf-8")

    provider = EnvProvider(git_status_limit=10, git_log_limit=6)

    system_env = provider.get_system_env(repo)
    assert "<system_env>" in system_env
    assert str(repo.resolve()) in system_env

    git_env = provider.get_git_env(repo)
    assert git_env is not None
    assert "<git_env>" in git_env
    assert "Current Branch:" in git_env

    lines = git_env.splitlines()

    status_start = lines.index("Status:") + 1
    commits_start = lines.index("Recent Commits:")
    status_lines = [ln for ln in lines[status_start:commits_start] if ln.strip()]
    assert len(status_lines) == 11
    assert status_lines[-1].startswith("...（已截断，省略 ")

    commit_lines = [
        ln
        for ln in lines[commits_start + 1 :]
        if ln.strip() and not ln.strip().startswith("</git_env>")
    ]
    assert len(commit_lines) == 6


def test_context_ir_session_state_env_order_is_stable() -> None:
    ctx = ContextIR()
    ctx.set_system_env("<system_env>\nX\n</system_env>")
    ctx.set_skill_strategy("SKILL")

    header_types = [item.item_type for item in ctx.header.items]
    assert header_types == [ItemType.SKILL_STRATEGY]

    state_types = [item.item_type for item in ctx.session_state.items]
    assert state_types == [ItemType.SYSTEM_ENV]

    ctx.set_git_env("<git_env>\nY\n</git_env>")
    state_types = [item.item_type for item in ctx.session_state.items]
    assert state_types == [ItemType.SYSTEM_ENV, ItemType.GIT_ENV]


def test_context_info_categories_include_env() -> None:
    ctx = ContextIR()
    ctx.set_system_env("<system_env>\nX\n</system_env>")
    ctx.set_git_env("<git_env>\nY\n</git_env>")

    status = ctx.get_budget_status()
    categories = _build_categories(status.tokens_by_type, ctx)
    labels = {c.label for c in categories}
    assert "System Env" in labels
    assert "Git Env" in labels
