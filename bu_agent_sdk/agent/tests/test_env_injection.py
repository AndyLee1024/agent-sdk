import subprocess
from pathlib import Path

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import ComateAgentOptions
from bu_agent_sdk.context import EnvOptions


class _FakeChatModel:
    def __init__(self):
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("This test should not call the LLM")


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
    (repo_dir / "a.txt").write_text("a", encoding="utf-8")
    _run_git(["add", "a.txt"], repo_dir)
    _run_git(["commit", "-m", "init"], repo_dir)


def test_agent_injects_env_snapshot_from_project_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    agent = Agent(
        llm=_FakeChatModel(),  # type: ignore[arg-type]
        options=ComateAgentOptions(
            tools=[],
            agents=[],
            offload_enabled=False,
            project_root=repo,
            setting_sources=None,
            env_options=EnvOptions(system_env=True, git_env=True),
        ),
    )

    system_msg = agent.messages[0]
    header = system_msg.text

    assert "<system_env>" in header
    assert "<git_env>" in header
    assert str(repo.resolve()) in header
    assert header.index("<system_env>") < header.index("<git_env>")
