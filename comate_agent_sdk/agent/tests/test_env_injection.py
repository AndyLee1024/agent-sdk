import subprocess
from pathlib import Path

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.context import EnvOptions
from comate_agent_sdk.llm.messages import UserMessage


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

    template = Agent(
        llm=_FakeChatModel(),  # type: ignore[arg-type]
        config=AgentConfig(
            tools=(),
            agents=(),
            offload_enabled=False,
            project_root=repo,
            setting_sources=None,
            env_options=EnvOptions(system_env=True, git_env=True),
        ),
    )
    agent = template.create_runtime()

    session_state_texts = [
        m.text
        for m in agent.messages
        if isinstance(m, UserMessage) and bool(getattr(m, "is_meta", False))
    ]
    merged = "\n".join(session_state_texts)

    assert "<system_env>" in merged
    assert "<git_env>" in merged
    assert str(repo.resolve()) in merged
    assert merged.index("<system_env>") < merged.index("<git_env>")
