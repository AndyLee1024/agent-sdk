from __future__ import annotations

import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger("comate_agent_sdk.context.env")


@dataclass(frozen=True)
class EnvOptions:
    """环境信息配置选项（用于初始化时快照）"""

    system_env: bool = False
    """是否启用系统环境信息"""
    git_env: bool = False
    """是否启用 Git 状态信息"""
    working_dir: Path | None = None
    """工作目录（默认：agent.project_root > cwd）"""
    git_status_limit: int = 10
    """git status 最大保留行数（超过截断）"""
    git_log_limit: int = 6
    """git log 最大保留行数（超过截断）"""


@dataclass(frozen=True)
class EnvProvider:
    """环境信息提供者（负责采集并格式化）"""

    git_status_limit: int = 10
    git_log_limit: int = 6

    def get_system_env(self, working_dir: Path) -> str:
        """获取系统环境信息，返回 <system_env>...</system_env> 格式"""
        normalized_dir = working_dir.expanduser().resolve()
        is_git_repo = self._is_git_repo(normalized_dir)

        tz_name = time.tzname[0] if time.tzname else "Unknown"

        lines = [
            "<system_env>",
            f"project root : {normalized_dir}",
            f"is git repo: {'Yes' if is_git_repo else 'No'}",
            f"platform: {sys.platform}",
            f"OS Version: {platform.system()} {platform.release()}",
            f"today's date: {date.today().isoformat()}",
            f"user timezone: {tz_name}",
            f"python version: {platform.python_version()}",
            "</system_env> \n",
        ]
        return "\n".join(lines)

    def get_git_env(self, working_dir: Path) -> str | None:
        """获取 Git 状态信息，返回 <git_env>...</git_env> 格式

        若 working_dir 不在 git 仓库中，返回 None。
        """
        normalized_dir = working_dir.expanduser().resolve()
        if not self._is_git_repo(normalized_dir):
            return None

        branch = self._git_current_branch(normalized_dir)
        status_text = self._git_status_text(normalized_dir, limit=self.git_status_limit)
        log_text = self._git_recent_commits_text(normalized_dir, limit=self.git_log_limit)

        lines = [
            "<git_env>",
            "git_env: this is a snapshot of the git status at the start of the session. Please note that this status will not be automatically updated during the session.",
            f"Current Branch: {branch}",
            "",
            "Status:",
            status_text,
            "",
            "Recent Commits:",
            log_text,
            "</git_env> \n",
        ]
        return "\n".join(lines)

    def _run_git(self, args: list[str], cwd: Path) -> str | None:
        cmd = ["git", *args]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            logger.warning("未找到 git 可执行文件，无法采集 GIT_ENV")
            return None
        except Exception:
            logger.exception(f"执行 git 命令失败: {cmd}")
            return None

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            if stderr:
                logger.debug(f"git 命令返回非零: {cmd}, stderr={stderr}")
            return None

        return (proc.stdout or "").rstrip("\n")

    def _is_git_repo(self, cwd: Path) -> bool:
        out = self._run_git(["rev-parse", "--is-inside-work-tree"], cwd=cwd)
        return (out or "").strip().lower() == "true"

    def _git_current_branch(self, cwd: Path) -> str:
        branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
        if not branch:
            return "unknown"
        if branch.strip() != "HEAD":
            return branch.strip()

        sha = self._run_git(["rev-parse", "--short", "HEAD"], cwd=cwd) or "unknown"
        return f"detached@{sha.strip()}"

    def _git_status_text(self, cwd: Path, *, limit: int) -> str:
        out = self._run_git(["status", "--porcelain"], cwd=cwd)
        if out is None:
            return "（无法获取 git status）"

        lines = [ln.rstrip() for ln in out.splitlines() if ln.strip()]
        if not lines:
            return "（无变更）"

        if limit <= 0 or len(lines) <= limit:
            return "\n".join(lines)

        kept = lines[:limit]
        omitted = len(lines) - limit
        kept.append(f"...（已截断，省略 {omitted} 行）")
        return "\n".join(kept)

    def _git_recent_commits_text(self, cwd: Path, *, limit: int) -> str:
        if limit <= 0:
            return "（已关闭）"

        out = self._run_git(
            ["log", f"-n{limit}", "--pretty=format:%h %s"],
            cwd=cwd,
        )
        if out is None:
            return "（无法获取 git log）"

        lines = [ln.rstrip() for ln in out.splitlines() if ln.strip()]
        return "\n".join(lines) if lines else "（无提交记录）"

