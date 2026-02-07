"""
Example demonstrating Claude Code-style tools with sandboxed filesystem.

Includes bash, file operations (read/write/edit), search (glob/grep),
todo management - all with dependency injection
for secure filesystem access.

Run with:
    python -m comate_agent_sdk.examples.claude_code
"""

import asyncio
import logging
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import (
    AgentConfig,
    StopEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from comate_agent_sdk.agent.events import StepCompleteEvent, StepStartEvent
from comate_agent_sdk.llm import ChatAnthropic
from comate_agent_sdk.tools import Depends, tool

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("comate_agent_sdk.examples.claude_code")


# =============================================================================
# Sandbox Context - Filesystem management with security
# =============================================================================


class SecurityError(Exception):
    """Raised when a path escapes the sandbox."""

    pass


@dataclass
class SandboxContext:
    """Sandboxed filesystem context. All file operations are restricted to root_dir."""

    root_dir: Path
    working_dir: Path
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @classmethod
    def create(cls, root_dir: Path | str | None = None) -> "SandboxContext":
        """Create a new sandbox context, defaulting to a temp directory."""
        session_id = str(uuid.uuid4())[:8]
        if root_dir is None:
            root = Path(f"./tmp/sandbox/{session_id}")
        else:
            root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        return cls(root_dir=root, working_dir=root, session_id=session_id)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve and validate a path is within the sandbox."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (self.working_dir / path_obj).resolve()

        # Security check: ensure path is within sandbox
        try:
            resolved.relative_to(self.root_dir)
        except ValueError:
            raise SecurityError(f"Path escapes sandbox: {path} -> {resolved}")
        return resolved


def get_sandbox_context() -> SandboxContext:
    """Dependency injection marker. Override this in the agent."""
    raise RuntimeError(
        "get_sandbox_context() must be overridden via dependency_overrides"
    )


# =============================================================================
# Bash Tool
# =============================================================================


@tool("Execute a shell command and return output")
async def bash(
    command: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    timeout: int = 30,
) -> str:
    """Run a bash command in the sandbox working directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ctx.working_dir),
        )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# File Operations
# =============================================================================


@tool("Read contents of a file")
async def read(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Read a file and return its contents with line numbers."""
    try:
        path = ctx.resolve_path(file_path)
    except SecurityError as e:
        return f"Security error: {e}"

    if not path.exists():
        return f"File not found: {file_path}"
    if path.is_dir():
        return f"Path is a directory: {file_path}"
    try:
        lines = path.read_text().splitlines()
        numbered = [f"{i + 1:4d}  {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


@tool("Write content to a file")
async def write(
    file_path: str,
    content: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Write content to a file, creating directories if needed."""
    try:
        path = ctx.resolve_path(file_path)
    except SecurityError as e:
        return f"Security error: {e}"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool("Replace text in a file")
async def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Replace old_string with new_string in a file."""
    try:
        path = ctx.resolve_path(file_path)
    except SecurityError as e:
        return f"Security error: {e}"

    if not path.exists():
        return f"File not found: {file_path}"
    try:
        content = path.read_text()
        if old_string not in content:
            return f"String not found in {file_path}"
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string)
        path.write_text(new_content)
        return f"Replaced {count} occurrence(s) in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


# =============================================================================
# Search Tools
# =============================================================================


@tool("Find files matching a glob pattern")
async def glob_search(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Find files matching a glob pattern like **/*.py"""
    try:
        search_dir = ctx.resolve_path(path) if path else ctx.working_dir
    except SecurityError as e:
        return f"Security error: {e}"

    try:
        matches = list(search_dir.glob(pattern))
        files = [str(m.relative_to(ctx.root_dir)) for m in matches if m.is_file()][:50]
        if not files:
            return f"No files match pattern: {pattern}"
        return f"Found {len(files)} file(s):\n" + "\n".join(files)
    except Exception as e:
        return f"Error: {e}"


@tool("Search file contents with regex")
async def grep(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Search for pattern in files recursively."""
    try:
        search_dir = ctx.resolve_path(path) if path else ctx.working_dir
    except SecurityError as e:
        return f"Security error: {e}"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []
    for file_path in search_dir.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            for i, line in enumerate(file_path.read_text().splitlines(), 1):
                if regex.search(line):
                    rel_path = file_path.relative_to(ctx.root_dir)
                    results.append(f"{rel_path}:{i}: {line[:100]}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n... (truncated)"
        except Exception:
            pass  # Skip binary/unreadable files
    return "\n".join(results) if results else f"No matches for: {pattern}"


# =============================================================================
# Todo Tools (session-scoped via context)
# =============================================================================

_todos: dict[str, list[dict]] = {}  # session_id -> todos


@tool("Read current todo list")
async def todo_read(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Get the current todo list."""
    todos = _todos.get(ctx.session_id, [])
    if not todos:
        return "Todo list is empty"
    lines = []
    for i, t in enumerate(todos, 1):
        status = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[
            t["status"]
        ]
        lines.append(f"{i}. {status} {t['content']}")
    return "\n".join(lines)


@tool("Update the todo list")
async def todo_write(
    todos: list[dict],
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Set the todo list. Each item needs: content, status, activeForm"""
    _todos[ctx.session_id] = todos
    stats = {
        "pending": sum(1 for t in todos if t.get("status") == "pending"),
        "in_progress": sum(1 for t in todos if t.get("status") == "in_progress"),
        "completed": sum(1 for t in todos if t.get("status") == "completed"),
    }
    return f"Updated todos: {stats['pending']} pending, {stats['in_progress']} in progress, {stats['completed']} completed"


# =============================================================================
# All Tools
# =============================================================================

ALL_TOOLS = [bash, read, write, edit, glob_search, grep, todo_read, todo_write]


# =============================================================================
# Main
# =============================================================================


async def main():
    # Create a sandbox context
    ctx = SandboxContext.create()
    logger.info(f"🏗️  Sandbox: {ctx.root_dir}")

    # Create some test files in the sandbox
    (ctx.root_dir / "hello.py").write_text('print("Hello, World!")\n')
    (ctx.root_dir / "utils.py").write_text("def add(a, b):\n    return a + b\n")

    # Create agent with dependency override for the sandbox context
    agent = Agent(
        llm=ChatAnthropic(model="gemini-3-pro-low", base_url="http://127.0.0.1:8045/", api_key="sk-b3e2affa66e5466c9952c8e768e7ba8f"),
        config=AgentConfig(
            tools=ALL_TOOLS,
            system_prompt=f"You are a coding assistant. Working directory: {ctx.working_dir}。 你总是需要使用中文回复用户",
            dependency_overrides={get_sandbox_context: lambda: ctx},
        ),
    )

    logger.info("🚀 Starting query...\n")
    async for event in agent.query_stream(
        "使用frontend design 来设计一个登录页面, write为login.html"
    ):
        match event:
            case ThinkingEvent(content=text):
                preview = text[:100] + "..." if len(text) > 100 else text
                logger.info(f"🧠 {preview}")
            case TextEvent(content=text):
                logger.info(text)
            case StepStartEvent(step_number=n, title=title):
                logger.info(f"\n▶️  Step {n}: {title}")
            case ToolCallEvent(tool=name, args=args):
                logger.info(f"🔧 [{name}] {args}")
            case ToolResultEvent(tool=name, result=result):
                result_str = str(result)
                logger.info(
                    f"  ✅ {result_str[:200]}..."
                    if len(result_str) > 200
                    else f"  ✅ {result_str}"
                )
            case StepCompleteEvent(status=status, duration_ms=ms):
                icon = "✅" if status == "completed" else "❌"
                logger.info(f"  {icon} {status} ({ms:.0f}ms)")
            case StopEvent(reason=reason):
                logger.info(f"\n🏁 stop={reason}")


if __name__ == "__main__":
    asyncio.run(main())
