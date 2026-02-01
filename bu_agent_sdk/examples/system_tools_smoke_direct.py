import asyncio
import logging
import os
import uuid
from pathlib import Path

from bu_agent_sdk.agent.llm_levels import resolve_llm_levels
from bu_agent_sdk.system_tools.tools import LS, MultiEdit, TodoWrite, WebFetch
from bu_agent_sdk.tools.system_context import bind_system_tool_context
from bu_agent_sdk.tokens import TokenCost

logger = logging.getLogger("bu_agent_sdk.examples.system_tools_smoke_direct")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    session_id = os.getenv("BU_AGENT_SDK_SMOKE_SESSION_ID") or f"smoke-{uuid.uuid4().hex[:8]}"
    session_root = (Path.home() / ".agent" / "sessions" / session_id).resolve()
    project_root = Path.cwd().resolve()

    llm_levels = resolve_llm_levels(explicit=None)  # env-only
    low_model = llm_levels["LOW"].model

    token_cost = TokenCost(include_cost=False)

    logger.info(f"Session: {session_id}")
    logger.info(f"Session root: {session_root}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"LOW LLM: {low_model}")

    with bind_system_tool_context(
        project_root=project_root,
        session_id=session_id,
        session_root=session_root,
        token_cost=token_cost,
        llm_levels=llm_levels,
    ):
        # MultiEdit: create a new file by using first edit with old_string == ""
        tmp_dir = Path("/tmp") / "bu_agent_sdk_smoke"
        tmp_file = (tmp_dir / "hello.txt").resolve()

        multi_out = await MultiEdit.execute(
            file_path=str(tmp_file),
            edits=[
                {"old_string": "", "new_string": "hello\n", "replace_all": False},
                {"old_string": "hello", "new_string": "hi", "replace_all": False},
            ],
        )
        logger.info(f"MultiEdit -> {multi_out}")

        # LS: list /tmp/bu_agent_sdk_smoke
        ls_out = await LS.execute(path=str(tmp_dir), ignore=["*.log"])
        logger.info(f"LS -> {ls_out}")

        # TodoWrite: persist to ~/.agent/sessions/{session_id}/todos.json
        todo_out = await TodoWrite.execute(
            todos=[
                {"content": "verify system tools", "status": "in_progress", "id": "1"},
                {"content": "verify webfetch usage", "status": "pending", "id": "2"},
            ]
        )
        logger.info(f"TodoWrite -> {todo_out}")

        # WebFetch: real network fetch + LOW model extraction
        before = len(token_cost.usage_history)
        answer = await WebFetch.execute(
            url="https://example.com",
            prompt="用中文简要总结这个网页的主题，并列出 3 个关键点。",
        )
        after = len(token_cost.usage_history)
        logger.info(f"WebFetch answer -> {answer[:500]}")
        logger.info(f"TokenCost entries: {before} -> {after}")
        if after == before:
            logger.warning(
                "WebFetch 没有产生 usage 记录；请确认已配置 LOW 模型对应 provider 的 API key"
            )


if __name__ == "__main__":
    asyncio.run(main())
