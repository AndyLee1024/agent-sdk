from __future__ import annotations

from typing import Any

from comate_agent_sdk.llm.messages import ToolMessage
from comate_agent_sdk.system_tools.tool_result import is_tool_result_envelope


def extract_diff_metadata(tool_name: str, tool_result: ToolMessage) -> dict[str, Any] | None:
    if tool_name not in ("Edit", "MultiEdit") or tool_result.is_error:
        return None

    payload = tool_result.raw_envelope
    if not isinstance(payload, dict):
        return None

    data = payload.get("data", {})
    if not isinstance(data, dict):
        return None

    meta: dict[str, Any] = {}
    diff_lines = data.get("diff")
    if isinstance(diff_lines, list) and len(diff_lines) > 0:
        meta["diff"] = diff_lines

    start_line = data.get("start_line")
    end_line = data.get("end_line")
    if isinstance(start_line, int) and start_line > 0:
        meta["start_line"] = start_line
        meta["end_line"] = end_line if isinstance(end_line, int) else start_line

    return meta or None


def extract_todos(tool_result: ToolMessage) -> list[dict]:
    payload = tool_result.raw_envelope
    if not is_tool_result_envelope(payload):
        return []

    data = payload.get("data", {})
    if not isinstance(data, dict):
        return []

    raw_todos = data.get("todos", [])
    if not isinstance(raw_todos, list):
        return []

    return [todo for todo in raw_todos if isinstance(todo, dict)]


def extract_questions(tool_result: ToolMessage) -> list[dict]:
    payload = tool_result.raw_envelope
    if not is_tool_result_envelope(payload):
        return []

    data = payload.get("data", {})
    if not isinstance(data, dict):
        return []

    raw_questions = data.get("questions", [])
    if not isinstance(raw_questions, list):
        return []

    return [question for question in raw_questions if isinstance(question, dict)]


def extract_plan_approval(tool_result: ToolMessage) -> dict[str, str] | None:
    payload = tool_result.raw_envelope
    if not is_tool_result_envelope(payload):
        return None

    data = payload.get("data", {})
    if not isinstance(data, dict):
        return None

    status = str(data.get("status", "")).strip().lower()
    if status != "waiting_for_plan_approval":
        return None

    plan_path = str(data.get("plan_path", "")).strip()
    summary = str(data.get("summary", "")).strip()
    execution_prompt = str(data.get("execution_prompt", "")).strip()
    if not execution_prompt:
        execution_prompt = "请基于已批准的计划开始执行。"
    if not plan_path:
        return None
    return {
        "plan_path": plan_path,
        "summary": summary,
        "execution_prompt": execution_prompt,
    }
