from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from comate_agent_sdk.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ToolCall,
    ToolMessage,
)
from comate_agent_sdk.observability import Laminar
from comate_agent_sdk.system_tools.output_formatter import OutputFormatter
from comate_agent_sdk.system_tools.tool_result import is_tool_result_envelope

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


def _detect_tool_error(result: str | list[ContentPartTextParam | ContentPartImageParam]) -> bool:
    if isinstance(result, list):
        return False

    text = result.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if is_tool_result_envelope(payload):
            return not bool(payload.get("ok", False))

    return text.lower().startswith("error:")


def _extract_tool_envelope(
    result: str | list[ContentPartTextParam | ContentPartImageParam] | dict,
) -> dict | None:
    """从工具返回值中提取标准 envelope。"""
    if isinstance(result, dict) and is_tool_result_envelope(result):
        return result

    if not isinstance(result, str):
        return None

    text = result.strip()
    if not text or not text.startswith("{"):
        return None

    try:
        payload = json.loads(text)
    except Exception:
        return None

    if is_tool_result_envelope(payload):
        return payload
    return None


async def execute_tool_call(agent: "AgentRuntime", tool_call: ToolCall) -> ToolMessage:
    """执行单个 tool call，返回 ToolMessage。"""
    tool_name = tool_call.function.name

    # Check if this is a Skill tool call (special handling)
    if tool_name == "Skill":
        from comate_agent_sdk.agent.setup import execute_skill_call

        return await execute_skill_call(agent, tool_call)

    tool = agent._tool_map.get(tool_name)

    if tool is None:
        # 收集诊断信息
        available_tools = list(agent._tool_map.keys())
        mcp_tools_available = [t for t in available_tools if t.startswith("mcp__")]

        error_msg = f"Error: Unknown tool '{tool_name}'."
        if tool_name.startswith("mcp__"):
            if mcp_tools_available:
                error_msg += f" Available MCP tools: {mcp_tools_available}"
            else:
                error_msg += " No MCP tools are currently loaded. Check MCP server connection."

        logger.warning(f"工具执行失败: {error_msg}")
        return ToolMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            content=error_msg,
            is_error=True,
        )

    # Create Laminar span for tool execution
    if Laminar is not None:
        span_context = Laminar.start_as_current_span(
            name=tool_name,
            input={
                "tool": tool_name,
                "arguments": tool_call.function.arguments,
            },
            span_type="TOOL",
        )
    else:
        span_context = nullcontext()

    from comate_agent_sdk.tools.system_context import bind_system_tool_context

    project_root = (agent.project_root or Path.cwd()).resolve()
    if agent.offload_root_path:
        session_root = Path(agent.offload_root_path).expanduser().resolve().parent
    else:
        session_root = (Path.home() / ".agent" / "sessions" / agent._session_id).resolve()

    with span_context, bind_system_tool_context(
        project_root=project_root,
        session_id=agent._session_id,
        session_root=session_root,
        workspace_root=project_root,  # CLI 场景：新文件直接写入 project_root
        subagent_name=agent.name if agent._is_subagent else None,
        tool_call_id=tool_call.id,
        subagent_source_prefix=agent._subagent_source_prefix,
        token_cost=agent._token_cost,
        llm_levels=agent.llm_levels,  # type: ignore[arg-type]
        agent_context=agent._context,
    ):
        try:
            # Parse arguments
            args = json.loads(tool_call.function.arguments)

            # Execute the tool (with dependency overrides if configured)
            result = await tool.execute(_overrides=agent.dependency_overrides, **args)

            raw_envelope = _extract_tool_envelope(result)
            execution_meta = None
            formatted_content = result
            is_error = False
            truncation_record = None

            if raw_envelope is not None:
                formatted = OutputFormatter.format(
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    result_dict=raw_envelope,
                )
                formatted_content = formatted.text
                execution_meta = formatted.meta.to_dict()
                truncation_record = formatted.meta.truncation
                is_error = formatted.meta.status == "error"
            else:
                if isinstance(result, (str, list)):
                    is_error = _detect_tool_error(result)
                    formatted_content = result
                else:
                    formatted_content = str(result)
                    is_error = False

            # Check if the tool is marked as ephemeral (can be bool or int for keep count)
            is_ephemeral = bool(tool.ephemeral)  # Convert int to bool (2 -> True)

            tool_message = ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=formatted_content,
                is_error=is_error,
                ephemeral=is_ephemeral,
                raw_envelope=raw_envelope,
                execution_meta=execution_meta,
                truncation_record=truncation_record,
            )

            # Set span output
            if Laminar is not None:
                if is_error:
                    Laminar.set_span_output(
                        {
                            "error": formatted_content[:500]
                            if isinstance(formatted_content, str)
                            else str(formatted_content)[:500]
                        }
                    )
                    return tool_message
                Laminar.set_span_output(
                    {
                        "result": formatted_content[:500]
                        if isinstance(formatted_content, str)
                        else str(formatted_content)[:500]
                    }
                )

            return tool_message

        except json.JSONDecodeError as e:
            error_msg = f"Error parsing arguments: {e}"
            if Laminar is not None:
                Laminar.set_span_output({"error": error_msg})
            return ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=error_msg,
                is_error=True,
            )
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            if Laminar is not None:
                Laminar.set_span_output({"error": error_msg})
            return ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=error_msg,
                is_error=True,
            )


def extract_screenshot(tool_message: ToolMessage) -> str | None:
    """Extract screenshot base64 from a tool message if present."""
    content = tool_message.content

    # If content is a string, no screenshot
    if isinstance(content, str):
        return None

    # If content is a list of content parts, look for images
    if isinstance(content, list):
        for part in content:
            # Check if it's an image content part
            if hasattr(part, "type") and part.type == "image_url":
                image_url = getattr(part, "image_url", None)
                if image_url:
                    url = getattr(image_url, "url", "")
                    if not url and isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    if url.startswith("data:image/png;base64,"):
                        return url.split(",", 1)[1]
                    if url.startswith("data:image/jpeg;base64,"):
                        return url.split(",", 1)[1]
            # Handle dict format
            elif isinstance(part, dict) and part.get("type") == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:image/png;base64,"):
                    return url.split(",", 1)[1]
                if url.startswith("data:image/jpeg;base64,"):
                    return url.split(",", 1)[1]

    return None


ChatMessage = str | list[ContentPartTextParam | ContentPartImageParam]
