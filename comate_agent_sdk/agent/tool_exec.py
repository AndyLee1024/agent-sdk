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

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


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
        subagent_name=agent.name if agent._is_subagent else None,
        token_cost=agent._token_cost,
        llm_levels=agent.llm_levels,  # type: ignore[arg-type]
        agent_context=agent._context,
    ):
        try:
            # Parse arguments
            args = json.loads(tool_call.function.arguments)

            # Execute the tool (with dependency overrides if configured)
            result = await tool.execute(_overrides=agent.dependency_overrides, **args)

            # Heuristic error detection:
            # Many tools return "Error: ..." strings instead of raising exceptions.
            is_error = False
            if isinstance(result, str):
                if result.lstrip().lower().startswith("error:"):
                    is_error = True

            # Check if the tool is marked as ephemeral (can be bool or int for keep count)
            is_ephemeral = bool(tool.ephemeral)  # Convert int to bool (2 -> True)

            tool_message = ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=result,
                is_error=is_error,
                ephemeral=is_ephemeral,
            )

            # Set span output
            if Laminar is not None:
                if is_error:
                    Laminar.set_span_output(
                        {
                            "error": result[:500]
                            if isinstance(result, str)
                            else str(result)[:500]
                        }
                    )
                    return tool_message
                Laminar.set_span_output(
                    {
                        "result": result[:500] if isinstance(result, str) else str(result)[:500]
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
