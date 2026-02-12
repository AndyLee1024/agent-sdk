from __future__ import annotations

import asyncio
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _extract_types(prop_schema: dict[str, Any]) -> set[str]:
    """从 JSON Schema 属性定义中提取期望的类型集合。

    兼容三种常见格式：
    - "type": "integer"                          → {"integer"}
    - "type": ["object", "null"]                  → {"object", "null"}  (OpenAI strict nullable)
    - "anyOf": [{"type": "object"}, {"type": "null"}] → {"object", "null"}  (Pydantic nullable)
    """
    types: set[str] = set()

    # 直接 type 字段
    raw_type = prop_schema.get("type")
    if isinstance(raw_type, str):
        types.add(raw_type)
    elif isinstance(raw_type, list):
        types.update(raw_type)

    # anyOf 格式
    any_of = prop_schema.get("anyOf")
    if isinstance(any_of, list):
        for variant in any_of:
            if isinstance(variant, dict):
                vt = variant.get("type")
                if isinstance(vt, str):
                    types.add(vt)
                elif isinstance(vt, list):
                    types.update(vt)

    return types


def _coerce_value(value: Any, prop_schema: dict[str, Any]) -> Any:
    """单字段类型修复：对照 schema 将 LLM 返回的字符串化值转换为正确类型。

    转换规则：
    - 值已是正确类型 → 原样返回
    - "null"/"none"/"" + schema 允许 null → None
    - string + 期望 integer → int(value)
    - string + 期望 number → float(value)
    - string + 期望 boolean → "true"/"1" → True, "false"/"0" → False
    - string + 期望 array → json.loads(value) + 递归 coerce items
    - string + 期望 object → json.loads(value) + 递归 coerce properties
    - 无法转换 → 原样返回，让 Pydantic 报清晰错误
    """
    expected = _extract_types(prop_schema)

    # 没有类型信息，无法修复
    if not expected:
        return value

    # --- 非 string 值：类型已正确，但可能有嵌套结构需要递归处理 ---
    if not isinstance(value, str):
        return _coerce_non_string(value, prop_schema)

    # --- 以下处理 string 值 → 目标类型的转换 ---
    normalized = value.strip().lower()

    # null 语义识别（优先于 string 快速返回，因为 "null" 在 string|null 场景中应转为 None）
    if normalized in ("null", "none", "") and "null" in expected:
        return None

    # 如果 schema 期望的就是 string，不需要转换
    if "string" in expected:
        return value

    # string → integer
    if "integer" in expected:
        try:
            return int(value)
        except (ValueError, TypeError):
            pass

    # string → number (float)
    if "number" in expected:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

    # string → boolean
    if "boolean" in expected:
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False

    # string → array
    if "array" in expected:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                items_schema = prop_schema.get("items")
                if isinstance(items_schema, dict):
                    return [_coerce_value(item, items_schema) for item in parsed]
                return parsed
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug(f"Coerce string→array json.loads failed: {exc}, value[:100]={value[:100]!r}")

    # string → object
    if "object" in expected:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return _coerce_object(parsed, prop_schema)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug(f"Coerce string→object json.loads failed: {exc}, value[:100]={value[:100]!r}")

    # 无法转换，原样返回让 Pydantic 报清晰错误
    return value


def _coerce_object(obj: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """递归处理 object 类型的属性，对照 schema.properties 逐字段 coerce。"""
    properties = schema.get("properties", {})
    if not properties:
        return obj

    result = {}
    for key, val in obj.items():
        if key in properties:
            result[key] = _coerce_value(val, properties[key])
        else:
            result[key] = val
    return result


def _coerce_non_string(value: Any, prop_schema: dict[str, Any]) -> Any:
    """对非 string 值递归处理嵌套结构（如 list of objects、nested dict）。"""
    if isinstance(value, dict):
        return _coerce_object(value, prop_schema)

    if isinstance(value, list):
        items_schema = prop_schema.get("items")
        if isinstance(items_schema, dict):
            return [_coerce_value(item, items_schema) for item in value]

    return value


def _coerce_tool_arguments(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """入口：遍历工具参数，对照 schema 做类型修复。

    对每个顶层参数字段，查找对应的 schema property 定义，
    调用 _coerce_value 进行类型转换。
    """
    properties = schema.get("properties", {})
    if not properties:
        return args

    coerced = {}
    for key, value in args.items():
        if key in properties:
            new_value = _coerce_value(value, properties[key])
            if new_value is not value:
                logger.debug(f"Coerced tool argument '{key}': {type(value).__name__} → {type(new_value).__name__}")
            coerced[key] = new_value
        else:
            coerced[key] = value
    return coerced


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

            # Schema-aware coerce: 修复非原生 OpenAI 模型返回的字符串化参数
            args = _coerce_tool_arguments(args, tool.definition.parameters)

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
        except asyncio.CancelledError:
            raise
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
