from __future__ import annotations

import json
from typing import Any

from comate_agent_sdk.system_tools.tool_result import is_tool_result_envelope
from comate_agent_sdk.system_tools.formatters.types import FormattedToolResult, ToolExecutionMeta
from comate_agent_sdk.system_tools.formatters.read_formatter import format_read_result
from comate_agent_sdk.system_tools.formatters.write_formatter import format_write_like_result
from comate_agent_sdk.system_tools.formatters.search_formatter import (
    format_glob_result,
    format_grep_result,
    format_ls_result,
)
from comate_agent_sdk.system_tools.formatters.bash_formatter import format_bash_result
from comate_agent_sdk.system_tools.formatters.misc_formatter import (
    format_ask_user_question_result,
    format_error,
    format_generic_ok,
    format_todo_result,
    format_webfetch_result,
)

# Re-export types for backward compatibility
__all__ = ["OutputFormatter", "FormattedToolResult", "ToolExecutionMeta"]


_TOOL_FORMATTERS = {
    "Read": format_read_result,
    "Write": format_write_like_result,
    "Edit": format_write_like_result,
    "MultiEdit": format_write_like_result,
    "Glob": format_glob_result,
    "Grep": format_grep_result,
    "LS": format_ls_result,
    "Bash": format_bash_result,
    "TodoWrite": format_todo_result,
    "WebFetch": format_webfetch_result,
    "AskUserQuestion": format_ask_user_question_result,
}


class OutputFormatter:
    """工具输出格式化器"""

    @staticmethod
    def format(
        *,
        tool_name: str,
        tool_call_id: str,
        result_dict: dict[str, Any],
    ) -> FormattedToolResult:
        if not is_tool_result_envelope(result_dict):
            text = (
                json.dumps(result_dict, ensure_ascii=False, indent=2)
                if isinstance(result_dict, dict)
                else str(result_dict)
            )
            return FormattedToolResult(
                text=text,
                meta=ToolExecutionMeta(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="ok",
                ),
            )

        if not bool(result_dict.get("ok", False)):
            return format_error(tool_name, tool_call_id, result_dict)

        formatter = _TOOL_FORMATTERS.get(tool_name, format_generic_ok)
        return formatter(tool_name, tool_call_id, result_dict)
