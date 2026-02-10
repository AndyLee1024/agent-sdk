"""Tool Strategy 提示生成

参考 subagent/prompts.py 和 skill/prompts.py 的架构模式。

动态生成 SYSTEM_TOOLS_DEFINITION 段，包含：
工具概览列表（仅简洁说明）
"""

from comate_agent_sdk.agent.tool_visibility import visible_tools
from comate_agent_sdk.tools.decorator import Tool

TOOL_STRATEGY_TEMPLATE = """
<tools>
You have access to the following built-in tools for interacting with the environment:
{tool_overview} 
</tools>"""


def generate_tool_strategy(tools: list[Tool], *, is_subagent: bool = False) -> str:
    """根据 Agent 实际注册的工具动态生成 SYSTEM_TOOLS_DEFINITION 内容

    只为有 usage_rules 的工具生成概览列表。
    没有 usage_rules 的工具（如自定义工具）不会出现在此段中。
    详细规则通过工具定义 description 字段（ToolDefinition.description）传递给模型。

    Args:
        tools: Agent 当前注册的 Tool 列表
        is_subagent: 当前 runtime 是否为 subagent（subagent 需要隐藏 Task/AskUserQuestion）

    Returns:
        格式化的工具策略提示文本，若无工具有 usage_rules 则返回空字符串

    Example:
        >>> from comate_agent_sdk.system_tools.tools import SYSTEM_TOOLS
        >>> prompt = generate_tool_strategy(SYSTEM_TOOLS)
        >>> assert "[SYSTEM_TOOLS_DEFINITION]" in prompt
        >>> assert "Bash" in prompt
    """
    visible = visible_tools(tools, is_subagent=is_subagent)

    # 只处理有 usage_rules 的工具
    tools_with_rules = [t for t in visible if t.usage_rules]

    if not tools_with_rules:
        return ""

    # 生成概览列表：- **ToolName**: 短描述
    overview_lines = [f"- **{t.name}**: {t.description}" for t in tools_with_rules]

    return TOOL_STRATEGY_TEMPLATE.format(
        tool_overview="\n".join(overview_lines),
    )
