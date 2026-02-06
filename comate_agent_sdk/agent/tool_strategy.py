"""Tool Strategy 提示生成

参考 subagent/prompts.py 和 skill/prompts.py 的架构模式。

动态生成 <system_tools_definition> 段，包含：
1. <tool_overview>: 所有工具的概览列表
2. 每个工具的详细使用规则（从 Tool.usage_rules 获取）
"""

from comate_agent_sdk.tools.decorator import Tool

TOOL_STRATEGY_TEMPLATE = """
<system_tools_definition>

<tool_overview>
You have access to the following built-in tools for interacting with the environment:
{tool_overview}
</tool_overview>

{tool_details}

</system_tools_definition>
"""


def generate_tool_strategy(tools: list[Tool]) -> str:
    """根据 Agent 实际注册的工具动态生成 <system_tools_definition> 内容

    只为有 usage_rules 的工具生成详细说明。
    没有 usage_rules 的工具（如自定义工具）不会出现在此段中。

    Args:
        tools: Agent 当前注册的 Tool 列表

    Returns:
        格式化的工具策略提示文本，若无工具有 usage_rules 则返回空字符串

    Example:
        >>> from comate_agent_sdk.system_tools.tools import SYSTEM_TOOLS
        >>> prompt = generate_tool_strategy(SYSTEM_TOOLS)
        >>> assert "<system_tools_definition>" in prompt
        >>> assert "Bash" in prompt
    """
    # 只处理有 usage_rules 的工具
    tools_with_rules = [t for t in tools if t.usage_rules]

    if not tools_with_rules:
        return ""

    # 生成概览列表：- **ToolName**: 短描述
    overview_lines = [f"- **{t.name}**: {t.description}" for t in tools_with_rules]

    # 生成详细说明块：<tool name="ToolName">详细规则</tool>
    detail_blocks = [
        f'<tool name="{t.name}">\n{t.usage_rules}\n</tool>'
        for t in tools_with_rules
    ]

    return TOOL_STRATEGY_TEMPLATE.format(
        tool_overview="\n".join(overview_lines),
        tool_details="\n\n".join(detail_blocks),
    )
