"""
Subagent 系统提示模板
"""

from comate_agent_sdk.subagent.models import AgentDefinition

SUBAGENT_STRATEGY_PROMPT = """
<subagent>
{subagent_list}
</subagent>"""


def generate_subagent_prompt(agents: list[AgentDefinition]) -> str:
    """生成 Subagent 策略提示

    Args:
        agents: AgentDefinition 列表

    Returns:
        完整的系统提示字符串
    """
    subagent_list = "\n".join([f"- **{a.name}**: {a.description}" for a in agents])
    return SUBAGENT_STRATEGY_PROMPT.format(subagent_list=subagent_list)
