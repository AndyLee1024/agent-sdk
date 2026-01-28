"""
Task 工具 - 用于启动 Subagent 的统一入口
"""

import asyncio
import logging
from dataclasses import is_dataclass, replace

from bu_agent_sdk.llm.base import BaseChatModel
from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
from bu_agent_sdk.subagent.models import AgentDefinition
from bu_agent_sdk.tools.decorator import Tool, tool
from bu_agent_sdk.tools.depends import DependencyOverrides
from bu_agent_sdk.tools.registry import ToolRegistry


def create_task_tool(
    agents: list[AgentDefinition],
    tool_registry: ToolRegistry,
    parent_llm: BaseChatModel,
    parent_dependency_overrides: DependencyOverrides | None = None,
) -> Tool:
    """创建 Task 工具，用于启动 Subagent

    Args:
        agents: AgentDefinition 列表
        tool_registry: 全局工具注册表
        parent_llm: 父 Agent 的 LLM 实例
        parent_dependency_overrides: 父 Agent 的依赖覆盖（会继承给 Subagent）

    Returns:
        Task 工具实例
    """
    # 构建 subagent 名称映射
    agent_map = {a.name: a for a in agents}

    @tool("Launch a subagent to handle a specific task.")
    async def Task(
        subagent_type: str,
        prompt: str,
        description: str = "",
    ) -> str:
        """
        Args:
            subagent_type: The name of the subagent to launch (e.g., "code-reviewer", "researcher")
            prompt: The task for the subagent to perform
            description: A short (3-5 word) description of the task
        """
        # 延迟导入避免循环依赖
        from bu_agent_sdk.agent.service import Agent

        if subagent_type not in agent_map:
            available = ", ".join(agent_map.keys())
            return f"Error: Unknown subagent '{subagent_type}'. Available: {available}"

        agent_def = agent_map[subagent_type]

        # 解析模型
        llm = resolve_model(agent_def.model, parent_llm)

        # 解析工具
        tools = tool_registry.filter(agent_def.tools or [])

        logging.info(
            f"Launching subagent '{subagent_type}' with {len(tools)} tools: {agent_def.tools}"
        )

        # 创建 Subagent（继承父级依赖覆盖）
        subagent = Agent(
            llm=llm,
            tools=tools,
            system_prompt=agent_def.prompt,
            max_iterations=agent_def.max_iterations,
            compaction=agent_def.compaction,
            dependency_overrides=parent_dependency_overrides,  # 继承父级
            _is_subagent=True,  # 禁止嵌套
        )

        # Subagent Skills 筛选（如果 AgentDefinition.skills 不为空）
        if agent_def.skills is not None and subagent.skills:
            allowed_skill_names = set(agent_def.skills)
            subagent.skills = [s for s in subagent.skills if s.name in allowed_skill_names]
            # 重新创建 Skill 工具（更新工具描述）
            subagent._rebuild_skill_tool()
            logging.info(
                f"Filtered subagent '{subagent_type}' skills to: {[s.name for s in subagent.skills]}"
            )

        # 执行（带超时）
        try:
            if agent_def.timeout:
                result = await asyncio.wait_for(
                    subagent.query(prompt), timeout=agent_def.timeout
                )
            else:
                result = await subagent.query(prompt)

            logging.info(f"Subagent '{subagent_type}' completed successfully")
            return result

        except asyncio.TimeoutError:
            error_msg = (
                f"Error: Subagent '{subagent_type}' timeout after {agent_def.timeout}s"
            )
            logging.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error in subagent '{subagent_type}': {e}"
            logging.error(error_msg, exc_info=True)
            return error_msg

    return Task


def resolve_model(
    model: str | None,
    parent_llm: BaseChatModel,
) -> BaseChatModel:
    """解析模型配置

    Args:
        model: 模型名称（None/"inherit" 代表继承；也支持任意模型字符串）
        parent_llm: 父 Agent 的 LLM 实例

    Returns:
        解析后的 LLM 实例

    Notes:
        - "sonnet"/"opus"/"haiku" 会使用内置 Anthropic 预设
        - 其他字符串会尽量克隆 parent_llm，仅替换 model 字段
    """
    if model is None:
        return parent_llm

    model = model.strip()
    if not model or model.lower() == "inherit":
        return parent_llm

    # 根据 model 名称创建对应的 LLM
    model_map = {
        "sonnet": lambda: ChatAnthropic(model="claude-sonnet-4-5"),
        "opus": lambda: ChatAnthropic(model="claude-opus-4-5"),
        "haiku": lambda: ChatAnthropic(model="claude-haiku-4-5"),
    }

    if model in model_map:
        return model_map[model]()

    # 允许任意模型名称：优先克隆 parent_llm，只替换 model
    if is_dataclass(parent_llm):
        try:
            return replace(parent_llm, model=model)
        except TypeError:
            pass

    # 兜底：使用 Anthropic（会从环境变量读取 API Key 等配置）
    return ChatAnthropic(model=model)
