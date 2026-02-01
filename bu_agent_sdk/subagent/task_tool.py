"""
Task 工具 - 用于启动 Subagent 的统一入口
"""

import asyncio
import logging
from dataclasses import is_dataclass, replace

from bu_agent_sdk.llm.base import BaseChatModel
from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
from bu_agent_sdk.agent.llm_levels import LLMLevel
from bu_agent_sdk.subagent.models import AgentDefinition
from bu_agent_sdk.tokens import TokenCost
from bu_agent_sdk.tools.decorator import Tool, tool
from bu_agent_sdk.tools.depends import DependencyOverrides
from bu_agent_sdk.tools.registry import ToolRegistry

logger = logging.getLogger("bu_agent_sdk.subagent.task_tool")


def create_task_tool(
    agents: list[AgentDefinition],
    parent_tools: list[Tool],
    tool_registry: ToolRegistry,
    parent_llm: BaseChatModel,
    parent_dependency_overrides: DependencyOverrides | None = None,
    parent_llm_levels: dict[LLMLevel, BaseChatModel] | None = None,
    parent_token_cost: TokenCost | None = None,
) -> Tool:
    """创建 Task 工具，用于启动 Subagent

    Args:
        agents: AgentDefinition 列表
        parent_tools: 父 Agent 当前可见的工具列表（用于继承与动态工具解析）
        tool_registry: 全局工具注册表
        parent_llm: 父 Agent 的 LLM 实例
        parent_dependency_overrides: 父 Agent 的依赖覆盖（会继承给 Subagent）

    Returns:
        Task 工具实例
    """
    # 构建 subagent 名称映射
    agent_map = {a.name: a for a in agents}

    def _resolve_subagent_tools(agent_def: AgentDefinition) -> tuple[list[Tool], list[str]]:
        """解析 subagent 可用工具列表。

        规则：
        - agent_def.tools is None：继承父 Agent 工具（禁止 Task，避免嵌套 subagent）
        - agent_def.tools 是列表：按名称解析（优先从 parent_tools 解析动态工具；再从 registry 解析）
        """
        parent_map = {t.name: t for t in parent_tools}

        if agent_def.tools is None:
            tools = [t for t in parent_tools if t.name != "Task"]
            return tools, []

        resolved: list[Tool] = []
        missing: list[str] = []
        for name in agent_def.tools or []:
            if name == "Task":
                continue
            if name in parent_map:
                resolved.append(parent_map[name])
                continue
            if name in tool_registry:
                resolved.append(tool_registry.get(name))
                continue
            missing.append(name)

        return resolved, missing

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
        llm = resolve_model(
            model=agent_def.model,
            level=agent_def.level,
            parent_llm=parent_llm,
            llm_levels=parent_llm_levels,
        )

        # 解析工具
        tools, missing_tools = _resolve_subagent_tools(agent_def)

        if missing_tools:
            logger.warning(
                f"Subagent '{subagent_type}' requested missing tool(s): {missing_tools}"
            )

        logger.info(
            f"Launching subagent '{subagent_type}' with {len(tools)} tool(s): "
            f"{[t.name for t in tools]}"
        )

        # 创建 Subagent（继承父级依赖覆盖）
        subagent = Agent(
            name=agent_def.name,
            llm=llm,
            tools=tools,
            system_prompt=agent_def.prompt,
            max_iterations=agent_def.max_iterations,
            compaction=agent_def.compaction,
            dependency_overrides=parent_dependency_overrides,  # 继承父级
            llm_levels=parent_llm_levels,  # 继承三档池
            _parent_token_cost=parent_token_cost,  # 共享 TokenCost
            _is_subagent=True,  # 禁止嵌套
        )

        # Subagent Skills 筛选（如果 AgentDefinition.skills 不为空）
        if agent_def.skills is not None and subagent.skills:
            allowed_skill_names = set(agent_def.skills)
            subagent.skills = [s for s in subagent.skills if s.name in allowed_skill_names]
            # 重新创建 Skill 工具（更新工具描述）
            subagent._rebuild_skill_tool()
            logger.info(
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

            logger.info(f"Subagent '{subagent_type}' completed successfully")
            return result

        except asyncio.TimeoutError:
            error_msg = (
                f"Error: Subagent '{subagent_type}' timeout after {agent_def.timeout}s"
            )
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error in subagent '{subagent_type}': {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    return Task


def resolve_model(
    model: str | None,
    level: LLMLevel | None,
    parent_llm: BaseChatModel,
    llm_levels: dict[LLMLevel, BaseChatModel] | None = None,
) -> BaseChatModel:
    """解析模型配置

    Args:
        model: 模型名称（None/"inherit" 代表继承；也支持任意模型字符串）
        level: 档位（LOW/MID/HIGH）。当 model 未指定时优先生效。
        parent_llm: 父 Agent 的 LLM 实例
        llm_levels: 父 Agent 的三档模型池（用于 level/alias 映射）

    Returns:
        解析后的 LLM 实例

    Notes:
        - "sonnet"/"opus"/"haiku" 会使用内置 Anthropic 预设
        - 其他字符串会尽量克隆 parent_llm，仅替换 model 字段
    """
    # 1) model= 非 None 且非 "inherit"
    if model and model.strip().lower() != "inherit":
        model = model.strip()

        alias: dict[str, LLMLevel] = {"sonnet": "MID", "opus": "HIGH", "haiku": "LOW"}
        pool_key = alias.get(model.lower())
        if pool_key and llm_levels and pool_key in llm_levels:
            return llm_levels[pool_key]

        hard = {
            "sonnet": lambda: ChatAnthropic(model="claude-sonnet-4-5"),
            "opus": lambda: ChatAnthropic(model="claude-opus-4-5"),
            "haiku": lambda: ChatAnthropic(model="claude-haiku-4-5"),
        }
        if model.lower() in hard:
            return hard[model.lower()]()

        if is_dataclass(parent_llm):
            try:
                return replace(parent_llm, model=model)
            except TypeError:
                pass

        return ChatAnthropic(model=model)

    # 2) level= 从池取
    if level is not None:
        if llm_levels and level in llm_levels:
            return llm_levels[level]
        raise ValueError(f"level='{level}' 不在 llm_levels 中")

    # 3) 默认继承
    return parent_llm
