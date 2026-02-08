from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from comate_agent_sdk.llm.messages import ToolCall, ToolMessage, UserMessage

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


def setup_tool_strategy(agent: "AgentRuntime") -> None:
    """设置工具策略提示（写入 ContextIR header）。"""
    from comate_agent_sdk.agent.tool_strategy import generate_tool_strategy

    tool_strategy = generate_tool_strategy(agent.tools)
    if tool_strategy:
        agent._context.set_tool_strategy(tool_strategy)


def setup_agent_loop(agent: "AgentRuntime") -> None:
    """设置 Agent 循环控制指令（写入 ContextIR header）。"""
    from comate_agent_sdk.agent.prompts import AGENT_LOOP_PROMPT

    agent._context.set_agent_loop(AGENT_LOOP_PROMPT, cache=False)


def setup_subagents(agent: "AgentRuntime") -> None:
    """设置 Subagent 支持

    1. 生成 Subagent 策略提示并写入 ContextIR header
    2. 创建 Task 工具并添加到 tools 列表
    """
    from comate_agent_sdk.subagent.prompts import generate_subagent_prompt
    from comate_agent_sdk.subagent.task_tool import create_task_tool

    if not agent.agents or agent.tool_registry is None:
        return

    task_tools = [t for t in agent.tools if t.name == "Task"]
    user_task_tools = [
        t
        for t in task_tools
        if getattr(t, "_comate_agent_sdk_internal", False) is not True
    ]
    if user_task_tools:
        raise ValueError(
            "启用 subagent 时检测到用户提供了同名工具 'Task'。"
            "'Task' 为 subagent 调度保留名，SDK 会注入系统 Task 工具，禁止静默替换。"
            "解决方式：1) 将你的工具改名（不要叫 'Task'）；"
            "2) 显式禁用 subagent（例如 agents=[]/()）；"
            "3) 或移除/调整 .agent/subagents 下的定义。"
        )

    # 生成 Subagent 策略提示并写入 ContextIR header
    subagent_prompt = generate_subagent_prompt(agent.agents)
    agent._context.set_subagent_strategy(subagent_prompt)

    # 避免重复添加 Task 工具
    agent.tools[:] = [
        t
        for t in agent.tools
        if t.name != "Task" or getattr(t, "_comate_agent_sdk_internal", False) is not True
    ]
    agent._tool_map.pop("Task", None)

    # 创建 Task 工具
    task_tool = create_task_tool(
        agents=agent.agents,
        parent_tools=agent.tools,
        tool_registry=agent.tool_registry,  # type: ignore[arg-type]
        parent_llm=agent.llm,
        parent_dependency_overrides=agent.dependency_overrides,  # type: ignore
        parent_llm_levels=agent.llm_levels,
        parent_token_cost=agent._token_cost,
    )

    # 添加到工具列表
    agent.tools.append(task_tool)
    agent._tool_map[task_tool.name] = task_tool

    logger.info(
        f"Initialized {len(agent.agents)} subagents: {[a.name for a in agent.agents]}"
    )

def setup_memory(agent: "AgentRuntime") -> None:
    """设置 Memory 静态背景知识。"""
    from comate_agent_sdk.context.memory import MemoryConfig, load_memory_content

    if not isinstance(agent.memory, MemoryConfig):
        logger.warning("memory 参数不是 MemoryConfig 实例，跳过初始化")
        return

    # 加载文件内容
    content = load_memory_content(agent.memory)
    if not content:
        logger.warning("未能加载任何 memory 内容")
        return

    # Token 检查
    token_count = agent._context.token_counter.count(content)
    if token_count > agent.memory.max_tokens:
        logger.warning(
            f"Memory 内容超出 token 上限: {token_count} > {agent.memory.max_tokens} "
            f"(不会截断，但可能影响上下文预算)"
        )

    # 写入 ContextIR
    agent._context.set_memory(content, cache=agent.memory.cache)
    logger.info(f"Memory 已加载: {token_count} tokens 来自 {len(agent.memory.files)} 个文件")


def setup_skills(agent: "AgentRuntime") -> None:
    """设置 Skill 支持（基于模板已解析结果进行 prompt 注入 + Skill tool 创建）。"""
    from comate_agent_sdk.skill import create_skill_tool
    from comate_agent_sdk.skill.prompts import generate_skill_prompt

    if not agent.skills:
        return

    # 生成 Skill 策略提示并写入 ContextIR header
    skill_prompt = generate_skill_prompt(agent.skills)
    if skill_prompt:  # 只有当有 active skills 时才注入
        agent._context.set_skill_strategy(skill_prompt)

    # 避免重复添加 Skill 工具（例如用户手动传入 tools 里已包含 Skill）
    agent.tools[:] = [t for t in agent.tools if t.name != "Skill"]
    agent._tool_map.pop("Skill", None)

    # 创建 Skill 工具（现在只包含简洁描述）
    skill_tool = create_skill_tool(agent.skills)
    agent.tools.append(skill_tool)
    agent._tool_map[skill_tool.name] = skill_tool

    logger.info(f"Initialized {len(agent.skills)} skill(s): {[s.name for s in agent.skills]}")


async def execute_skill_call(agent: "AgentRuntime", tool_call: ToolCall) -> ToolMessage:
    """执行 Skill 调用（特殊处理）。"""
    args = json.loads(tool_call.function.arguments)
    skill_name = args.get("skill_name")

    # 查找 Skill
    skill_def = next((s for s in agent.skills if s.name == skill_name), None) if agent.skills else None
    if not skill_def:
        return ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Skill",
            content=f"Error: Unknown skill '{skill_name}'",
            is_error=True,
        )

    # 准备待注入的消息（将在 ToolMessage 添加到 context 后注入）
    metadata = (
        f'<skill-message>The "{skill_name}" skill is loading</skill-message>\n'
        f"<skill-name>{skill_name}</skill-name>"
    )
    full_prompt = skill_def.get_prompt()

    # 通过 ContextIR 管理 pending skill items
    agent._context.add_skill_injection(
        skill_name=skill_name,
        metadata_msg=UserMessage(content=metadata, is_meta=False),
        prompt_msg=UserMessage(content=full_prompt, is_meta=True),
    )

    # 返回 ToolMessage（调用方会先添加这个，再 flush pending items）
    return ToolMessage(
        tool_call_id=tool_call.id,
        tool_name="Skill",
        content=f"Skill '{skill_name}' loaded successfully and will remain active",
        is_error=False,
    )


def rebuild_skill_tool(agent: "AgentRuntime") -> None:
    """重建 Skill 工具（用于 Subagent Skills 筛选后更新工具描述）。"""
    from comate_agent_sdk.skill import create_skill_tool, generate_skill_prompt

    # 1. 移除旧的 Skill 工具
    agent.tools[:] = [t for t in agent.tools if t.name != "Skill"]
    agent._tool_map.pop("Skill", None)

    # 2. 移除 IR 中的旧 Skill 策略
    agent._context.remove_skill_strategy()

    # 3. 如果还有 skills，重新生成
    if agent.skills:
        skill_prompt = generate_skill_prompt(agent.skills)
        if skill_prompt:
            # 写入 IR header
            agent._context.set_skill_strategy(skill_prompt)

        # 创建新的 Skill 工具
        skill_tool = create_skill_tool(agent.skills)
        agent.tools.append(skill_tool)
        agent._tool_map["Skill"] = skill_tool

        logger.debug(f"Rebuilt Skill tool with {len(agent.skills)} skill(s)")
    else:
        logger.debug("Removed Skill tool (no skills remaining)")


def remove_skill_strategy(agent: "AgentRuntime") -> None:
    """辅助函数：仅移除 skill_strategy（用于调试/外部调用）。"""
    agent._context.remove_skill_strategy()
