"""测试 Skill 与 Subagent 的集成功能"""

import asyncio
from pathlib import Path

from bu_agent_sdk import Agent, AgentDefinition
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool


@tool("读取文件")
async def read_file(path: str) -> str:
    """读取文件

    Args:
        path: 文件路径
    """
    try:
        return Path(path).read_text(encoding="utf-8")[:100]
    except Exception as e:
        return f"Error: {e}"


@tool("写入文件")
async def write_file(path: str, content: str) -> str:
    """写入文件

    Args:
        path: 文件路径
        content: 内容
    """
    try:
        Path(path).write_text(content, encoding="utf-8")
        return f"写入成功: {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("搜索网络")
async def search_web(query: str) -> str:
    """搜索网络

    Args:
        query: 查询
    """
    return f"搜索结果 for: {query}"


async def test_agentdef_skills_field():
    """测试 AgentDefinition 的 skills 字段"""
    print("\n=== 测试 1: AgentDefinition.skills 字段 ===")

    # 创建 Subagent 定义（带 skills 限制）
    subagent_dir = Path(".agent/subagents")
    subagent_dir.mkdir(parents=True, exist_ok=True)

    subagent_content = """---
name: file-ops
description: 文件操作专家
tools: read_file, write_file
skills: file-skill
---

你是文件操作专家，使用工具帮助用户。
"""

    (subagent_dir / "file_ops.md").write_text(subagent_content, encoding="utf-8")

    # 从文件加载
    agent_def = AgentDefinition.from_file(subagent_dir / "file_ops.md")

    print(f"✅ AgentDefinition 解析成功:")
    print(f"   - name: {agent_def.name}")
    print(f"   - tools: {agent_def.tools}")
    print(f"   - skills: {agent_def.skills}")

    assert agent_def.skills == ["file-skill"], "Should parse skills field"


async def test_subagent_skills_filtering():
    """测试 Subagent Skills 筛选功能"""
    print("\n=== 测试 2: Subagent Skills 筛选 ===")

    # 创建两个 Skills
    skills_dir = Path(".agent/skills")
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Skill 1: file-skill
    file_skill_dir = skills_dir / "file-skill"
    file_skill_dir.mkdir(exist_ok=True)
    (file_skill_dir / "SKILL.md").write_text(
        """---
name: file-skill
description: 文件操作专家
allowed-tools: read_file, write_file
---

你是文件操作专家。
""",
        encoding="utf-8"
    )

    # Skill 2: web-skill
    web_skill_dir = skills_dir / "web-skill"
    web_skill_dir.mkdir(exist_ok=True)
    (web_skill_dir / "SKILL.md").write_text(
        """---
name: web-skill
description: 网络搜索专家
allowed-tools: search_web
---

你是网络搜索专家。
""",
        encoding="utf-8"
    )

    # 创建 Subagent 定义（限制只能用 file-skill）
    subagent_def = AgentDefinition(
        name="limited-agent",
        description="受限的 Agent",
        prompt="你是受限的 Agent",
        tools=["read_file", "write_file", "search_web"],
        skills=["file-skill"],  # 只允许 file-skill
    )

    # 创建主 Agent
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[read_file, write_file, search_web],
        agents=[subagent_def],
    )

    print(f"✅ 主 Agent Skills: {[s.name for s in agent.skills] if agent.skills else []}")

    # 模拟创建 Subagent（通过 Task 工具的内部逻辑）
    # 注意：这里我们直接创建 Subagent 来测试筛选逻辑
    from bu_agent_sdk.subagent.task_tool import resolve_model

    subagent = Agent(
        llm=resolve_model(subagent_def.model, agent.llm),
        tools=agent.tool_registry.filter(subagent_def.tools or []),
        system_prompt=subagent_def.prompt,
        max_iterations=subagent_def.max_iterations,
        _is_subagent=True,
    )

    print(f"✅ Subagent Skills (筛选前): {[s.name for s in subagent.skills] if subagent.skills else []}")

    # 应用筛选
    if subagent_def.skills is not None and subagent.skills:
        allowed_skill_names = set(subagent_def.skills)
        subagent.skills = [s for s in subagent.skills if s.name in allowed_skill_names]
        subagent._rebuild_skill_tool()

    print(f"✅ Subagent Skills (筛选后): {[s.name for s in subagent.skills] if subagent.skills else []}")

    # 验证只有 file-skill
    assert subagent.skills is not None, "Subagent should have skills"
    assert len(subagent.skills) == 1, "Should have only 1 skill after filtering"
    assert subagent.skills[0].name == "file-skill", "Should only have file-skill"

    print("✅ Subagent Skills 筛选成功")


async def test_skill_context_modification():
    """测试 Skill 的 execution context 修改"""
    print("\n=== 测试 3: Skill Execution Context 修改 ===")

    from bu_agent_sdk.skill import SkillDefinition, apply_skill_context

    # 创建 Agent
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[read_file, write_file, search_web],
    )

    print(f"✅ 初始工具数量: {len(agent.tools)}")
    initial_tool_count = len(agent.tools)

    # 创建限制工具的 Skill
    skill = SkillDefinition(
        name="limited-skill",
        description="Limited skill",
        prompt="Limited prompt",
        allowed_tools=["read_file", "write_file"],
    )

    # 应用 Skill context
    apply_skill_context(agent, skill)

    print(f"✅ 应用 Skill 后工具数量: {len(agent.tools)}")
    print(f"✅ 可用工具: {[t.name for t in agent.tools]}")

    # 验证工具被限制
    assert len(agent.tools) < initial_tool_count, "Tools should be restricted"
    tool_names = {t.name for t in agent.tools}
    assert "read_file" in tool_names, "read_file should be available"
    assert "write_file" in tool_names, "write_file should be available"
    assert "search_web" not in tool_names, "search_web should be removed"

    print("✅ Skill 工具权限限制成功")


async def test_skill_duplicate_protection():
    """测试 Skill 重复调用保护"""
    print("\n=== 测试 4: Skill 重复调用保护 ===")

    from bu_agent_sdk.skill import SkillDefinition

    # 创建两个 Skills
    skill1 = SkillDefinition(
        name="skill-1",
        description="First skill",
        prompt="First prompt",
    )

    skill2 = SkillDefinition(
        name="skill-2",
        description="Second skill",
        prompt="Second prompt",
    )

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[read_file],
        skills=[skill1, skill2],
    )

    # 模拟调用第一个 Skill
    from bu_agent_sdk.llm.messages import ToolCall, Function

    tool_call_1 = ToolCall(
        id="call_1",
        function=Function(name="Skill", arguments='{"skill_name": "skill-1"}'),
    )

    result_1 = await agent._execute_skill_call(tool_call_1)
    print(f"✅ 第一次调用结果: {result_1.content[:50]}...")

    # 验证 Skill 已激活
    assert agent._active_skill_name == "skill-1", "Skill 1 should be active"

    # 尝试调用第二个 Skill（应该失败）
    tool_call_2 = ToolCall(
        id="call_2",
        function=Function(name="Skill", arguments='{"skill_name": "skill-2"}'),
    )

    result_2 = await agent._execute_skill_call(tool_call_2)
    print(f"✅ 第二次调用结果: {result_2.content[:50]}...")

    # 验证第二次调用被拒绝
    assert result_2.is_error, "Second skill call should be rejected"
    assert "already active" in result_2.content.lower(), "Should mention already active"

    print("✅ Skill 重复调用保护成功")


async def main():
    """运行所有测试"""
    try:
        await test_agentdef_skills_field()
        await test_subagent_skills_filtering()
        await test_skill_context_modification()
        await test_skill_duplicate_protection()

        print("\n" + "=" * 50)
        print("✅ 所有集成测试通过！")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
