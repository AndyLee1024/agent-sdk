"""Skill 功能使用示例"""

import asyncio
from pathlib import Path

from bu_agent_sdk import Agent, SkillDefinition
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool


# ============================================
# 示例 1: 自动发现 Skills
# ============================================


async def example_auto_discovery():
    """自动发现并使用 Skills"""
    print("\n" + "=" * 60)
    print("示例 1: 自动发现 Skills")
    print("=" * 60)

    # 1. 创建一个 Skill 文件
    skill_dir = Path(".agent/skills/writer")
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_content = """---
name: writer
description: 专业的文档写作专家，擅长创建结构化的技术文档
allowed-tools: write_file, read_file
model: inherit
---

你是一个专业的技术文档写作专家。

## 你的职责
1. 帮助用户创建高质量的技术文档
2. 确保文档结构清晰、内容准确
3. 使用恰当的 Markdown 格式

## 可用工具
- `write_file`: 写入文件
- `read_file`: 读取文件

## 工作流程
1. 理解用户需求
2. 规划文档结构
3. 创建文档内容
4. 保存到文件
"""

    (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")

    # 2. 定义工具
    @tool("写入文件")
    async def write_file(path: str, content: str) -> str:
        try:
            Path(path).write_text(content, encoding="utf-8")
            return f"✅ 成功写入: {path}"
        except Exception as e:
            return f"❌ 错误: {e}"

    @tool("读取文件")
    async def read_file(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"❌ 错误: {e}"

    # 3. 创建 Agent（会自动发现 Skills）
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[write_file, read_file],
    )

    # 4. 查看自动发现的 Skills
    if agent.skills:
        print(f"\n📚 自动发现的 Skills:")
        for skill in agent.skills:
            print(f"   - {skill.name}: {skill.description}")

        # 5. 查看 Skill 工具
        skill_tool = agent._tool_map.get("Skill")
        if skill_tool:
            print(f"\n🛠️  Skill 工具已创建")
            print(f"   描述: {skill_tool.description[:100]}...")


# ============================================
# 示例 2: 手动定义 Skills
# ============================================


async def example_manual_skills():
    """手动定义并使用 Skills"""
    print("\n" + "=" * 60)
    print("示例 2: 手动定义 Skills")
    print("=" * 60)

    # 1. 定义工具
    @tool("计算加法")
    async def add(a: int, b: int) -> str:
        return f"{a} + {b} = {a + b}"

    @tool("计算乘法")
    async def multiply(a: int, b: int) -> str:
        return f"{a} × {b} = {a * b}"

    @tool("搜索网络")
    async def search(query: str) -> str:
        return f"搜索结果: {query}"

    # 2. 手动创建 SkillDefinition
    math_skill = SkillDefinition(
        name="math-expert",
        description="数学计算专家，只能使用数学工具",
        prompt="""你是一个数学计算专家。

你只能使用以下工具：
- add: 加法运算
- multiply: 乘法运算

你不能使用搜索工具。专注于数学计算任务。
""",
        allowed_tools=["add", "multiply"],  # 限制工具权限
        model="inherit",
    )

    # 3. 创建 Agent
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[add, multiply, search],
        skills=[math_skill],  # 手动传入 Skills
    )

    # 4. 查看 Skills
    print(f"\n📚 加载的 Skills:")
    for skill in agent.skills:
        print(f"   - {skill.name}: {skill.description}")
        if skill.allowed_tools:
            print(f"     允许的工具: {', '.join(skill.allowed_tools)}")


# ============================================
# 示例 3: Skill 资源打包
# ============================================


async def example_skill_with_resources():
    """使用带资源的 Skill"""
    print("\n" + "=" * 60)
    print("示例 3: Skill 资源打包")
    print("=" * 60)

    # 1. 创建 Skill 目录结构
    skill_dir = Path(".agent/skills/project-creator")
    skill_dir.mkdir(parents=True, exist_ok=True)

    # 2. 创建资源目录
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    references_dir = skill_dir / "references"
    references_dir.mkdir(exist_ok=True)

    # 3. 创建资源文件
    (scripts_dir / "init.sh").write_text("#!/bin/bash\necho 'Initializing project...'\n")
    (references_dir / "template.md").write_text("# Project Template\n\n...")

    # 4. 创建 SKILL.md
    skill_content = """---
name: project-creator
description: 项目创建专家，使用模板快速搭建项目
---

你是项目创建专家。

## 资源目录
- Scripts: {baseDir}/scripts/
- References: {baseDir}/references/

## 工作流程
1. 查看 references 中的模板
2. 使用 scripts 初始化项目
3. 创建项目结构
"""

    (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")

    # 5. 加载 Skill
    from bu_agent_sdk.skill import SkillDefinition

    skill = SkillDefinition.from_directory(skill_dir)

    # 6. 查看资源
    print(f"\n📦 Skill: {skill.name}")
    print(f"   描述: {skill.description}")
    print(f"   Base目录: {skill.base_dir}")
    if skill.scripts_dir:
        print(f"   Scripts: {skill.scripts_dir}")
    if skill.references_dir:
        print(f"   References: {skill.references_dir}")

    # 7. 获取替换后的 prompt
    prompt = skill.get_prompt()
    print(f"\n📝 Prompt (已替换 {{baseDir}}):")
    print(prompt[:200] + "...")


# ============================================
# 示例 4: Subagent + Skills
# ============================================


async def example_subagent_with_skills():
    """Subagent 使用 Skills"""
    print("\n" + "=" * 60)
    print("示例 4: Subagent 使用 Skills")
    print("=" * 60)

    # 1. 创建两个 Skills
    skills_dir = Path(".agent/skills")
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Skill A
    (skills_dir / "skill-a").mkdir(exist_ok=True)
    (skills_dir / "skill-a" / "SKILL.md").write_text(
        """---
name: skill-a
description: Skill A
---
Skill A prompt
""",
        encoding="utf-8"
    )

    # Skill B
    (skills_dir / "skill-b").mkdir(exist_ok=True)
    (skills_dir / "skill-b" / "SKILL.md").write_text(
        """---
name: skill-b
description: Skill B
---
Skill B prompt
""",
        encoding="utf-8"
    )

    # 2. 创建 Subagent 定义（限制只能用 skill-a）
    from bu_agent_sdk import AgentDefinition

    subagent_def = AgentDefinition(
        name="limited-agent",
        description="只能使用 Skill A 的受限 Agent",
        prompt="你是受限的 Agent",
        tools=["read_file"],
        skills=["skill-a"],  # 🔑 关键：限制可用的 Skills
    )

    @tool("读取文件")
    async def read_file(path: str) -> str:
        return f"File content: {path}"

    # 3. 创建主 Agent
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[read_file],
        agents=[subagent_def],
    )

    # 4. 查看配置
    print(f"\n📚 主 Agent Skills: {[s.name for s in agent.skills] if agent.skills else []}")
    print(f"\n🤖 Subagent 定义:")
    print(f"   - name: {subagent_def.name}")
    print(f"   - 允许的 Skills: {subagent_def.skills}")
    print(f"\n💡 当创建 Subagent 时，会自动筛选 Skills，只保留 'skill-a'")


# ============================================
# 主函数
# ============================================


async def main():
    """运行所有示例"""
    await example_auto_discovery()
    await example_manual_skills()
    await example_skill_with_resources()
    await example_subagent_with_skills()

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
