"""
Subagent 使用示例

演示如何使用 Subagent 功能来构建具有专业化子 Agent 的系统。

包含三种使用方式：
1. main() - 手动定义 Subagent（完全控制）
2. main_with_auto_discovery() - 自动发现 Subagent（从文件加载）
3. main_with_hybrid_mode() - 混合模式（自动发现 + 代码传入）

自动发现机制：
- Agent 初始化时自动从 ~/.agent/subagents/*.md 和 .agent/subagents/*.md 加载
- 优先级：project 存在时完全忽略 user
- Merge：代码传入的 subagent 会覆盖同名的自动发现的
"""

import asyncio
import logging
import os
from pathlib import Path

from comate_agent_sdk import Agent, AgentDefinition
from comate_agent_sdk.agent import ComateAgentOptions
from comate_agent_sdk.llm import ChatOpenAI
from comate_agent_sdk.tools import ToolRegistry, tool 

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    # 1. 创建工具注册表
    registry = ToolRegistry()

    # 2. 注册工具
    @tool("搜索网页获取信息", registry=registry)
    async def search(query: str) -> str:
        """
        Args:
            query: 搜索查询词
        """
        logger.info(f"搜索: {query}")
        # 模拟搜索结果
        return f"关于 '{query}' 的搜索结果：这是一个模拟的搜索结果。"

    @tool("读取文件内容", registry=registry)
    async def read_file(path: str) -> str:
        """
        Args:
            path: 文件路径
        """
        logger.info(f"读取文件: {path}")
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"错误: 无法读取文件 {path}: {e}"

    @tool("写入文件内容", registry=registry)
    async def write_file(path: str, content: str) -> str:
        """
        Args:
            path: 文件路径
            content: 文件内容
        """
        logger.info(f"写入文件: {path}")
        try:
            Path(path).write_text(content, encoding="utf-8")
            return f"成功写入文件: {path}"
        except Exception as e:
            return f"错误: 无法写入文件 {path}: {e}"

    os.environ["OPENAI_BASE_URL"] = "http://192.168.25.119:4141/v1"

    # 3. 定义 Subagent
    researcher = AgentDefinition(
        name="researcher",
        description="搜索和收集信息的专家。当需要搜索资料或阅读文件时使用。",
        prompt="""你是一个研究员，擅长搜索和整理信息。

当收到任务时：
1. 使用 search 工具搜索相关信息
2. 使用 read_file 工具阅读相关文件
3. 整理并总结你的发现
4. 返回清晰、有组织的研究结果""",
        tools=["search", "read_file"],
        level="LOW",  # 使用低档位模型（快速、便宜）
        timeout=60,  # 60秒超时
    )

    writer = AgentDefinition(
        name="writer",
        description="撰写内容的专家。当需要创作或编辑文档时使用。",
        prompt="""你是一个作家，擅长撰写清晰、有条理的文档。

当收到任务时：
1. 理解要写的内容主题
2. 组织内容结构
3. 使用 write_file 工具将内容写入文件
4. 确保内容质量高、易于理解""",
        tools=["write_file"],
        level="LOW",  # 使用低档位模型（快速、便宜）
    )

    # 4. 创建主 Agent
    logger.info("创建主 Agent...")
    agent = Agent(
        llm=ChatOpenAI(model="gpt-5-mini", api_key="sk-KXdioVlee9i7DpUNBZtKM4FCXDqUCONdJGgpeTQfGFmJrRwD",base_url="http://454443.xyz:3022/v1"),
        options=ComateAgentOptions(
            agents=[researcher, writer],
            system_prompt="你是一个项目经理，可以协调研究员和作家来完成任务。",
        ),
    )

    # 5. 测试场景 1: 串行调用（先研究后写作）
    logger.info("\n===== 场景 1: 串行调用 =====")
    result1 = await agent.query(
        "先研究一下 Python 异步编程的概念，然后根据研究结果写一篇简短的介绍文章，保存到 /tmp/async_intro.txt"
    )
    logger.info(f"结果: {result1}")

    # 6. 测试场景 2: 并行调用（同时研究多个主题）
    logger.info("\n===== 场景 2: 并行调用 =====")
    result2 = await agent.query("同时研究以下两个主题：1) Python GIL 2) asyncio 事件循环")
    logger.info(f"结果: {result2}")

    # 7. 查看 token 使用情况
    usage = await agent.get_usage()
    logger.info(f"\n===== Token 使用情况 =====")
    logger.info(f"总 tokens: {usage.total_tokens}")
    logger.info(f"总成本: ${usage.total_cost:.4f}")


async def main_with_auto_discovery():
    """使用自动发现 Subagent 的示例（新方式）

    Agent 会自动从以下位置发现 subagent：
    - ~/.agent/subagents/*.md  (user 级别)
    - .agent/subagents/*.md    (project 级别)

    优先级规则：
    - 如果 project 存在，完全忽略 user
    - 否则使用 user
    """
    logger.info("\n===== 自动发现示例 =====")

    # 不需要显式调用 discover_subagents！
    # Agent 初始化时会自动发现
    agent = Agent(
        llm=ChatOpenAI(model="gpt-5-mini", api_key="sk-test"),
    )

    if agent.agents:
        logger.info(f"✓ 自动发现 {len(agent.agents)} 个 Subagent:")
        for a in agent.agents:
            logger.info(f"  - {a.name}: {a.description}")
    else:
        logger.info("✗ 未发现任何 Subagent")
        logger.info("提示: 在以下位置创建 .md 文件定义 subagent:")
        logger.info("  - ~/.agent/subagents/  (user 级别)")
        logger.info("  - .agent/subagents/   (project 级别)")
        return

    # 使用 Agent
    result = await agent.query("请帮我完成一个任务")
    logger.info(f"结果: {result}")


async def main_with_hybrid_mode():
    """混合模式：自动发现 + 代码定义

    同名时，代码传入的会覆盖自动发现的。
    """
    logger.info("\n===== 混合模式示例 =====")

    # 定义一个额外的 subagent
    custom_agent = AgentDefinition(
        name="analyzer",
        description="代码分析专家",
        prompt="你是一个代码分析专家，擅长分析代码质量和提出改进建议。",
        level="MID",  # 使用中档位模型
    )

    # Agent 会自动发现 + 代码传入的 subagent
    agent = Agent(
        llm=ChatOpenAI(model="gpt-5-mini", api_key="sk-test"),
        options=ComateAgentOptions(
            agents=[custom_agent],  # 传入额外的 subagent
        ),
    )

    logger.info(f"总共加载 {len(agent.agents) if agent.agents else 0} 个 Subagent:")
    if agent.agents:
        for a in agent.agents:
            logger.info(f"  - {a.name}: {a.description}")

    # 如果自动发现的 subagent 中有同名的 "analyzer"，会被代码传入的覆盖
    # 如果自动发现了 "researcher", "writer" 等，它们会和 "analyzer" 一起存在


if __name__ == "__main__":
    # 方式 1: 手动定义 Subagent（完全控制）
    asyncio.run(main())

    # 方式 2: 完全自动发现（从文件加载）
    # asyncio.run(main_with_auto_discovery())

    # 方式 3: 混合模式（自动发现 + 代码传入）
    # asyncio.run(main_with_hybrid_mode())
