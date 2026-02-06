"""
Simple agentic loop implementation with native tool calling.

Usage:
    from comate_agent_sdk.llm import ChatOpenAI
    from comate_agent_sdk.tools import tool
    from comate_agent_sdk import Agent
    from comate_agent_sdk.agent import ComateAgentOptions

    @tool("Search the web")
    async def search(query: str) -> str:
        return f"Results for {query}"

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        options=ComateAgentOptions(tools=[search]),
    )

    response = await agent.query("Find information about Python")
    follow_up = await agent.query("Tell me more about that")

    # Compaction is enabled by default with dynamic thresholds based on model limits
    from comate_agent_sdk.agent.compaction import CompactionConfig

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        options=ComateAgentOptions(
            tools=[search],
            # Custom threshold ratio (default is 0.80 = 80% of model's context window)
            compaction=CompactionConfig(threshold_ratio=0.70),
            # Or disable compaction entirely:
            # compaction=CompactionConfig(enabled=False),
        ),
    )

    # Access usage statistics:
    summary = await agent.get_usage()
    print(f"Total tokens: {summary.total_tokens}")
    print(f"Total cost: ${summary.total_cost:.4f}")
"""

from comate_agent_sdk.agent.core import Agent as _Agent
from comate_agent_sdk.agent.system_prompt import SystemPromptConfig as _SystemPromptConfig
from comate_agent_sdk.agent.system_prompt import SystemPromptType

Agent = _Agent
SystemPromptConfig = _SystemPromptConfig

# 兼容性：保持 pickle / 反射里显示的 module 路径不变（历史上定义在 service.py）
Agent.__module__ = __name__
SystemPromptConfig.__module__ = __name__

__all__ = ["Agent", "SystemPromptConfig", "SystemPromptType"]
