import asyncio
import logging
import os
from typing import Literal

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.agent.llm_levels import resolve_llm_levels
from comate_agent_sdk.llm import ChatAnthropic, ChatGoogle, ChatOpenAI
from comate_agent_sdk.system_tools.registry import get_system_tools
from comate_agent_sdk.tools import ToolRegistry

logger = logging.getLogger("comate_agent_sdk.examples.system_tools_smoke_agent")


def _load_main_llm_from_env() -> object:
    raw = (os.getenv("COMATE_AGENT_SDK_MAIN_LLM") or "").strip()
    if not raw:
        raise RuntimeError(
            "缺少环境变量 COMATE_AGENT_SDK_MAIN_LLM，示例格式：openai:gpt-5.2"
        )
    if ":" not in raw:
        raise RuntimeError(
            f"COMATE_AGENT_SDK_MAIN_LLM 必须为 provider:model 格式，例如 openai:gpt-5.2，实际为：{raw}"
        )
    provider, model = raw.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider == "openai":
        return ChatOpenAI(model=model)
    if provider == "anthropic":
        return ChatAnthropic(model=model)
    if provider == "google":
        return ChatGoogle(model=model)
    raise RuntimeError(f"不支持的 provider：{provider}（COMATE_AGENT_SDK_MAIN_LLM）")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    main_llm = _load_main_llm_from_env()
    llm_levels = resolve_llm_levels(explicit=None)  # env-only

    registry = ToolRegistry()
    for t in get_system_tools():
        registry.register(t)

    template = Agent(
        llm=main_llm,  # type: ignore[arg-type]
        config=AgentConfig(
            tools=registry.all(),
            llm_levels=llm_levels,  # type: ignore[arg-type]
        ),
    )
    agent = template.create_runtime()

    prompt = (
        "请按顺序完成：\n"
        "1) 使用 WebFetch 抓取 https://example.com 并回答：这个网页主要讲什么？\n"
        "2) 使用 MultiEdit 在 /tmp/comate_agent_sdk_smoke/agent.txt 创建文件（第一条 old_string 为空），写入一行 'hello'，然后把 hello 改成 hi。\n"
        "3) 使用 LS 列出 /tmp/comate_agent_sdk_smoke\n"
        "4) 使用 TodoWrite 写入 2 条 todo。\n"
        "5) 最后总结你做了什么。\n"
        "注意：务必使用工具完成，不要直接编造结果。"
    )

    try:
        result = await agent.query(prompt)
        logger.info(f"Final -> {result}")
    finally:
        summary = await agent.get_usage()
        logger.info(f"Usage total_tokens={summary.total_tokens} entries={summary.entry_count}")
        if summary.by_model:
            for model, stats in summary.by_model.items():
                logger.info(f"Model usage: {model} tokens={stats.total_tokens} invocations={stats.invocations}")


if __name__ == "__main__":
    asyncio.run(main())
