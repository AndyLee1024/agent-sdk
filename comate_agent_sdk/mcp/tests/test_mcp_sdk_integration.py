import json

from comate_agent_sdk import Agent, create_sdk_mcp_server, mcp_tool
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.llm.messages import Function, ToolCall, ToolMessage
from comate_agent_sdk.llm.views import ChatInvokeCompletion


class _FakeChatModel:
    def __init__(self):
        self.model = "fake:model"
        self.calls = 0
        self.seen_tool_names: list[str] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.seen_tool_names = [t.name for t in (tools or [])]

        # 第一次：触发 MCP tool call
        if self.calls == 1:
            tc = ToolCall(
                id="tc1",
                function=Function(
                    name="mcp__calc__add",
                    arguments=json.dumps({"a": 2, "b": 3}),
                ),
            )
            return ChatInvokeCompletion(content=None, tool_calls=[tc])

        # 第二次：返回最终文本
        return ChatInvokeCompletion(content="done", tool_calls=[])


def test_sdk_mcp_tools_loaded_and_injected_into_system_prompt() -> None:
    @mcp_tool(name="add", description="Add two numbers")
    async def add(a: float, b: float) -> str:
        return f"Sum: {a + b}"

    calc = create_sdk_mcp_server(name="calculator", tools=[add])

    llm = _FakeChatModel()
    template = Agent(
        llm=llm,  # type: ignore[arg-type]
        config=AgentConfig(
            mcp_servers={"calc": calc},
            tools=("mcp__calc__add",),
            agents=(),
            offload_enabled=False,
            setting_sources=None,
        ),
    )
    agent = template.create_runtime()

    result = __import__("asyncio").run(agent.query("x"))
    assert result == "done"

    # LLM 在第一次调用时应已看见 MCP tool definition
    assert "mcp__calc__add" in llm.seen_tool_names

    # MCP tools 概览被注入到 system prompt（SystemMessage）
    system_msg = agent.messages[0]
    header = system_msg.text
    assert "<mcp_tools>" in header
    assert "mcp__calc__add" in header


def test_sdk_mcp_tool_call_executes_and_writes_tool_message() -> None:
    @mcp_tool(name="add", description="Add two numbers")
    async def add(a: float, b: float) -> str:
        return f"Sum: {a + b}"

    calc = create_sdk_mcp_server(name="calculator", tools=[add])

    template = Agent(
        llm=_FakeChatModel(),  # type: ignore[arg-type]
        config=AgentConfig(
            mcp_servers={"calc": calc},
            tools=("mcp__calc__add",),
            agents=(),
            offload_enabled=False,
            setting_sources=None,
        ),
    )
    agent = template.create_runtime()

    __import__("asyncio").run(agent.query("x"))

    tool_msgs = [m for m in agent.messages if isinstance(m, ToolMessage)]
    assert tool_msgs, "Expected at least one ToolMessage in conversation"
    assert tool_msgs[-1].tool_name == "mcp__calc__add"
    assert "Sum: 5" in str(tool_msgs[-1].content)
