import json
import unittest

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.tool_exec import execute_tool_call
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.tools import tool


class _FakeChatModel:
    def __init__(self) -> None:
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        return ChatInvokeCompletion(content="ok")


class TestHooksToolExecIntegrationBlocks(unittest.IsolatedAsyncioTestCase):
    async def test_pretooluse_deny_blocks_execution(self) -> None:
        calls = {"count": 0}
        hook_calls = {"post_fail": 0}

        @tool("Read tool", name="Read")
        async def Read(file_path: str) -> str:
            calls["count"] += 1
            return f"content:{file_path}"

        template = Agent(
            llm=_FakeChatModel(),  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(Read,),
                agents=(),
                offload_enabled=False,
            ),
        )
        runtime = template.create_runtime()
        runtime.register_python_hook(
            event_name="PreToolUse",
            matcher="^Read$",
            callback=lambda _: {"permissionDecision": "deny", "reason": "blocked-by-hook"},
        )
        runtime.register_python_hook(
            event_name="PostToolUseFailure",
            matcher="^Read$",
            callback=lambda _: hook_calls.__setitem__("post_fail", hook_calls["post_fail"] + 1),
        )

        tool_call = ToolCall(
            id="tc_read",
            function=Function(name="Read", arguments=json.dumps({"file_path": "a.txt"}, ensure_ascii=False)),
        )

        message = await execute_tool_call(runtime, tool_call)
        self.assertTrue(message.is_error)
        self.assertIn("blocked-by-hook", message.text)
        self.assertEqual(calls["count"], 0)
        self.assertEqual(hook_calls["post_fail"], 1)

    async def test_pretooluse_ask_uses_approval_callback(self) -> None:
        calls = {"count": 0}

        @tool("Write tool", name="Write")
        async def Write(path: str) -> str:
            calls["count"] += 1
            return f"ok:{path}"

        template = Agent(
            llm=_FakeChatModel(),  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(Write,),
                agents=(),
                offload_enabled=False,
            ),
        )
        runtime = template.create_runtime()
        runtime.register_python_hook(
            event_name="PreToolUse",
            matcher="^Write$",
            callback=lambda _: {"permissionDecision": "ask", "reason": "need-approval"},
        )
        runtime.options.tool_approval_callback = (
            lambda payload: {"decision": "deny", "reason": "user-rejected"}
        )

        tool_call = ToolCall(
            id="tc_write",
            function=Function(name="Write", arguments=json.dumps({"path": "x.txt"}, ensure_ascii=False)),
        )

        message = await execute_tool_call(runtime, tool_call)
        self.assertTrue(message.is_error)
        self.assertIn("user-rejected", message.text)
        self.assertEqual(calls["count"], 0)

    async def test_updated_input_is_recoerced_after_hook_modification(self) -> None:
        @tool("Tag joiner", name="JoinTags")
        async def JoinTags(tags: list[str]) -> str:
            return ",".join(tags)

        template = Agent(
            llm=_FakeChatModel(),  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(JoinTags,),
                agents=(),
                offload_enabled=False,
            ),
        )
        runtime = template.create_runtime()
        runtime.register_python_hook(
            event_name="PreToolUse",
            matcher="^JoinTags$",
            callback=lambda _: {"updatedInput": {"tags": '["x","y"]'}},
        )

        tool_call = ToolCall(
            id="tc_tags",
            function=Function(name="JoinTags", arguments=json.dumps({"tags": ["a"]}, ensure_ascii=False)),
        )

        message = await execute_tool_call(runtime, tool_call)
        self.assertFalse(message.is_error)
        self.assertEqual(message.text, "x,y")

    async def test_unknown_tool_still_emits_post_failure_hook(self) -> None:
        template = Agent(
            llm=_FakeChatModel(),  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(),
                agents=(),
                offload_enabled=False,
            ),
        )
        runtime = template.create_runtime()
        called = {"count": 0}
        runtime.register_python_hook(
            event_name="PostToolUseFailure",
            matcher="^MissingTool$",
            callback=lambda _: called.__setitem__("count", called["count"] + 1),
        )

        tool_call = ToolCall(
            id="tc_missing",
            function=Function(name="MissingTool", arguments=json.dumps({"x": 1}, ensure_ascii=False)),
        )

        message = await execute_tool_call(runtime, tool_call)
        self.assertTrue(message.is_error)
        self.assertIn("Unknown tool", message.text)
        self.assertEqual(called["count"], 1)
