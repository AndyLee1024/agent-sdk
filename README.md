# bu-agent-sdk

一个用于构建 Agent（工具调用 + for-loop）的 Python SDK。它强调“行动空间完整 + 显式退出 + 上下文工程”，同时尽量保持 API 简洁、可维护、可扩展。

![Agent Loop](./static/agent-loop.png)

## 你能用它做什么

- 把任意 `async def` 函数变成可被 LLM 调用的工具：签名 → JSON Schema（支持 Pydantic 参数模型）
- FastAPI 风格依赖注入：`Depends(...)`（适合注入 DB/客户端/配置/上下文）
- 上下文工程：自动压缩（compaction）、工具输出的 ephemeral 保留、超大内容 offload 到文件系统
- 会话能力：`ChatSession` 持久化 / 恢复 / 分叉（resume/fork）
- 可扩展能力包：
  - Subagent：通过 `.agent/subagents/*.md` 声明，自动注入 `Task` 工具
  - Skill：通过 `.agent/skills/*/SKILL.md` 声明，自动注入 `Skill` 工具
- Token 统计与可选计费（本地缓存定价数据）

## 安装

### 作为依赖接入（推荐）

```bash
uv add bu-agent-sdk
```

### 在本仓库开发

```bash
uv sync
uv run pytest -q
```

## 准备环境（以 OpenAI 为例）

SDK 会在 import 时自动加载 `.env`（依赖 `python-dotenv`），所以你可以在项目根目录放一个 `.env`：

```bash
OPENAI_API_KEY=...
```

如果你要使用内置 `WebFetch`（系统工具），它会调用 `llm_levels["LOW"]`，推荐你显式配置三档模型，避免默认落到 Anthropic：

```bash
BU_AGENT_SDK_LLM_LOW="openai:gpt-4o-mini"
BU_AGENT_SDK_LLM_MID="openai:gpt-4o"
BU_AGENT_SDK_LLM_HIGH="openai:gpt-4o"
BU_AGENT_SDK_LLM_LOW_BASE_URL="http://192.168.100.1:4141/v1"
BU_AGENT_SDK_LLM_MID_BASE_URL="http://192.168.100.1:4142/v1"
BU_AGENT_SDK_LLM_HIGH_BASE_URL="http://192.168.100.1:4143/v1"
```

## 快速上手：Claude Code 风格（系统工具 + 显式 done + Session）

下面这个最小示例具备：

- Claude Code 风格系统工具（`Bash/Read/Write/Edit/Grep/Glob/LS/TodoWrite/WebFetch`）
- 显式 `done` 工具（防止“无工具调用就提前结束”）
- `ChatSession`（会话持久化到 `~/.agent/sessions/<session_id>/`）

```python
import asyncio
import logging

from bu_agent_sdk import Agent, TaskComplete, tool
from bu_agent_sdk.agent import FinalResponseEvent, SessionInitEvent, ToolCallEvent, ToolResultEvent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import get_default_registry

logging.basicConfig(level=logging.INFO)


@tool("结束任务（必须调用）")
async def done(message: str) -> str:
    raise TaskComplete(message)


async def main() -> None:
    llm_levels = {
        "LOW": ChatOpenAI(model="gpt-4o-mini"),
        "MID": ChatOpenAI(model="gpt-4o"),
        "HIGH": ChatOpenAI(model="gpt-4o"),
    }

    agent = Agent(
        llm=llm_levels["MID"],
        llm_levels=llm_levels,
        tools=get_default_registry().all() + [done],
        require_done_tool=True,
        include_cost=False,
    )

    session = agent.chat()

    prompt = (
        "请在当前项目根目录里：\n"
        "1) 找到 pyproject.toml\n"
        "2) 读取并告诉我 [project] 的 name\n"
        "3) 完成后调用 done 给出结论\n"
    )

    async for event in session.query_stream(prompt):
        if isinstance(event, SessionInitEvent):
            logging.info(f"session_id={event.session_id}")
        elif isinstance(event, ToolCallEvent):
            logging.info(f"→ {event.tool}: {event.args}")
        elif isinstance(event, ToolResultEvent):
            logging.info(f"← {event.tool}: is_error={event.is_error}")
        elif isinstance(event, FinalResponseEvent):
            logging.info(event.content)


if __name__ == "__main__":
    asyncio.run(main())
```

运行方式：

```bash
uv run python your_script.py
```

## 核心概念与 API

### Agent（核心 for-loop）

- `Agent.query(...)`：一次输入，内部 for-loop 执行工具直到结束（或 `TaskComplete`）
- `Agent.query_stream(...)`：流式事件（便于 UI/日志/可观测性）
- `require_done_tool=True`：强制显式结束（配合 `done` 工具 + `TaskComplete`）

### Tool：`@tool(...)`

- 直接装饰 `async def`，自动：
  - 解析函数签名并生成 JSON Schema
  - 支持 Pydantic 参数模型（适合复杂输入）
  - 支持依赖注入（`typing.Annotated[..., Depends(...)]` 或默认值 `Depends(...)`）
- `ephemeral=<N>`：仅保留最近 N 次该工具输出，旧输出会被标记 destroyed，并（可选）offload 到文件系统

### Depends（依赖注入）

```python
from typing import Annotated

from bu_agent_sdk import Depends, tool


def get_db() -> "Database":
    return Database()


@tool("查询用户")
async def get_user(user_id: int, db: Annotated["Database", Depends(get_db)]) -> str:
    return await db.find(user_id)
```

## Session：持久化 / 恢复 / 分叉

`ChatSession` 会把对话增量写入：

- `~/.agent/sessions/<session_id>/context.jsonl`
- 以及（若发生 offload）`~/.agent/sessions/<session_id>/offload/`

### 恢复（resume）

```python
session = agent.chat(session_id="<已有 session_id>")
```

### 分叉（fork）

```python
forked = agent.chat(fork_session="<已有 session_id>")
```

### 获取统计和清空历史

```python
# 获取 token 使用统计
usage = await session.get_usage()
print(f"总 tokens: {usage.total_tokens}")
print(f"总成本: ${usage.total_cost:.4f}")

# 按模型查看
for model, stats in usage.by_model.items():
    print(f"{model}: {stats.total_tokens} tokens")

# 清空会话历史（包括 token 统计和持久化）
session.clear_history()
```

**注意**：
- `get_usage()` 即使在 session 关闭后也可调用（用于获取最终统计）
- `clear_history()` 会清空内存、token 统计，并向 JSONL 写入重置事件
- 如需保留历史，请先使用 `fork_session()` 创建副本

## Context：Compaction / Offload / Ephemeral

### 1) 自动压缩（Compaction）

```python
from bu_agent_sdk.agent import CompactionConfig

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[...],
    compaction=CompactionConfig(threshold_ratio=0.80),
)
```

### 2) Offload（卸载到文件系统）

默认开启（`offload_enabled=True`），并写入：

- `~/.agent/sessions/<session_id>/offload/`

常用配置：

```python
agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[...],
    offload_enabled=True,
    offload_token_threshold=2000,
    offload_root_path=None,
)
```

### 3) Ephemeral（工具输出只保留最近 N 条）

```python
from bu_agent_sdk import tool


@tool("读取大文件（只保留最近 2 次输出）", ephemeral=2)
async def read_big(path: str) -> str:
    return "..."
```

## Subagent：`.agent/subagents/*.md` + `Task`

在项目根目录创建 `.agent/subagents/`，每个文件一个 subagent，例如 `.agent/subagents/researcher.md`：

```markdown
---
name: researcher
description: 做信息收集与整理
tools:
  - WebFetch
  - Grep
  - Read
level: LOW
max_iterations: 30
timeout: 60
---

你是一个研究员，输出需要结构化、可复用。
```

启动后（只要项目里发现了 subagents），主 Agent 会自动注入 `Task` 工具，模型即可调用：

- `Task(subagent_type="researcher", prompt="...", description="...")`

## Skill：`.agent/skills/*/SKILL.md` + `Skill`

在项目根目录创建 `.agent/skills/<skill_name>/SKILL.md`，例如 `.agent/skills/release/SKILL.md`：

```markdown
---
name: release
description: 发布流程规范
---

发布步骤：
1) ...
2) ...

项目路径：{baseDir}
```

SDK 会自动发现 Skills 并注入：

- `Skill` meta-tool（用于加载某个 skill 的完整指令）
- skill 策略提示（写入 Context header）

模型可调用：

- `Skill(skill_name="release")`

## Token 统计与可选计费

### 只统计 tokens（默认）

```python
summary = await agent.get_usage()
```

### 计算成本（需要拉取定价并缓存）

- 代码层：`Agent(include_cost=True, ...)`
- 或环境变量：`bu_agent_sdk_CALCULATE_COST=true`

定价数据会缓存到 `XDG_CACHE_HOME`（默认 `~/.cache/bu_agent_sdk/token_cost/`）。

## 示例代码

仓库内已有更完整示例：

- `bu_agent_sdk/examples/claude_code.py`：Claude Code 风格（沙盒文件系统 + 依赖注入）
- `bu_agent_sdk/examples/chat_session_repl.py`：Session REPL
- `bu_agent_sdk/examples/chat_session_repl_fork.py`：Session 分叉 REPL
- `bu_agent_sdk/examples/subagent_example.py`：Subagent 示例
- `bu_agent_sdk/examples/dependency_injection.py`：依赖注入示例

运行（示例）：

```bash
uv run python bu_agent_sdk/examples/claude_code.py
```

## 许可证

MIT
