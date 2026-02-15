# comate-agent-sdk

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
uv add comate-agent-sdk
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
COMATE_AGENT_SDK_LLM_LOW="openai:gpt-4o-mini"
COMATE_AGENT_SDK_LLM_MID="openai:gpt-4o"
COMATE_AGENT_SDK_LLM_HIGH="openai:gpt-4o"
COMATE_AGENT_SDK_LLM_LOW_BASE_URL="http://192.168.100.1:4141/v1"
COMATE_AGENT_SDK_LLM_MID_BASE_URL="http://192.168.100.1:4142/v1"
COMATE_AGENT_SDK_LLM_HIGH_BASE_URL="http://192.168.100.1:4143/v1"
```

## 配置文件：settings.json 和 AGENTS.md

SDK 支持通过配置文件管理 LLM 配置和 Agent 指令，分为 **user 级**（全局）和 **project 级**（项目）两层：

| 配置文件 | User 级（全局） | Project 级 | 优先级 |
|---------|----------------|-----------|--------|
| `settings.json` | `~/.agent/settings.json` | `{项目根}/.agent/settings.json` | project **字段级覆盖** user |
| `AGENTS.md` | `~/.agent/AGENTS.md` | `{项目根}/AGENTS.md` 或 `{项目根}/.agent/AGENTS.md` | project **完全替代** user |

### settings.json 配置模板

```json
{
  "llm_levels": {
    "LOW": "openai:gpt-4o-mini",
    "MID": "openai:gpt-4o",
    "HIGH": "anthropic:claude-opus-4-5"
  },
  "llm_levels_base_url": {
    "LOW": "http://192.168.100.1:4141/v1",
    "MID": null,
    "HIGH": "https://api.anthropic.com"
  }
}
```

**说明**：
- `llm_levels`：三档模型配置，格式为 `"provider:model"`（`provider` 可选：`openai`、`anthropic`、`google`）
- `llm_levels_base_url`：可选，为每档模型指定 base_url（`null` 表示使用默认）
- **优先级**（从高到低）：代码参数 `llm_levels=` > project settings.json > user settings.json > 环境变量 > 默认值

#### 字段级覆盖示例

假设你有：

**`~/.agent/settings.json`（user 级）**：
```json
{
  "llm_levels": {
    "LOW": "openai:gpt-4o-mini",
    "MID": "openai:gpt-4o",
    "HIGH": "openai:gpt-4o"
  },
  "llm_levels_base_url": {
    "LOW": "http://192.168.100.1:4141/v1"
  }
}
```

**`{项目根}/.agent/settings.json`（project 级）**：
```json
{
  "llm_levels": {
    "HIGH": "anthropic:claude-opus-4-5"
  }
}
```

**最终生效**：
- `llm_levels`：project 的 `llm_levels` **完全覆盖** user（只有 `HIGH` 生效，`LOW` 和 `MID` 消失）
- `llm_levels_base_url`：project 未定义，回退到 user 的配置

### AGENTS.md 加载规则

`AGENTS.md` 是用于写入 Agent 背景指令的 Markdown 文件，会被自动注入到 `memory`（类似 system prompt，但不计入 prompt token 限制）。

**搜索路径**（按优先级）：

1. **Project 级**（任一存在即生效）：
   - `{项目根}/AGENTS.md`
   - `{项目根}/.agent/AGENTS.md`

2. **User 级**（仅当 project 级不存在时 fallback）：
   - `~/.agent/AGENTS.md`

**重要规则**：
- 当 project 级存在任何 `AGENTS.md` 时，user 级会被**完全忽略**（不会合并）
- 只有在 project 级完全不存在时，才会 fallback 到 user 级
- 如果用户代码手动指定了 `memory=...`，则不会自动加载任何 `AGENTS.md`

#### AGENTS.md 示例

**`~/.agent/AGENTS.md`（user 级，全局指令）**：
```markdown
# 全局 Agent 规则

- 所有代码必须使用 f-string
- 必须使用 logging 模块，禁止 print
```

**`{项目根}/.agent/AGENTS.md`（project 级，项目特定指令）**：
```markdown
# 本项目 Agent 规则

这是一个 Django 项目，请遵循：
- 使用 Django ORM 查询数据库
- 在 views.py 中编写视图函数
- 测试文件放在 tests/ 目录
```

### 控制配置加载：`setting_sources`

你可以在代码中显式控制加载哪些配置：

```python
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig

# 默认：加载 user 和 project 两层
agent = Agent(llm=..., config=AgentConfig(setting_sources=("user", "project")))

# 只加载 project 级配置
agent = Agent(llm=..., config=AgentConfig(setting_sources=("project",)))

# 只加载 user 级配置
agent = Agent(llm=..., config=AgentConfig(setting_sources=("user",)))

# 完全不加载配置文件（向后兼容模式）
agent = Agent(llm=..., config=AgentConfig(setting_sources=None))
```

**注意**：`setting_sources` 同时控制 `settings.json` 和 `AGENTS.md` 的加载范围。

## MCP：接入外部工具（stdio/sse/http）与本地 SDK Server

SDK 支持通过 MCP（Model Context Protocol）接入外部工具生态。MCP tools 会被映射为普通 `Tool`，并遵循统一命名规则：

- **所有 MCP tools 都以 `mcp__` 开头**
- 命名格式：`mcp__{server_alias}__{tool_name}`
  - `server_alias` 来自 `AgentConfig(mcp_servers={...})` 的 key，或 `.mcp.json` 里的 key
  - `tool_name` 来自 MCP server 返回的原始 tool name（会做安全字符规整）

> 说明：SDK 会在**第一次调用 LLM 前**懒加载 MCP tools；如果你使用 `ChatSession.resume()` 恢复会话，SDK 会在下一次调用 LLM 前自动刷新 MCP tools。

### 1) 配置 MCP server（stdio / sse / http）

默认配置文件位置（两层合并，project 覆盖同名 alias 的 user 配置）：

- **User 级**：`~/.agent/.mcp.json`
- **Project 级**：`{项目根}/.agent/.mcp.json`

`.mcp.json` 支持两种等价写法：

1) 直接写成 `alias -> config`：

```json
{
  "fs": { "type": "stdio", "command": "python", "args": ["-m", "my_fs_mcp_server"] }
}
```

2) 包一层 `servers`（更清晰，推荐）：

```json
{
  "servers": {
    "fs": { "type": "stdio", "command": "python", "args": ["-m", "my_fs_mcp_server"] }
  }
}
```

#### stdio 示例（本机启动子进程）

```json
{
  "servers": {
    "calc": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "my_calc_mcp_server"],
      "env": { "LOG_LEVEL": "INFO" }
    }
  }
}
```

> `type` 对 stdio 可省略（缺省即按 stdio 处理）。

#### SSE 示例（远程/本地 SSE 端点）

```json
{
  "servers": {
    "search": {
      "type": "sse",
      "url": "http://127.0.0.1:8000/sse",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

#### HTTP 示例（Streamable HTTP 端点）

```json
{
  "servers": {
    "internal": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "X-API-Key": "YOUR_KEY" }
    }
  }
}
```

#### 代码中显式指定/覆盖配置

你也可以直接在代码里传 `mcp_servers`（会覆盖默认文件发现逻辑）：

```python
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig

agent = Agent(
    llm=...,
    config=AgentConfig(
        mcp_servers={
            "internal": {"type": "http", "url": "http://127.0.0.1:8000/mcp"},
        },
        tools=["mcp__internal__some_tool"],
    ),
)
```

或者传一个配置文件路径：

```python
from comate_agent_sdk import Agent

from comate_agent_sdk.agent import AgentConfig

agent = Agent(llm=..., config=AgentConfig(mcp_servers="/abs/path/to/.mcp.json", tools=[...]))
```

> 注意：`.mcp.json` **不支持** `type="sdk"`（因为 `instance` 无法序列化），`sdk` 只能代码注入。

### 2) 创建本地 MCP server（SDK in-process，FastMCP）

如果你希望把一组工具“像 MCP server 一样”以内嵌方式提供给 Agent，可以用 `create_sdk_mcp_server()`。

要点：
- 使用 `@mcp_tool(name=..., description=...)` 声明工具
- **输入 schema 来自函数签名的类型注解**（推荐显式参数，不要用 `args: dict`）
- 注册到 Agent 时，用 `mcp_servers` dict 的 key 作为 `server_alias`（决定最终 tool name 前缀）

```python
import asyncio
import logging

from comate_agent_sdk import Agent, create_sdk_mcp_server, mcp_tool
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.llm import ChatOpenAI

logging.basicConfig(level=logging.INFO)


@mcp_tool(name="add", description="Add two numbers")
async def add(a: float, b: float) -> str:
    return f"Sum: {a + b}"


@mcp_tool(name="multiply", description="Multiply two numbers")
async def multiply(a: float, b: float) -> str:
    return f"Product: {a * b}"


calculator = create_sdk_mcp_server(
    name="calculator",
    version="2.0.0",
    tools=[add, multiply],
)


async def main() -> None:
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        config=AgentConfig(
            mcp_servers={"calc": calculator},  # alias = "calc"
            tools=["mcp__calc__add", "mcp__calc__multiply"],  # allowlist（推荐）
            agents=[],  # 如不需要 subagent，建议显式禁用，减少自动注入
        ),
    )

    result = await agent.query("请用工具计算 12.5 + 3.5，然后再乘以 2。")
    logging.info(result)


if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
uv run python your_script.py
```

## 快速上手：Claude Code 风格（系统工具 + 显式 done + Session）

下面这个最小示例具备：

- Claude Code 风格系统工具（`Bash/Read/Write/Edit/Grep/Glob/LS/TodoWrite/WebFetch`）
- 显式 `done` 工具（防止“无工具调用就提前结束”）
- `ChatSession`（会话持久化到 `~/.agent/sessions/<session_id>/`）

### 系统工具依赖（按工具）

> 说明：Python 包依赖会由 `uv sync` / `uv add comate-agent-sdk` 自动安装；外部命令依赖需要你在系统层安装。

| 工具 | 依赖 | 缺失时行为 | 备注 |
|---|---|---|---|
| `Bash` | 系统 shell（Linux/macOS 默认具备） | 无法执行命令 | 具体依赖取决于你在命令里调用的程序（如 `git` / `uv` / `python` 等） |
| `Read` | 无 | - | - |
| `Write` | 无 | - | - |
| `Edit` | 无 | - | - |
| `MultiEdit` | 无 | - | - |
| `Glob` | 无 | - | - |
| `LS` | 无 | - | - |
| `Grep` | 可选：`rg`（ripgrep） | 若未安装 `rg`，会自动回退到 Python 实现（较慢，部分高级能力可能缺失） | 推荐安装 ripgrep：Ubuntu/Debian `sudo apt-get install ripgrep`；macOS `brew install ripgrep`；Windows `winget install BurntSushi.ripgrep.MSVC` |
| `TodoWrite` | 无 | - | 会在 session 目录下写入/清理 `todos.json` |
| `WebFetch` | Python 包：`curl-cffi`、`markdownify`；以及 `llm_levels["LOW"]` | 缺包则报错；未配置 LOW 模型则无法按预期调用低档模型 | 需要联网访问目标 URL；会调用低档模型做摘要/抽取 |

```python
import asyncio
import logging

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig, SessionInitEvent, StopEvent, TextEvent, ToolCallEvent, ToolResultEvent
from comate_agent_sdk.llm import ChatOpenAI
from comate_agent_sdk.tools import get_default_registry

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    llm_levels = {
        "LOW": ChatOpenAI(model="gpt-4o-mini"),
        "MID": ChatOpenAI(model="gpt-4o"),
        "HIGH": ChatOpenAI(model="gpt-4o"),
    }

    agent = Agent(
        llm=llm_levels["MID"],
        config=AgentConfig(
            llm_levels=llm_levels,
            tools=get_default_registry().all(),
            include_cost=False,
        ),
    )

    session = agent.chat()

    prompt = (
        "请在当前项目根目录里：\n"
        "1) 找到 pyproject.toml\n"
        "2) 读取并告诉我 [project] 的 name\n"
        "3) 完成后给出结论\n"
    )

    async for event in session.query_stream(prompt):
        if isinstance(event, SessionInitEvent):
            logging.info(f"session_id={event.session_id}")
        elif isinstance(event, ToolCallEvent):
            logging.info(f"→ {event.tool}: {event.args}")
        elif isinstance(event, ToolResultEvent):
            logging.info(f"← {event.tool}: is_error={event.is_error}")
        elif isinstance(event, TextEvent):
            logging.info(event.content)
        elif isinstance(event, StopEvent):
            logging.info(f"done reason={event.reason}")


if __name__ == "__main__":
    asyncio.run(main())
```

运行方式：

```bash
uv run python your_script.py
```

## 核心概念与 API

### Agent（核心 for-loop）

- `Agent.query(...)`：一次输入，内部 for-loop 执行工具直到结束
- `Agent.query_stream(...)`：流式事件（便于 UI/日志/可观测性）

### Tool：`@tool(...)`

- 直接装饰 `async def`，自动：
  - 解析函数签名并生成 JSON Schema
  - 支持 Pydantic 参数模型（适合复杂输入）
  - 支持依赖注入（`typing.Annotated[..., Depends(...)]` 或默认值 `Depends(...)`）
- `ephemeral=<N>`：仅保留最近 N 次该工具输出，旧输出会被标记 destroyed，并（可选）offload 到文件系统

### Depends（依赖注入）

```python
from typing import Annotated

from comate_agent_sdk import Depends, tool


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
from comate_agent_sdk.agent import AgentConfig, CompactionConfig

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    config=AgentConfig(
        tools=[...],
        compaction=CompactionConfig(threshold_ratio=0.80),
        emit_compaction_meta_events=False,  # 调试事件开关（默认关闭）
    ),
)
```

当前压缩行为（直接替换旧策略）：

- 工具历史按“工具块”处理：仅保留最近 5 块，更早块整块删除
- 保留块内字段阈值截断：
  - `tool_call.arguments > 500 tokens` 才截断
  - `tool_result.content > 600 tokens` 才截断
  - 截断保留前 200 tokens，并追加 `[TRUNCATED original~N tokens]`
- `user/assistant` 至少保留最近 12 轮（`is_meta=True` 不计轮次）
- 选择性压缩后始终执行 summary；任一步失败原子回滚
- summary 失败会自动短重试；连续失败进入短冷却，避免高频重复失败

### 2) Offload（卸载到文件系统）

默认开启（`offload_enabled=True`），并写入：

- `~/.agent/sessions/<session_id>/offload/`

常用配置：

```python
from comate_agent_sdk.agent import AgentConfig

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    config=AgentConfig(
        tools=[...],
        offload_enabled=True,
        offload_token_threshold=2000,
        offload_root_path=None,
    ),
)
```

### 3) Ephemeral（工具输出只保留最近 N 条）

```python
from comate_agent_sdk import tool


@tool("读取大文件（只保留最近 2 次输出）", ephemeral=2)
async def read_big(path: str) -> str:
    return "..."
```

### 4) 相关文档

- [上下文压缩策略说明](./docs/compression-strategy.md)
- [上下文压缩流程详解](./docs/compaction-flow.md)
- [上下文文件系统（Offload）](./docs/context_filesystem.md)

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

### Subagent 模型配置

Subagent 支持两种方式指定使用的 LLM 模型：

#### 1. 使用档位 (level)

推荐使用档位来控制模型性能和成本：

```yaml
---
name: quick-helper
description: 快速助手
level: LOW      # 使用低档位（快速、便宜）
---
```

支持的档位：
- `LOW`: 快速模型（如 haiku）
- `MID`: 标准模型（如 sonnet）
- `HIGH`: 高性能模型（如 opus）

#### 2. 使用别名 (model)

也可以使用别名直接指定：

```yaml
---
name: expert
description: 专家
model: opus     # 使用opus模型
---
```

支持的别名：
- `sonnet`: 映射到MID档位
- `opus`: 映射到HIGH档位
- `haiku`: 映射到LOW档位
- `inherit`: 继承父agent（等同于不指定）

#### 3. 默认行为

如果不指定 `model` 或 `level`，subagent 将继承父 agent 的模型。

**注意**: 不支持直接指定完整的模型名称（如 `model: gpt-4o`），仅支持上述别名。

启动后（只要项目里发现了 subagents），主 Agent 会自动注入 `Task` 工具，模型即可调用：

- `Task(subagent_type="researcher", prompt="...", description="...")`

### 自动发现与 `agents` 语义（重要）

Subagent 支持**自动发现**（从文件系统加载）与**代码显式传入**两种方式，并且可以混用。

#### 自动发现路径与优先级

SDK 会从以下路径发现 subagent 定义（`.md` 文件）：

- **Project 级（优先）**：`{project_root}/.agent/subagents/*.md`
- **User 级（fallback）**：`~/.agent/subagents/*.md`

优先级规则：
- 当 project 级存在任意 `.md` 文件时，**完全忽略** user 级（不会合并）。
- `project_root` 未显式传入时，默认使用当前工作目录（`cwd`）。

#### `AgentConfig(agents=...)` 的约定

`agents` 参数用于控制是否启用/合并 subagent（注意 `None` 与 `[]` 的语义不同）：

| 传参 | 行为 | 典型用途 |
|---|---|---|
| `agents=None`（默认） | 允许自动发现；若发现到 subagent，会注入系统 `Task` 工具 | 纯自动发现 |
| `agents=[]` | **显式禁用**自动发现；不会注入系统 `Task` 工具 | 测试隔离 / 完全不启用 subagent |
| `agents=[...]`（非空） | 自动发现 + 代码传入 **合并**（同名以代码传入为准） | 混合模式（推荐） |

### `Task` 是保留名：避免静默覆盖

当启用了 subagent（最终 `agent.agents` 非空）时，SDK 会注入系统级 `Task` 工具作为 subagent 调度入口，因此：

- 若你在 `tools=[...]` 中手动提供了同名工具 `Tool(name="Task")`，SDK 会直接抛出 `ValueError`（避免静默替换导致误用）。

解决方式（三选一）：
1) 将你的工具改名（不要叫 `Task`）  
2) 显式禁用 subagent：`Agent(..., config=AgentConfig(agents=[]))`  
3) 移除/调整自动发现的 subagent定义（例如删除/修改 `.agent/subagents/*.md`）  

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

## 可观测性：Langfuse 集成

SDK 支持通过 [Langfuse](https://langfuse.com) 进行 LLM 调用的可观测性监控（tracing、usage、latency 等）。

### 启用方式

只需设置以下三个环境变量（**必须全部设置**才会启用）：

```bash
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # 或你的自托管地址
```

设置后，SDK 会自动启用 Langfuse 追踪：

- **OpenAI**：使用 Langfuse 包装的 OpenAI 客户端
- **Anthropic**：使用 OpenTelemetry instrumentation

### Anthropic 额外依赖

如果使用 Anthropic 模型并希望启用 Langfuse 追踪，需要额外安装：

```bash
pip install opentelemetry-instrumentation-anthropic
# 或
uv add opentelemetry-instrumentation-anthropic
```

> **注意**：如果未安装此包但设置了环境变量，SDK 会输出警告日志但不会影响正常运行。

### 不启用时的行为

如果未设置（或未完整设置）上述三个环境变量，SDK 会使用原生的 LLM 客户端，不会有任何额外开销。

## Token 统计与可选计费

### 只统计 tokens（默认）

```python
summary = await agent.get_usage()
```

### 计算成本（需要拉取定价并缓存）

- 代码层：`Agent(config=AgentConfig(include_cost=True, ...))`
- 或环境变量：`comate_agent_sdk_CALCULATE_COST=true`

定价数据会缓存到 `XDG_CACHE_HOME`（默认 `~/.cache/comate_agent_sdk/token_cost/`）。

## 示例代码

仓库内已有更完整示例：

- `comate_agent_sdk/examples/claude_code.py`：Claude Code 风格（沙盒文件系统 + 依赖注入）
- `comate_agent_sdk/examples/chat_session_repl.py`：Session REPL
- `comate_agent_sdk/examples/chat_session_repl_fork.py`：Session 分叉 REPL
- `comate_agent_sdk/examples/subagent_example.py`：Subagent 示例
- `comate_agent_sdk/examples/dependency_injection.py`：依赖注入示例

运行（示例）：

```bash
uv run python comate_agent_sdk/examples/claude_code.py
```

## 许可证

MIT
