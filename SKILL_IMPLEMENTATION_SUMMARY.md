# Skill 功能实现总结

## ✅ 完成状态

已完全按照 plan 文件（`/home/hyc/.claude/plans/enchanted-scribbling-pike.md`）成功实现完整的 Skill 系统。

---

## 📋 实现的功能清单

### 核心特性 ✅

- ✅ **工具权限临时修改**（`allowed-tools` 字段）
- ✅ **模型临时切换**（`model` 字段）
- ✅ **资源打包支持**（`scripts/`, `references/`, `assets/` 目录）
- ✅ **与 Claude Code 完全兼容**（SKILL.md 格式）
- ✅ **每个 Agent 独立支持 Skill**（主 Agent 和 Subagent 都能使用）
- ✅ **仅 LLM 调用**（不支持手动调用如 `/skill-name`）

---

## 📁 新增文件

### Skill 模块 (`bu_agent_sdk/skill/`)

| 文件 | 功能 | 行数 |
|------|------|------|
| `__init__.py` | 模块入口，导出核心类和函数 | 11 |
| `models.py` | SkillDefinition 数据模型（兼容 Claude Code） | 146 |
| `loader.py` | Skill 自动发现和加载 | 66 |
| `skill_tool.py` | Skill meta-tool 实现（动态描述生成） | 92 |
| `context.py` | apply_skill_context() 函数（持久化修改） | 35 |

### 测试和示例

| 文件 | 功能 |
|------|------|
| `test_skill_basic.py` | 基础功能测试 |
| `test_skill_integration.py` | 集成测试（含 Subagent） |
| `examples/skill_examples.py` | 使用示例 |

---

## 🔧 修改的文件

### 1. **消息类型扩展** (`bu_agent_sdk/llm/messages.py`)

```python
class UserMessage(_MessageBase):
    # ... 现有字段 ...

    is_meta: bool = False  # 新增
    """Whether this is a meta message (for Skill prompt injection).

    Meta messages are sent to the LLM but hidden from the user interface.
    Used to inject Skill prompts without cluttering the conversation UI.
    """
```

**影响**: 小（向后兼容，默认值为 `False`）

---

### 2. **AgentDefinition 扩展** (`bu_agent_sdk/subagent/models.py`)

```python
@dataclass
class AgentDefinition:
    # ... 现有字段 ...
    skills: list[str] | None = None  # 新增
    """可用的 Skills 名称列表（限制 Subagent 可用的 Skills）"""
```

**功能**: 支持在 Subagent frontmatter 中通过 `skills` 字段限制可用的 Skills

**示例**:
```yaml
---
name: file-ops
tools: read_file, write_file
skills: file-skill, editor-skill  # 只允许这两个 Skills
---
```

---

### 3. **Agent 类集成** (`bu_agent_sdk/agent/service.py`)

#### 新增字段

```python
@dataclass
class Agent:
    # ... 现有字段 ...

    # Skill support
    skills: list | None = None  # list[SkillDefinition]
    """List of SkillDefinition for Skill support. Auto-discovered if None."""

    _active_skill_name: str | None = field(default=None, repr=False)
    """Currently active Skill name (only one Skill can be active per Agent)."""
```

#### 新增方法

| 方法 | 功能 | 行数 |
|------|------|------|
| `_setup_skills()` | 自动发现并加载 Skills，创建 Skill 工具 | 33 |
| `_execute_skill_call()` | 执行 Skill 调用（注入消息、应用 context） | 45 |
| `_rebuild_skill_tool()` | 重建 Skill 工具（用于 Subagent 筛选后） | 17 |

#### 修改逻辑

1. **`__post_init__`**: 添加 `self._setup_skills()` 调用
2. **`_execute_tool_call()`**: 检测 Skill 工具调用并特殊处理

---

### 4. **Task 工具修改** (`bu_agent_sdk/subagent/task_tool.py`)

```python
# 创建 Subagent（继承父级依赖覆盖）
subagent = Agent(
    llm=llm,
    tools=tools,
    system_prompt=agent_def.prompt,
    max_iterations=agent_def.max_iterations,
    compaction=agent_def.compaction,
    dependency_overrides=parent_dependency_overrides,
    _is_subagent=True,
)

# ⭐ 新增：Subagent Skills 筛选
if agent_def.skills is not None and subagent.skills:
    allowed_skill_names = set(agent_def.skills)
    subagent.skills = [s for s in subagent.skills if s.name in allowed_skill_names]
    # 重新创建 Skill 工具（更新工具描述）
    subagent._rebuild_skill_tool()
    logging.info(
        f"Filtered subagent '{subagent_type}' skills to: {[s.name for s in subagent.skills]}"
    )
```

**功能**: 根据 `AgentDefinition.skills` 字段筛选 Subagent 可用的 Skills

---

### 5. **公开接口导出** (`bu_agent_sdk/__init__.py`)

```python
from bu_agent_sdk.skill import (
    SkillDefinition,
    apply_skill_context,
    create_skill_tool,
    discover_skills,
)

__all__ = [
    # ... 现有导出 ...
    # Skill support
    "SkillDefinition",
    "discover_skills",
    "create_skill_tool",
    "apply_skill_context",
]
```

---

## 🎯 核心设计决策

### 1. Skill vs Subagent 差异

| 维度 | Subagent | Skill |
|------|----------|-------|
| **执行方式** | 启动新 Agent 实例 | 注入 prompt 到当前 Agent |
| **消息历史** | 独立的消息历史 | 共享父 Agent 的消息历史 |
| **工具权限** | 通过 `tools` 筛选 | 临时修改父 Agent 的 `_tool_map` |
| **模型** | 可指定不同模型 | 临时替换父 Agent 的 `llm` |
| **返回值** | 返回文本结果 | 无返回值（修改 Agent 行为） |
| **使用场景** | 独立任务（如代码审查） | 临时专业化（如使用特定格式写作） |

### 2. Skill 作用范围

- ✅ **统一模式**：主 Agent 和 Subagent 都通过 `Skill` 工具调用 Skills
- ✅ **独立发现**：每个 Agent 启动时独立扫描并加载可用 Skills
  - 项目级：`.agent/skills/skillname/SKILL.md`
  - 用户级：`~/.agent/skills/skillname/SKILL.md`
  - 优先级：项目级 > 用户级（同名时项目级覆盖）
- ✅ **可配置过滤**：Subagent 可通过 `AgentDefinition.skills` 字段限制可用的 Skills
- ✅ **防嵌套保护**：避免同一 Agent 重复调用同名 Skill

### 3. 消息注入机制

**双消息注入**：

1. **元数据消息**（`is_meta=False`，用户可见）
   ```xml
   <skill-message>The "skill-creator" skill is loading</skill-message>
   <skill-name>skill-creator</skill-name>
   ```

2. **Prompt 消息**（`is_meta=True`，用户不可见）
   ```markdown
   你是一个 Skill 创建专家...

   ## 工作流程
   1. 询问用户需求
   2. 创建 SKILL.md
   ...

   Base directory: /home/user/.agent/skills/skill-creator
   ```

### 4. Execution Context 修改（持久化）

**设计决策**：Skill 的 execution context 修改是**持久化的**，一旦激活就保持生效，不会自动退出。

**理由**：
1. 难以界定退出时机（LLM 停止调用工具？用户发送新消息？固定轮数？）
2. 符合 Claude Code 设计（Skill 也是注入后持续生效）
3. 简化实现（避免复杂的退出判断逻辑和状态管理）

**实现方式**：

```python
def apply_skill_context(agent: "Agent", skill_def: "SkillDefinition") -> None:
    """应用 Skill 的 execution context 修改（持久化）"""

    # 1. 应用 Skill 工具权限
    if skill_def.allowed_tools:
        allowed_set = set(skill_def.allowed_tools)
        agent.tools = [t for t in agent.tools if t.name in allowed_set]
        agent._tool_map = {k: v for k, v in agent._tool_map.items() if k in allowed_set}

    # 2. 应用 Skill 模型切换
    if skill_def.model and skill_def.model != "inherit":
        from bu_agent_sdk.subagent.task_tool import resolve_model
        agent.llm = resolve_model(skill_def.model, agent.llm)
```

**注意事项**：
- Skill 的修改会**一直生效**，直到 Agent 实例销毁
- 如果多个 Skill 被依次调用，后面的 Skill 会在前面 Skill 的基础上进一步限制
- 同一 Agent 不能同时激活多个 Skill（有重复调用保护）

---

## 📖 SKILL.md 格式

### 完整示例

```yaml
---
name: explain-code              # 可选，省略则使用目录名
description: Explains code with visual diagrams  # 推荐
allowed-tools: Read, Write      # 可选，允许的工具列表（逗号分隔或 YAML 列表）
model: inherit                  # 可选，使用的模型（inherit/gpt-4o/claude-sonnet-4 等）
disable-model-invocation: false # 可选，是否禁用自动加载
user-invocable: true            # 可选（SDK 暂不使用）
argument-hint: [code-path]      # 可选（SDK 暂不使用）
---

你是一个代码解释专家，擅长用清晰的语言和可视化图表解释代码逻辑。

## 你的职责
1. 分析代码结构
2. 解释核心逻辑
3. 生成流程图

## 资源目录
- Base: {baseDir}
- Scripts: {baseDir}/scripts/
- References: {baseDir}/references/
```

### 字段说明

| 字段 | 必需 | 类型 | 说明 |
|------|------|------|------|
| `name` | ❌ | string | Skill 名称，省略时使用目录名 |
| `description` | ❌ | string | Skill 描述，LLM 用来决定何时使用 |
| `allowed-tools` | ❌ | string \| list | 允许的工具列表（限制 Skill 可用的工具） |
| `model` | ❌ | string | 使用的模型（`inherit` 继承父级，或指定模型名） |
| `disable-model-invocation` | ❌ | boolean | 是否禁用自动加载（默认 `false`） |
| `user-invocable` | ❌ | boolean | 暂不使用（预留字段） |
| `argument-hint` | ❌ | string | 暂不使用（预留字段） |

### 资源打包

```
.agent/skills/skillname/
├── SKILL.md       # 必需
├── scripts/       # 可选：脚本文件
├── references/    # 可选：参考文档
└── assets/        # 可选：资源文件（图片等）
```

---

## 🚀 使用方式

### 1. 自动发现 Skills

```python
from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI

# 1. 创建 .agent/skills/my-skill/SKILL.md
# 2. 创建 Agent（会自动发现）
agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[...],
)

# 3. 查看发现的 Skills
print([s.name for s in agent.skills])
```

### 2. 手动定义 Skills

```python
from bu_agent_sdk import Agent, SkillDefinition

skill = SkillDefinition(
    name="my-skill",
    description="My custom skill",
    prompt="You are a specialized assistant...",
    allowed_tools=["tool1", "tool2"],
    model="gpt-4o-mini",
)

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[...],
    skills=[skill],  # 手动传入
)
```

### 3. Subagent 限制 Skills

```python
from bu_agent_sdk import Agent, AgentDefinition

# 创建 Subagent 定义（限制可用的 Skills）
subagent_def = AgentDefinition(
    name="limited-agent",
    description="受限的 Agent",
    prompt="你是受限的 Agent",
    tools=["read_file", "write_file"],
    skills=["file-skill"],  # 只允许 file-skill
)

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[...],
    agents=[subagent_def],
)

# 当 LLM 调用 Task(subagent_type="limited-agent", ...)
# 创建的 Subagent 只能使用 file-skill，其他 Skills 会被过滤掉
```

### 4. LLM 调用 Skill

```python
# LLM 会看到 Skill 工具的描述：
# "Execute a skill within the main conversation.
#  Available skills:
#    - "explain-code": Explains code with visual diagrams
#    - "writer": Professional document writer
#  ..."

# LLM 调用 Skill：
# Skill(skill_name="explain-code")

# Agent 会：
# 1. 注入元数据消息（用户可见）
# 2. 注入 prompt 消息（用户不可见）
# 3. 应用 execution context 修改（工具权限、模型）
# 4. 返回成功消息
```

---

## ✅ 测试验证

### 基础功能测试 (`test_skill_basic.py`)

- ✅ Skill 自动发现
- ✅ 手动定义 Skill
- ✅ 从 Markdown 解析 Skill
- ✅ `disable_model_invocation` 过滤

### 集成测试 (`test_skill_integration.py`)

- ✅ `AgentDefinition.skills` 字段解析
- ✅ Subagent Skills 筛选
- ✅ Skill execution context 修改（工具权限限制）
- ✅ Skill 重复调用保护

### 运行结果

```bash
$ uv run python test_skill_basic.py
==================================================
✅ 所有测试通过！
==================================================

$ uv run python test_skill_integration.py
==================================================
✅ 所有集成测试通过！
==================================================

$ uv run python examples/skill_examples.py
==================================================
✅ 所有示例运行完成
==================================================
```

---

## 🎉 完成情况

### 按 Plan 阶段完成

| 阶段 | 任务 | 状态 |
|------|------|------|
| **阶段 1** | 数据模型和加载器 | ✅ |
| **阶段 2** | 消息类型扩展 | ✅ |
| **阶段 3** | Execution Context 修改 | ✅ |
| **阶段 4** | Skill 工具实现 | ✅ |
| **阶段 5** | Agent 集成 | ✅ |
| **阶段 6** | 公开接口导出 | ✅ |

### 额外完成

- ✅ 完整的单元测试和集成测试
- ✅ 使用示例和文档
- ✅ 向后兼容（所有现有代码无需修改）

---

## 📚 后续建议

### 可选增强（未来版本）

1. **Skill 退出机制**
   - 提供 `ExitSkill` 工具，让 LLM 可以主动退出当前 Skill
   - 提供 `agent.exit_skill()` 方法，让外部代码控制退出
   - 在 SkillDefinition 中添加 `persistent: bool` 字段

2. **Skill 历史记录**
   - 记录哪些 Skill 被调用过，避免重复

3. **Skill 缓存**
   - 缓存 SkillDefinition 解析结果，提升性能

4. **热重载**
   - 监听 `.agent/skills/` 目录变化，自动重新加载

5. **Subagent 预加载模式**
   - 为特定 Subagent 支持将 Skills 直接注入到 system prompt
   - 对齐 Claude Code 原始设计

6. **多 Skill 支持**
   - 允许同一 Agent 激活多个 Skill
   - 需要设计工具权限和模型的合并策略

---

## 📝 总结

✅ **完全按照 plan 实现了完整的 Skill 系统**
✅ **与 Claude Code SKILL.md 格式 100% 兼容**
✅ **主 Agent 和 Subagent 都支持 Skill**
✅ **支持工具权限隔离、模型切换、资源打包**
✅ **通过了完整的测试验证**
✅ **提供了详细的使用示例**

**Skill 系统已可以投入使用！** 🚀
