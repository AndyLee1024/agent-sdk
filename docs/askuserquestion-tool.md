# AskUserQuestion 系统工具

## 概述

`AskUserQuestion` 是一个系统工具,允许 Agent 在执行过程中向用户询问澄清性问题。这个工具遵循事件驱动的设计模式,通过暂停 Agent 执行流来等待用户输入。

## 核心特性

- ✅ **多问题支持**: 一次可以询问 1-4 个问题
- ✅ **单选/多选**: 每个问题支持单选或多选模式
- ✅ **类型安全**: 使用 Pydantic models 确保参数验证
- ✅ **事件驱动**: 通过 `UserQuestionEvent` 和 `StopEvent` 实现执行流控制

## 工具参数

### 输入格式

```python
{
    "questions": [                    # 1-4 个问题
        {
            "question": str,          # 完整问题文本
            "header": str,            # 短标签 (最多12字符)
            "options": [              # 2-4 个选项
                {
                    "label": str,     # 显示文本 (1-5词)
                    "description": str # 选项说明
                }
            ],
            "multiSelect": bool       # 是否多选 (默认: false)
        }
    ]
}
```

### 约束条件

- `questions`: 1-4 个问题
- `header`: 最多 12 字符
- `options`: 每个问题 2-4 个选项
- `multiSelect`: 默认为 `false`

## 执行流程

```
1. Agent 调用 AskUserQuestion(questions=[...])
   ↓
2. 工具返回 ToolResult (status='waiting_for_input')
   ↓
3. runner_stream 检测到 AskUserQuestion
   ↓
4. yield UserQuestionEvent(questions=..., tool_call_id=...)
   ↓
5. yield StopEvent(reason='waiting_for_input')
   ↓
6. Agent 执行暂停 - UI 展示问题给用户
   ↓
7. 用户回答 → 通过 session.send(UserMessage(...)) 发送（如果在后台长链接模式）或普通 session.query_stream 恢复
   ↓
8. 新一轮 Agent 执行 → 从 UserMessage 中读取答案
```

## 使用示例

### 单个问题 (单选)

```python
{
    "questions": [{
        "question": "Which authentication method should we use?",
        "header": "Auth method",
        "options": [
            {
                "label": "JWT tokens (Recommended)",
                "description": "Stateless, scalable, works well with REST APIs"
            },
            {
                "label": "Session cookies",
                "description": "Traditional server-side sessions"
            },
            {
                "label": "OAuth 2.0",
                "description": "Third-party authentication"
            }
        ],
        "multiSelect": false
    }]
}
```

### 多个问题 (混合单选/多选)

```python
{
    "questions": [
        {
            "question": "Which testing framework would you like to use?",
            "header": "Framework",
            "options": [
                {"label": "Jest", "description": "Most popular"},
                {"label": "Vitest", "description": "Faster"}
            ],
            "multiSelect": false
        },
        {
            "question": "Which types of tests do you want to set up?",
            "header": "Test types",
            "options": [
                {"label": "Unit tests", "description": "Test individual functions"},
                {"label": "Integration tests", "description": "Test components together"},
                {"label": "E2E tests", "description": "Test full application flow"}
            ],
            "multiSelect": true  # 用户可以选择多个
        }
    ]
}
```

## 事件类型

### UserQuestionEvent

当 Agent 调用 AskUserQuestion 时触发。

```python
@dataclass
class UserQuestionEvent:
    questions: list[dict[str, Any]]  # 问题列表
    tool_call_id: str                 # 关联的 tool call ID
```

### StopEvent (waiting_for_input)

Agent 执行暂停,等待用户输入。

```python
@dataclass
class StopEvent:
    reason: Literal['completed', 'max_iterations', 'waiting_for_input']
```

## UI 集成建议

UI 层应该:

1. **监听 UserQuestionEvent**: 当收到此事件时,展示问题给用户
2. **收集用户答案**: 提供界面让用户选择答案
3. **自动添加 "Other" 选项**: 允许用户提供自定义文本输入
4. **通过 UserMessage 发送答案**: 将用户答案格式化为自然语言发送回 Agent

### 答案格式示例

```
我的选择:
- 测试框架: Jest (Recommended)
- 代码风格工具: ESLint, Prettier

请根据这些选择帮我配置项目。
```

## 使用场景

AskUserQuestion 适用于以下场景:

1. **收集用户偏好** - 了解用户想要什么
2. **澄清模糊指令** - 当指令不明确时询问
3. **获取实现决策** - 在工作中让用户选择方案
4. **提供方向选择** - 让用户决定下一步方向

## Plan Mode 注意事项

在 plan mode 下:
- ✅ **可以使用** AskUserQuestion 来澄清需求或选择方案
- ❌ **不要使用** AskUserQuestion 问"计划是否可以"
- 💡 **应该使用** ExitPlanMode 来请求计划批准

## 测试

运行测试:

```bash
uv run python -m pytest comate_agent_sdk/system_tools/tests/test_askuserquestion.py -v
```

查看演示:

```bash
uv run python examples/askuserquestion_example.py
```

## 实现文件

- **工具定义**: `comate_agent_sdk/system_tools/tools.py`
- **使用规则**: `comate_agent_sdk/system_tools/description.py`
- **事件定义**: `comate_agent_sdk/agent/events.py`
- **执行流控制**: `comate_agent_sdk/agent/runner_stream.py`
- **测试**: `comate_agent_sdk/system_tools/tests/test_askuserquestion.py`
- **示例**: `examples/askuserquestion_example.py`
