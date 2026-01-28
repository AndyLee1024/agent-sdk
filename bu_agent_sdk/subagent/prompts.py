"""
Subagent 系统提示模板
"""

from bu_agent_sdk.subagent.models import AgentDefinition

SUBAGENT_STRATEGY_PROMPT = """
## Task 工具使用指南

你可以使用 Task 工具启动专门化的 Subagent 来完成特定任务。

### 可用的 Subagent 类型

{subagent_list}

### 调用方式

使用 Task 工具时需要提供：
- `subagent_type`: Subagent 名称（如 "code-reviewer", "researcher"）
- `prompt`: 要执行的具体任务
- `description`: 简短描述（3-5 词）

### 并行 vs 串行调用策略

**并行调用**（在同一回复中多次调用 Task 工具）：
- 任务之间相互独立，无数据依赖
- 各任务可以同时进行，不需要等待彼此结果
- 例如：同时让不同 Subagent 处理不同文件

示例：用户要求 "审查 A.py 和 B.py 两个文件"
→ 在同一回复中并行调用：
   - Task(subagent_type="code-reviewer", prompt="审查 A.py", description="审查 A.py")
   - Task(subagent_type="code-reviewer", prompt="审查 B.py", description="审查 B.py")

**串行调用**（等待上一个 Task 完成后再调用下一个）：
- 后续任务依赖前序任务的结果
- 需要根据前序结果决定后续行动
- 例如：先研究，再根据研究结果写作

示例：用户要求 "先研究主题，然后写文章"
→ 分步执行：
   1. Task(subagent_type="researcher", prompt="研究 X 主题")
   2. 等待结果
   3. Task(subagent_type="writer", prompt="根据研究结果写文章: {{研究结果}}")
"""


def generate_subagent_prompt(agents: list[AgentDefinition]) -> str:
    """生成 Subagent 策略提示

    Args:
        agents: AgentDefinition 列表

    Returns:
        完整的系统提示字符串
    """
    subagent_list = "\n".join([f"- **{a.name}**: {a.description}" for a in agents])
    return SUBAGENT_STRATEGY_PROMPT.format(subagent_list=subagent_list)
