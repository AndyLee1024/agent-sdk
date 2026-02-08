"""
内置 Subagent 模板 —— 复制此文件并修改以添加新的内置 subagent

步骤：
1. 复制此文件并重命名（如 explore.py）
2. 修改下方的 AgentDefinition 配置
3. 在 __init__.py 中导入并添加到 BUILTIN_AGENTS 列表
"""
from comate_agent_sdk.subagent.models import AgentDefinition

PROMPT = """
Your system prompt here...
"""

example_agent = AgentDefinition(
    name="example",                    # 唯一标识（用于 Task(subagent_type="example")）
    description="Example subagent",    # LLM 可见的描述
    prompt=PROMPT,                     # 系统提示
    tools=None,                        # None=继承父agent全部工具, ["Read","Grep"]=指定工具
    skills=None,                       # None=继承父agent全部skills, ["skill1"]=指定skills
    model=None,                        # None=继承, "sonnet"/"opus"/"haiku"
    level=None,                        # None=继承, "LOW"/"MID"/"HIGH"
    max_iterations=50,                 # 最大迭代次数
    timeout=None,                      # 超时（秒），None=不限
    source="builtin",                  # 固定为 "builtin"，不要修改
)
