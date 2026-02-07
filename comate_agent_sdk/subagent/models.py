"""
Subagent 数据模型定义
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from comate_agent_sdk.agent.compaction import CompactionConfig
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.tokens.views import UsageSummary


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Subagent 定义"""

    name: str  # 唯一标识
    description: str  # 描述（LLM 可见）
    prompt: str  # 系统提示（Markdown 正文部分）
    tools: list[str] | None = None  # 可用工具名称
    skills: list[str] | None = None  # 可用的 Skills 名称列表（限制 Subagent 可用的 Skills）
    model: str | None = None
    level: LLMLevel | None = None
    max_iterations: int = 50  # 最大迭代次数
    timeout: float | None = None  # 超时时间（秒）
    compaction: CompactionConfig | None = None  # 上下文压缩配置

    @classmethod
    def from_markdown(cls, content: str) -> "AgentDefinition":
        """从 Markdown 文件内容解析 AgentDefinition

        Args:
            content: Markdown 文件内容

        Returns:
            解析后的 AgentDefinition

        Raises:
            ValueError: 如果格式无效
        """
        # 分离 frontmatter 和正文
        if not content.startswith("---"):
            raise ValueError("Markdown must start with YAML frontmatter (---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Invalid frontmatter format")

        frontmatter = yaml.safe_load(parts[1])
        prompt = parts[2].strip()

        # 解析 tools（支持逗号分隔字符串或列表）
        tools_raw = frontmatter.get("tools")
        if isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        else:
            tools = tools_raw

        # 解析 skills（支持逗号分隔字符串或列表）
        skills_raw = frontmatter.get("skills")
        if isinstance(skills_raw, str):
            skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
        else:
            skills = skills_raw

        init_kwargs: dict[str, object] = {}
        if "model" in frontmatter:
            model = frontmatter.get("model")
            if model is not None and not isinstance(model, str):
                model = str(model)
            if model is not None:
                model = model.strip()
                if not model or model.lower() == "inherit":
                    model = None
                elif model.lower() not in ("sonnet", "opus", "haiku"):
                    # 警告：不支持的model值
                    logging.warning(
                        f"Subagent '{frontmatter.get('name')}': "
                        f"不支持的model值 '{model}'。仅支持：sonnet/opus/haiku/inherit。"
                        f"该值将被忽略。"
                    )
                    model = None  # 忽略无效值
            init_kwargs["model"] = model

        if "level" in frontmatter:
            level = frontmatter.get("level")
            if level is not None and not isinstance(level, str):
                level = str(level)
            if level is not None:
                level = level.strip().upper()
                if not level:
                    level = None
            if level is not None and level not in ("LOW", "MID", "HIGH"):
                raise ValueError(f"Invalid level '{level}', expected one of: LOW/MID/HIGH")
            init_kwargs["level"] = level

        return cls(
            name=frontmatter["name"],
            description=frontmatter["description"],
            prompt=prompt,
            tools=tools,
            skills=skills,
            **init_kwargs,
            timeout=frontmatter.get("timeout"),
            max_iterations=frontmatter.get("max_iterations", 50),
        )

    @classmethod
    def from_file(cls, path: Path) -> "AgentDefinition":
        """从文件加载 AgentDefinition

        Args:
            path: Markdown 文件路径

        Returns:
            解析后的 AgentDefinition
        """
        content = path.read_text(encoding="utf-8")
        try:
            return cls.from_markdown(content)
        except Exception as e:
            logging.error(f"Failed to parse {path}: {e}")
            raise


@dataclass
class SubagentResult:
    """Subagent 执行结果"""

    name: str  # Subagent 名称
    success: bool  # 是否成功
    result: str | None = None  # 成功时的结果
    error: str | None = None  # 失败时的错误信息
    usage: UsageSummary | None = None  # Token 使用情况
