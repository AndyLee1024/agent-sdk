"""Skill 定义模型（兼容 Claude Code SKILL.md 格式）"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillDefinition:
    """Skill 定义（兼容 Claude Code SKILL.md 格式）"""

    # 核心字段
    name: str  # 如果 frontmatter 没有 name，使用目录名
    description: str  # 如果 frontmatter 没有 description，使用 prompt 第一段
    prompt: str  # Markdown 正文

    # 可选字段（Frontmatter）
    model: str | None = None  # [已弃用] 该字段已被忽略，请使用 Subagent 代替
    argument_hint: str | None = None  # 暂时保留但不使用
    disable_model_invocation: bool = False
    user_invocable: bool = True  # 暂时保留但不使用

    # 资源路径（自动发现）
    base_dir: Path | None = None
    scripts_dir: Path | None = None
    references_dir: Path | None = None
    assets_dir: Path | None = None

    @classmethod
    def from_markdown(cls, content: str, dir_name: str, base_dir: Path | None = None) -> "SkillDefinition":
        """从 Markdown 内容解析 Skill

        Args:
            content: SKILL.md 文件内容
            dir_name: Skill 目录名（用作 name 的后备值）
            base_dir: Skill 基础目录路径
        """
        import yaml

        # 解析 frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                prompt = parts[2].strip()
            else:
                frontmatter = {}
                prompt = content
        else:
            frontmatter = {}
            prompt = content

        # 提取 name（优先 frontmatter，后备为目录名）
        name = frontmatter.get("name") or dir_name

        # 提取 description（优先 frontmatter，后备为 prompt 第一段）
        description = frontmatter.get("description")
        if not description and prompt:
            # 使用第一段作为 description
            first_paragraph = prompt.split("\n\n")[0].strip()
            description = first_paragraph[:200]  # 限制长度
            # 移除 markdown 标题标记
            if description.startswith("#"):
                description = description.lstrip("#").strip()

        if not description:
            description = f"Skill: {name}"

        return cls(
            name=name,
            description=description,
            prompt=prompt,
            model=frontmatter.get("model"),
            argument_hint=frontmatter.get("argument-hint") or frontmatter.get("argument_hint"),
            disable_model_invocation=frontmatter.get("disable-model-invocation", False)
            or frontmatter.get("disable_model_invocation", False),
            user_invocable=frontmatter.get("user-invocable", True) or frontmatter.get("user_invocable", True),
            base_dir=base_dir,
        )

    @classmethod
    def from_directory(cls, skill_dir: Path) -> "SkillDefinition":
        """从目录加载 Skill（支持资源打包）

        目录结构：
        .agent/skills/skillname/
        ├── SKILL.md  (必需)
        ├── scripts/  (可选)
        ├── references/  (可选)
        └── assets/  (可选)
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise ValueError(f"No SKILL.md found in {skill_dir}")

        content = skill_md.read_text(encoding="utf-8")
        skill = cls.from_markdown(content, dir_name=skill_dir.name, base_dir=skill_dir)

        # 发现资源目录
        if (skill_dir / "scripts").exists():
            skill.scripts_dir = skill_dir / "scripts"
        if (skill_dir / "references").exists():
            skill.references_dir = skill_dir / "references"
        if (skill_dir / "assets").exists():
            skill.assets_dir = skill_dir / "assets"

        return skill

    def get_prompt(self) -> str:
        """获取完整 prompt（替换变量）"""
        prompt = self.prompt
        if self.base_dir:
            prompt = prompt.replace("{baseDir}", str(self.base_dir))
        return prompt
