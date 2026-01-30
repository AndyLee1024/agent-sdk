"""卸载策略配置

定义哪些类型的 ContextItem 可以被卸载到文件系统。
"""

from dataclasses import dataclass, field


@dataclass
class OffloadPolicy:
    """卸载策略配置

    Attributes:
        enabled: 是否启用卸载功能
        token_threshold: 超过此 token 数的条目才会被卸载
        type_enabled: 各类型是否启用卸载
    """

    enabled: bool = True
    token_threshold: int = 2000  # 超过此阈值才卸载

    # 各类型是否启用卸载
    type_enabled: dict[str, bool] = field(
        default_factory=lambda: {
            "tool_result": True,
            "assistant_message": False,
            "user_message": False,
            "skill_prompt": True,
            # 以下类型不卸载
            "compaction_summary": False,  # 已压缩的摘要不卸载
            "system_prompt": False,
            "memory": False,
            "subagent_strategy": False,
            "skill_strategy": False,
        }
    )
