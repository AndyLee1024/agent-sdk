"""统一截断记录类型。

三层上下文管理共享：
- Layer 1 (OutputFormatter): 写入 formatter_* 字段
- Layer 2 (Ephemeral): 读取 formatter 截断信息用于销毁优先级判断
- Layer 3 (Compaction): 追加 compaction_details
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TruncationRecord:
    """统一截断记录。"""

    formatter_truncated: bool = False
    formatter_reason: str = ""
    formatter_total_estimate: int = 0
    formatter_shown_range: dict[str, int] | None = None
    compaction_details: list[dict[str, object]] = field(default_factory=list)

    @property
    def is_truncated(self) -> bool:
        return self.formatter_truncated or bool(self.compaction_details)

    @property
    def is_formatter_truncated(self) -> bool:
        return self.formatter_truncated

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "formatter_truncated": self.formatter_truncated,
        }
        if self.formatter_reason:
            result["formatter_reason"] = self.formatter_reason
        if self.formatter_total_estimate:
            result["formatter_total_estimate"] = self.formatter_total_estimate
        if self.formatter_shown_range:
            result["formatter_shown_range"] = self.formatter_shown_range
        if self.compaction_details:
            result["compaction_details"] = self.compaction_details
        return result
