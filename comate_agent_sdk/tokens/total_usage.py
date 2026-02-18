"""TotalUsageSnapshot: session 级 token 使用累计（单调递增，不随 compaction/rollback 丢失）。

与 context.jsonl 的 usage_delta 不同：
- context.jsonl 记录 context 状态，usage_reset 时 in-context usage 重置；
- total_usage.json 只追加，是整个会话的"计费账本"。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.tokens.views import TokenUsageEntry

logger = logging.getLogger("comate_agent_sdk.tokens.total_usage")

_COMPACTION_SOURCE_KEYWORD = "compaction"


@dataclass
class LevelUsage:
    """单一档位/来源的 token 累计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    invocations: int = 0

    def _add(self, entry: "TokenUsageEntry") -> None:
        usage = entry.usage
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.cached_tokens += usage.prompt_cached_tokens or 0
        self.invocations += 1

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "invocations": self.invocations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LevelUsage":
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            cached_tokens=int(data.get("cached_tokens", 0)),
            invocations=int(data.get("invocations", 0)),
        )


@dataclass
class TotalUsageSnapshot:
    """Session 级 token 使用累计快照。

    Attributes:
        session_id: 对应的会话 ID。
        by_level: 按档位（"LOW"/"MID"/"HIGH"/其他）分组的 token 累计（不含 compaction）。
        compacted_usage: compaction LLM 调用自身消耗的 token（单独记账）。
        grand_total: 全局总计（含 compaction + by_level 所有）。
    """

    session_id: str
    by_level: dict[str, LevelUsage] = field(default_factory=dict)
    compacted_usage: LevelUsage = field(default_factory=LevelUsage)
    grand_total: LevelUsage = field(default_factory=LevelUsage)

    def _is_compaction_entry(self, entry: "TokenUsageEntry") -> bool:
        source = entry.source or ""
        return _COMPACTION_SOURCE_KEYWORD in source

    def add_turn_entries(self, entries: list["TokenUsageEntry"]) -> None:
        """处理本 turn 产生的所有 usage entries。

        compaction 来源的 entry 同时计入 compacted_usage 和 grand_total；
        非 compaction 来源的 entry 按 level 分组计入 by_level 和 grand_total。
        """
        for entry in entries:
            self.grand_total._add(entry)
            if self._is_compaction_entry(entry):
                self.compacted_usage._add(entry)
            else:
                level_key = entry.level or "UNKNOWN"
                if level_key not in self.by_level:
                    self.by_level[level_key] = LevelUsage()
                self.by_level[level_key]._add(entry)

    def add_compaction_entry(self, entry: "TokenUsageEntry") -> None:
        """显式添加一条 compaction entry（计入 compacted_usage 和 grand_total）。"""
        self.compacted_usage._add(entry)
        self.grand_total._add(entry)

    def save(self, path: Path) -> None:
        """持久化到 JSON 文件（atomic write via tmp+rename 不适用简单场景，直接写入）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "by_level": {k: v.to_dict() for k, v in self.by_level.items()},
            "compacted_usage": self.compacted_usage.to_dict(),
            "grand_total": self.grand_total.to_dict(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug(f"total_usage saved: session={self.session_id}, grand_total={self.grand_total.total_tokens}")

    @classmethod
    def load(cls, path: Path) -> "TotalUsageSnapshot":
        """从 JSON 文件恢复。文件不存在时抛 FileNotFoundError。"""
        data = json.loads(path.read_text(encoding="utf-8"))
        by_level = {k: LevelUsage.from_dict(v) for k, v in (data.get("by_level") or {}).items()}
        return cls(
            session_id=str(data.get("session_id", "")),
            by_level=by_level,
            compacted_usage=LevelUsage.from_dict(data.get("compacted_usage") or {}),
            grand_total=LevelUsage.from_dict(data.get("grand_total") or {}),
        )

    @classmethod
    def new(cls, session_id: str) -> "TotalUsageSnapshot":
        """创建一个空快照。"""
        return cls(session_id=session_id)
