"""Shared types for formatters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from comate_agent_sdk.context.truncation import TruncationRecord


@dataclass
class ToolExecutionMeta:
    """工具执行元数据（存储到 ContextItem.metadata）"""

    tool_name: str
    tool_call_id: str
    status: Literal["ok", "error"]
    truncation: TruncationRecord | None = None
    file_ops: dict[str, Any] | None = None
    retrieval_hints: list[dict[str, Any]] | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    error_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "status": self.status,
            "truncation": self.truncation.to_dict() if self.truncation else None,
            "file_ops": self.file_ops,
            "retrieval_hints": self.retrieval_hints,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
            "error_field": self.error_field,
        }
        return {k: v for k, v in payload.items() if v is not None}


@dataclass
class FormattedToolResult:
    """格式化后的工具结果"""

    text: str
    meta: ToolExecutionMeta
