"""上下文文件系统

负责将上下文条目持久化到文件系统，避免占用 token。

除通用的 ContextItem.offload() 外，还提供面向 tool_call_id 的落盘：
- tool_call：函数名 + arguments
- tool_result：工具输出（可按 tool_call_id 命名，便于块级压缩与回看）
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from comate_agent_sdk.context.redaction import redact_json_text, redact_text

if TYPE_CHECKING:
    from comate_agent_sdk.context.items import ContextItem

logger = logging.getLogger("comate_agent_sdk.context.fs")


@dataclass
class OffloadedMeta:
    """卸载元数据

    Attributes:
        item_id: 条目 ID
        item_type: 条目类型
        file_path: 文件路径（相对于 root_path）
        original_token_count: 原始 token 数
        offloaded_at: 卸载时间戳
        metadata: 附加元数据
    """

    item_id: str
    item_type: str
    file_path: str
    original_token_count: int
    offloaded_at: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "file_path": self.file_path,
            "original_token_count": self.original_token_count,
            "offloaded_at": self.offloaded_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OffloadedMeta":
        """从字典创建"""
        return cls(
            item_id=data["item_id"],
            item_type=data["item_type"],
            file_path=data["file_path"],
            original_token_count=data["original_token_count"],
            offloaded_at=data["offloaded_at"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ToolCallRecord:
    """tool_call_id 维度的落盘索引（用于块级占位符与追踪）"""

    tool_call_id: str
    tool_name: str | None = None
    tool_call_path: str | None = None  # 相对 root_path
    tool_result_path: str | None = None  # 相对 root_path
    assistant_item_id: str | None = None
    tool_result_item_ids: list[str] = field(default_factory=list)
    arguments_token_count: int | None = None
    result_token_count: int | None = None
    redacted: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "tool_call_path": self.tool_call_path,
            "tool_result_path": self.tool_result_path,
            "assistant_item_id": self.assistant_item_id,
            "tool_result_item_ids": list(self.tool_result_item_ids),
            "arguments_token_count": self.arguments_token_count,
            "result_token_count": self.result_token_count,
            "redacted": self.redacted,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCallRecord":
        return cls(
            tool_call_id=str(data.get("tool_call_id") or ""),
            tool_name=data.get("tool_name"),
            tool_call_path=data.get("tool_call_path"),
            tool_result_path=data.get("tool_result_path"),
            assistant_item_id=data.get("assistant_item_id"),
            tool_result_item_ids=list(data.get("tool_result_item_ids") or []),
            arguments_token_count=data.get("arguments_token_count"),
            result_token_count=data.get("result_token_count"),
            redacted=bool(data.get("redacted", False)),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass(frozen=True)
class OffloadWriteResult:
    relative_path: str
    redacted: bool


def _safe_component(value: str) -> str:
    """将任意字符串转换为安全的路径片段。"""
    v = (value or "").strip()
    if not v:
        return "_"
    out: list[str] = []
    for ch in v:
        if ch.isalnum() or ch in ("-", "_", ".", "@"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


@dataclass
class ContextFileSystem:
    """上下文文件系统

    Attributes:
        root_path: 根目录（例如 ~/.agent/context/{session_id}）
        session_id: 会话 ID
        index: 索引字典（item_id -> OffloadedMeta）
    """

    root_path: Path
    session_id: str
    index: dict[str, OffloadedMeta] = field(default_factory=dict)
    tool_calls: dict[str, ToolCallRecord] = field(default_factory=dict)

    def __post_init__(self):
        """初始化文件系统"""
        self.root_path = Path(self.root_path).expanduser().resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

        # 加载索引
        self._load_index()

    def _load_index(self):
        """加载索引文件"""
        index_path = self.root_path / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                self.index = {
                    item_id: OffloadedMeta.from_dict(meta)
                    for item_id, meta in data.get("items", {}).items()
                }
                self.tool_calls = {
                    call_id: ToolCallRecord.from_dict(meta)
                    for call_id, meta in data.get("tool_calls", {}).items()
                }
                logger.debug(f"Loaded {len(self.index)} items from index")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self.index = {}
                self.tool_calls = {}

    def _save_index(self):
        """保存索引文件"""
        index_path = self.root_path / "index.json"
        data = {
            "version": "1.0",
            "session_id": self.session_id,
            "created_at": time.time(),
            "items": {item_id: meta.to_dict() for item_id, meta in self.index.items()},
            "tool_calls": {call_id: meta.to_dict() for call_id, meta in self.tool_calls.items()},
        }
        try:
            index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def offload(self, item: "ContextItem") -> str:
        """卸载内容到文件系统

        Args:
            item: 要卸载的条目

        Returns:
            相对路径（相对于 root_path）
        """
        # 构建文件路径
        item_type_dir = self.root_path / item.item_type.value
        item_type_dir.mkdir(parents=True, exist_ok=True)

        # 如果有 tool_name，在类型目录下再创建工具子目录
        if item.tool_name:
            tool_dir = item_type_dir / item.tool_name
            tool_dir.mkdir(parents=True, exist_ok=True)
            file_path = tool_dir / f"{item.id}.json"
        else:
            file_path = item_type_dir / f"{item.id}.json"

        # 构建卸载数据
        offload_data = {
            "schema_version": "1.0",
            "item_id": item.id,
            "item_type": item.item_type.value,
            "offloaded_at": time.time(),
            "metadata": item.metadata,
            "content": item.content_text,
        }

        # 写入文件
        try:
            file_path.write_text(json.dumps(offload_data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to offload item {item.id}: {e}")
            raise

        # 更新索引
        relative_path = str(file_path.relative_to(self.root_path))
        meta = OffloadedMeta(
            item_id=item.id,
            item_type=item.item_type.value,
            file_path=relative_path,
            original_token_count=item.token_count,
            offloaded_at=time.time(),
            metadata=item.metadata,
        )
        self.index[item.id] = meta
        self._save_index()

        logger.debug(
            f"Offloaded item {item.id} ({item.item_type.value}) "
            f"to {relative_path} (~{item.token_count} tokens)"
        )

        return relative_path

    def load(self, item_id: str) -> str | None:
        """加载已卸载的内容

        Args:
            item_id: 条目 ID

        Returns:
            内容文本，如果不存在返回 None
        """
        meta = self.index.get(item_id)
        if not meta:
            return None

        file_path = self.root_path / meta.file_path
        if not file_path.exists():
            logger.warning(f"Offloaded file not found: {file_path}")
            return None

        try:
            data = json.loads(file_path.read_text())
            return data.get("content")
        except Exception as e:
            logger.error(f"Failed to load item {item_id}: {e}")
            return None

    def get_placeholder(self, item: "ContextItem") -> str:
        """生成占位符消息

        Args:
            item: 已卸载的条目

        Returns:
            占位符文本
        """
        if not item.offloaded or not item.offload_path:
            return item.content_text

        # 构建绝对路径供用户查看
        abs_path = self.root_path / item.offload_path

        return (
            f"[Content offloaded (~{item.token_count} tokens)]\n"
            f"Path: {abs_path}\n"
            f"Use Read tool to view details."
        )

    def offload_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: str,
        assistant_item_id: str | None = None,
        arguments_token_count: int | None = None,
    ) -> OffloadWriteResult:
        """将 tool_call（函数名 + arguments）落盘到 tool_call/ 目录。"""
        safe_tool = _safe_component(tool_name)
        safe_id = _safe_component(tool_call_id)
        dir_path = self.root_path / "tool_call" / safe_tool
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{safe_id}.json"

        rr = redact_json_text(arguments)
        payload = {
            "schema_version": "1.0",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "assistant_item_id": assistant_item_id,
            "offloaded_at": time.time(),
            "redacted": rr.redacted,
            "arguments": rr.text,
        }

        file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        rel = str(file_path.relative_to(self.root_path))

        rec = self.tool_calls.get(tool_call_id) or ToolCallRecord(tool_call_id=tool_call_id)
        rec.tool_name = tool_name
        rec.tool_call_path = rel
        rec.assistant_item_id = assistant_item_id or rec.assistant_item_id
        if arguments_token_count is not None:
            rec.arguments_token_count = int(arguments_token_count)
        rec.redacted = bool(rec.redacted or rr.redacted)
        self.tool_calls[tool_call_id] = rec
        self._save_index()

        return OffloadWriteResult(relative_path=rel, redacted=rr.redacted)

    def offload_tool_result(
        self,
        item: "ContextItem",
        *,
        tool_call_id: str,
        tool_name: str,
        result_token_count: int | None = None,
    ) -> OffloadWriteResult:
        """将 tool_result（工具输出）按 tool_call_id 命名落盘到 tool_result/ 目录。"""
        safe_tool = _safe_component(tool_name)
        safe_id = _safe_component(tool_call_id or item.id)
        dir_path = self.root_path / "tool_result" / safe_tool
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{safe_id}.json"

        rr = redact_text(item.content_text)
        payload = {
            "schema_version": "1.0",
            "item_id": item.id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "offloaded_at": time.time(),
            "redacted": rr.redacted,
            "metadata": item.metadata,
            "content": rr.text,
        }
        file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        relative_path = str(file_path.relative_to(self.root_path))

        # 更新 item_id 索引（保持 load(item_id) 可用）
        meta = OffloadedMeta(
            item_id=item.id,
            item_type=item.item_type.value,
            file_path=relative_path,
            original_token_count=item.token_count,
            offloaded_at=time.time(),
            metadata=item.metadata,
        )
        self.index[item.id] = meta

        # 更新 tool_call_id 索引
        rec = self.tool_calls.get(tool_call_id) or ToolCallRecord(tool_call_id=tool_call_id)
        rec.tool_name = tool_name
        rec.tool_result_path = relative_path
        if item.id not in rec.tool_result_item_ids:
            rec.tool_result_item_ids.append(item.id)
        if result_token_count is not None:
            rec.result_token_count = int(result_token_count)
        rec.redacted = bool(rec.redacted or rr.redacted)
        self.tool_calls[tool_call_id] = rec

        self._save_index()

        return OffloadWriteResult(relative_path=relative_path, redacted=rr.redacted)
