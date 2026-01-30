"""上下文文件系统

负责将上下文条目持久化到文件系统，避免占用 token。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bu_agent_sdk.context.items import ContextItem

logger = logging.getLogger("bu_agent_sdk.context.fs")


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
                logger.debug(f"Loaded {len(self.index)} items from index")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self.index = {}

    def _save_index(self):
        """保存索引文件"""
        index_path = self.root_path / "index.json"
        data = {
            "version": "1.0",
            "session_id": self.session_id,
            "created_at": time.time(),
            "items": {item_id: meta.to_dict() for item_id, meta in self.index.items()},
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
