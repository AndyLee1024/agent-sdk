from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_INDEX_LOCK = Lock()


def _safe_component(value: str) -> str:
    out: list[str] = []
    for ch in (value or "").strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out).strip("._")
    return text or "_"


def _mime_extension(mime: str) -> str:
    if mime == "application/json":
        return ".json"
    if mime == "text/markdown":
        return ".md"
    if mime.startswith("text/"):
        return ".txt"
    return ".bin"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        except Exception:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except FileNotFoundError:
                pass


class ArtifactStore:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()
        self.base_dir = (self.workspace_root / ".agent" / "artifacts").resolve()
        self.index_path = self.base_dir / "index.json"

    async def put_text(
        self,
        *,
        namespace: str,
        text: str,
        mime: str = "text/plain",
        ttl_seconds: int = 3600,
    ) -> dict[str, Any]:
        return await self.put_bytes(
            namespace=namespace,
            payload=text.encode("utf-8"),
            mime=mime,
            ttl_seconds=ttl_seconds,
        )

    async def put_json(
        self,
        *,
        namespace: str,
        data: Any,
        ttl_seconds: int = 3600,
    ) -> dict[str, Any]:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        return await self.put_text(
            namespace=namespace,
            text=payload,
            mime="application/json",
            ttl_seconds=ttl_seconds,
        )

    async def put_bytes(
        self,
        *,
        namespace: str,
        payload: bytes,
        mime: str,
        ttl_seconds: int = 3600,
    ) -> dict[str, Any]:
        digest = hashlib.sha256(payload).hexdigest()
        artifact_id = f"sha256:{digest}"
        safe_namespace = _safe_component(namespace)
        ext = _mime_extension(mime)
        file_name = f"{digest}{ext}"
        target = self.base_dir / safe_namespace / file_name

        _atomic_write_bytes(target, payload)

        relpath = os.path.relpath(str(target), str(self.workspace_root))
        artifact = {
            "id": artifact_id,
            "relpath": relpath,
            "bytes": len(payload),
            "sha256": digest,
            "mime": mime,
            "created_at": int(time.time()),
            "ttl_seconds": int(ttl_seconds),
        }
        await self._append_index(artifact)
        return artifact

    async def _append_index(self, artifact: dict[str, Any]) -> None:
        self._append_index_sync(artifact)

    def _append_index_sync(self, artifact: dict[str, Any]) -> None:
        with _INDEX_LOCK:
            data: dict[str, Any]
            if self.index_path.exists():
                try:
                    raw = self.index_path.read_text(encoding="utf-8")
                    data = json.loads(raw)
                except Exception as exc:
                    logger.warning(f"artifact index damaged, rebuilding: {exc}")
                    data = {}
            else:
                data = {}

            items = data.get("items")
            if not isinstance(items, dict):
                items = {}
            items[artifact["id"]] = artifact

            payload = {
                "schema_version": 1,
                "updated_at": int(time.time()),
                "items": items,
            }
            _atomic_write_bytes(
                self.index_path,
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            )
