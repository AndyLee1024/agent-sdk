from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from comate_agent_sdk.agent import ChatSession
from comate_agent_sdk.context.items import ItemType

logger = logging.getLogger(__name__)

_TRACKED_TOOLS = {"Write", "Edit", "MultiEdit"}
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RewindCheckpoint:
    checkpoint_id: int
    turn_number: int
    user_preview: str
    user_message: str
    created_at: str
    touched_files: tuple[str, ...]


@dataclass(frozen=True)
class RewindPlanFile:
    relpath: str
    action: Literal["write", "delete", "skip_binary", "skip_unknown"]
    added_lines: int = 0
    removed_lines: int = 0
    note: str | None = None


@dataclass(frozen=True)
class RewindRestorePlan:
    checkpoint: RewindCheckpoint
    files: tuple[RewindPlanFile, ...]
    total_added_lines: int
    total_removed_lines: int
    writable_files_count: int
    skipped_binary_count: int
    skipped_unknown_count: int


class RewindStore:
    def __init__(self, *, session: ChatSession, project_root: Path) -> None:
        self._session = session
        self._project_root = project_root.resolve()

    def bind_session(self, session: ChatSession) -> None:
        self._session = session

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def storage_root(self) -> Path:
        return Path(self._session._storage_root)

    @property
    def rewind_root(self) -> Path:
        return self.storage_root / "rewind"

    @property
    def checkpoint_root(self) -> Path:
        return self.rewind_root / "checkpoints"

    @property
    def blob_root(self) -> Path:
        return self.rewind_root / "blobs"

    @property
    def index_path(self) -> Path:
        return self.rewind_root / "index.json"

    def list_checkpoints(self) -> list[RewindCheckpoint]:
        index = self._load_index()
        checkpoints = [self._checkpoint_from_dict(cp) for cp in index.get("checkpoints", [])]
        checkpoints.sort(key=lambda cp: cp.checkpoint_id)
        return checkpoints

    def prune_after_checkpoint(self, *, checkpoint_id: int) -> int:
        """删除指定 checkpoint 之后的所有 checkpoint。"""
        index = self._load_index()
        raw_checkpoints = index.get("checkpoints", [])
        if not raw_checkpoints:
            return 0

        kept: list[dict[str, Any]] = []
        dropped_ids: list[int] = []
        for cp in raw_checkpoints:
            cp_id = int(cp.get("id", 0))
            if cp_id <= checkpoint_id:
                kept.append(cp)
            else:
                dropped_ids.append(cp_id)

        if not dropped_ids:
            return 0

        index["checkpoints"] = kept
        next_id = (max((int(cp.get("id", 0)) for cp in kept), default=0) + 1)
        index["next_checkpoint_id"] = next_id
        self._save_index(index)

        for cp_id in dropped_ids:
            cp_path = self.checkpoint_root / f"{cp_id}.json"
            try:
                cp_path.unlink()
            except FileNotFoundError:
                pass

        return len(dropped_ids)

    def capture_checkpoint_for_latest_turn(self, *, user_preview: str) -> RewindCheckpoint | None:
        current_turn = self._current_turn()
        if current_turn <= 0:
            return None

        index = self._load_index()
        checkpoints = index.get("checkpoints", [])
        if any(int(cp.get("turn_number", 0)) == current_turn for cp in checkpoints):
            return None

        prev_manifest = self._normalize_manifest(
            checkpoints[-1].get("manifest", {}) if checkpoints else {}
        )
        touched_files, file_events = self._collect_turn_file_events(turn_number=current_turn)

        manifest: dict[str, dict[str, Any]] = {k: dict(v) for k, v in prev_manifest.items()}
        for relpath in touched_files:
            manifest[relpath] = self._capture_file_state(relpath)

        checkpoint_id = int(index.get("next_checkpoint_id", 1))
        checkpoint = {
            "id": checkpoint_id,
            "turn_number": current_turn,
            "user_preview": self._normalize_preview(user_preview),
            "user_message": str(user_preview).strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "touched_files": sorted(touched_files),
            "file_events": file_events,
            "manifest": manifest,
        }
        checkpoints.append(checkpoint)
        index["next_checkpoint_id"] = checkpoint_id + 1
        index["checkpoints"] = checkpoints
        self._save_index(index)
        self._save_checkpoint_file(checkpoint)
        return self._checkpoint_from_dict(checkpoint)

    def build_restore_plan(self, *, checkpoint_id: int) -> RewindRestorePlan:
        index = self._load_index()
        checkpoints = index.get("checkpoints", [])
        if not checkpoints:
            raise ValueError("No checkpoints available")

        by_id = {int(cp.get("id", 0)): cp for cp in checkpoints}
        target_raw = by_id.get(checkpoint_id)
        if target_raw is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        latest_manifest = self._normalize_manifest(checkpoints[-1].get("manifest", {}))
        target_manifest = self._normalize_manifest(target_raw.get("manifest", {}))
        tracked_paths = sorted(set(latest_manifest.keys()) | set(target_manifest.keys()))

        planned_files: list[RewindPlanFile] = []
        total_added = 0
        total_removed = 0
        writable_count = 0
        skipped_binary = 0
        skipped_unknown = 0

        target_id = int(target_raw.get("id", 0))
        for relpath in tracked_paths:
            current_state = self._capture_file_state(relpath)
            target_state = target_manifest.get(relpath)
            if target_state is None:
                target_state = self._infer_missing_target_state(
                    relpath=relpath,
                    checkpoints=checkpoints,
                    target_checkpoint_id=target_id,
                )
                if target_state is None:
                    if current_state.get("exists", False):
                        skipped_unknown += 1
                        planned_files.append(
                            RewindPlanFile(
                                relpath=relpath,
                                action="skip_unknown",
                                note="missing baseline before first tracked change",
                            )
                        )
                    continue

            file_plan = self._plan_file_change(
                relpath=relpath,
                current_state=current_state,
                target_state=target_state,
            )
            if file_plan is None:
                continue
            planned_files.append(file_plan)

            if file_plan.action == "write" or file_plan.action == "delete":
                writable_count += 1
                total_added += file_plan.added_lines
                total_removed += file_plan.removed_lines
            elif file_plan.action == "skip_binary":
                skipped_binary += 1
            elif file_plan.action == "skip_unknown":
                skipped_unknown += 1

        return RewindRestorePlan(
            checkpoint=self._checkpoint_from_dict(target_raw),
            files=tuple(planned_files),
            total_added_lines=total_added,
            total_removed_lines=total_removed,
            writable_files_count=writable_count,
            skipped_binary_count=skipped_binary,
            skipped_unknown_count=skipped_unknown,
        )

    def restore_code_to_checkpoint(self, *, checkpoint_id: int) -> RewindRestorePlan:
        index = self._load_index()
        checkpoints = index.get("checkpoints", [])
        by_id = {int(cp.get("id", 0)): cp for cp in checkpoints}
        target_raw = by_id.get(checkpoint_id)
        if target_raw is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        target_manifest = self._normalize_manifest(target_raw.get("manifest", {}))

        plan = self.build_restore_plan(checkpoint_id=checkpoint_id)
        for file_plan in plan.files:
            relpath = file_plan.relpath
            abs_path = (self._project_root / relpath).resolve()
            if file_plan.action == "delete":
                if abs_path.exists() and abs_path.is_file():
                    abs_path.unlink()
                continue
            if file_plan.action == "write":
                target_state = target_manifest.get(relpath) or {}
                sha256 = str(target_state.get("sha256") or "").strip()
                if not sha256:
                    continue
                blob_path = self._blob_path(sha256)
                if not blob_path.exists():
                    logger.warning(f"rewind blob missing for {relpath}: {sha256}")
                    continue
                data = blob_path.read_bytes()
                self._atomic_write_bytes(abs_path, data)
                continue
        return plan

    def _plan_file_change(
        self,
        *,
        relpath: str,
        current_state: dict[str, Any],
        target_state: dict[str, Any],
    ) -> RewindPlanFile | None:
        current_exists = bool(current_state.get("exists", False))
        target_exists = bool(target_state.get("exists", False))

        if target_exists and bool(target_state.get("binary", False)):
            if current_exists:
                return RewindPlanFile(
                    relpath=relpath,
                    action="skip_binary",
                    note="target is binary; skipped",
                )
            return None

        if not target_exists:
            if not current_exists:
                return None
            if bool(current_state.get("binary", False)):
                return RewindPlanFile(
                    relpath=relpath,
                    action="skip_binary",
                    note="current is binary; skipped delete",
                )
            before_bytes = self._read_blob_for_state(current_state)
            added, removed = self._compute_line_delta(before=before_bytes, after=b"")
            return RewindPlanFile(
                relpath=relpath,
                action="delete",
                added_lines=added,
                removed_lines=removed,
            )

        if bool(current_state.get("binary", False)):
            return RewindPlanFile(
                relpath=relpath,
                action="skip_binary",
                note="current is binary; skipped",
            )

        target_bytes = self._read_blob_for_state(target_state)
        if target_bytes is None:
            return RewindPlanFile(
                relpath=relpath,
                action="skip_unknown",
                note="target blob missing",
            )

        if not current_exists:
            added, removed = self._compute_line_delta(before=b"", after=target_bytes)
            return RewindPlanFile(
                relpath=relpath,
                action="write",
                added_lines=added,
                removed_lines=removed,
            )

        current_bytes = self._read_blob_for_state(current_state)
        if current_bytes is None:
            return RewindPlanFile(
                relpath=relpath,
                action="skip_unknown",
                note="current blob missing",
            )
        if current_bytes == target_bytes:
            return None

        added, removed = self._compute_line_delta(before=current_bytes, after=target_bytes)
        return RewindPlanFile(
            relpath=relpath,
            action="write",
            added_lines=added,
            removed_lines=removed,
        )

    @staticmethod
    def _compute_line_delta(*, before: bytes, after: bytes) -> tuple[int, int]:
        before_text = before.decode("utf-8", errors="replace").splitlines()
        after_text = after.decode("utf-8", errors="replace").splitlines()
        added = 0
        removed = 0
        for line in difflib.unified_diff(before_text, after_text, lineterm=""):
            if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        return added, removed

    def _read_blob_for_state(self, state: dict[str, Any]) -> bytes | None:
        if not bool(state.get("exists", False)):
            return b""
        sha256 = str(state.get("sha256") or "").strip()
        if not sha256:
            return None
        blob_path = self._blob_path(sha256)
        if not blob_path.exists():
            return None
        return blob_path.read_bytes()

    def _infer_missing_target_state(
        self,
        *,
        relpath: str,
        checkpoints: list[dict[str, Any]],
        target_checkpoint_id: int,
    ) -> dict[str, Any] | None:
        first_seen: dict[str, Any] | None = None
        for cp in checkpoints:
            manifest = self._normalize_manifest(cp.get("manifest", {}))
            if relpath in manifest:
                first_seen = cp
                break
        if first_seen is None:
            return None

        first_id = int(first_seen.get("id", 0))
        if first_id <= target_checkpoint_id:
            return None

        events = first_seen.get("file_events", {}) or {}
        event_meta = events.get(relpath, {}) if isinstance(events, dict) else {}
        created = event_meta.get("created")
        if created is True:
            return {"exists": False, "binary": False, "sha256": None}
        return None

    def _collect_turn_file_events(
        self,
        *,
        turn_number: int,
    ) -> tuple[set[str], dict[str, dict[str, Any]]]:
        touched: set[str] = set()
        events: dict[str, dict[str, Any]] = {}
        items = getattr(self._session._agent._context.conversation, "items", [])
        for item in items:
            if item.item_type != ItemType.TOOL_RESULT:
                continue
            if int(getattr(item, "created_turn", 0) or 0) != turn_number:
                continue
            if bool(getattr(item, "is_tool_error", False)):
                continue

            tool_name = str(item.tool_name or "").strip()
            if not tool_name and getattr(item, "message", None) is not None:
                tool_name = str(getattr(item.message, "tool_name", "")).strip()
            if tool_name not in _TRACKED_TOOLS:
                continue

            envelope = (item.metadata or {}).get("tool_raw_envelope")
            if not isinstance(envelope, dict):
                continue

            meta = envelope.get("meta", {})
            data = envelope.get("data", {})
            if not isinstance(meta, dict):
                meta = {}
            if not isinstance(data, dict):
                data = {}

            raw_path = data.get("relpath") or meta.get("file_path")
            relpath = self._normalize_relpath(raw_path)
            if relpath is None:
                continue
            touched.add(relpath)

            created_val = data.get("created")
            created: bool | None
            if isinstance(created_val, bool):
                created = created_val
            else:
                created = None
            operation = str(meta.get("operation") or "").strip() or None
            events[relpath] = {
                "tool": tool_name,
                "operation": operation,
                "created": created,
            }
        return touched, events

    @staticmethod
    def _normalize_preview(text: str) -> str:
        line = " ".join(str(text).strip().split())
        if len(line) <= 80:
            return line
        return f"{line[:77]}..."

    def _current_turn(self) -> int:
        try:
            return int(getattr(self._session._agent._context, "_turn_number", 0) or 0)
        except Exception:
            return 0

    def _capture_file_state(self, relpath: str) -> dict[str, Any]:
        abs_path = (self._project_root / relpath).resolve()
        if not self._is_under(abs_path, self._project_root):
            return {"exists": False, "binary": False, "sha256": None}
        if not abs_path.exists() or not abs_path.is_file():
            return {"exists": False, "binary": False, "sha256": None}

        data = abs_path.read_bytes()
        if self._is_binary_content(data):
            return {"exists": True, "binary": True, "sha256": None}

        sha256 = hashlib.sha256(data).hexdigest()
        self._store_blob(sha256, data)
        return {"exists": True, "binary": False, "sha256": sha256}

    def _save_checkpoint_file(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        cp_id = int(checkpoint.get("id", 0))
        path = self.checkpoint_root / f"{cp_id}.json"
        path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _blob_path(self, sha256: str) -> Path:
        return self.blob_root / f"{sha256}.bin"

    def _store_blob(self, sha256: str, data: bytes) -> None:
        self.blob_root.mkdir(parents=True, exist_ok=True)
        path = self._blob_path(sha256)
        if path.exists():
            return
        self._atomic_write_bytes(path, data)

    @staticmethod
    def _is_binary_content(data: bytes) -> bool:
        if b"\x00" in data[:4096]:
            return True
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False

    @staticmethod
    def _checkpoint_from_dict(data: dict[str, Any]) -> RewindCheckpoint:
        touched = data.get("touched_files", []) or []
        user_message = str(data.get("user_message", "")).strip()
        user_preview = str(data.get("user_preview", ""))
        if not user_message:
            user_message = user_preview
        return RewindCheckpoint(
            checkpoint_id=int(data.get("id", 0)),
            turn_number=int(data.get("turn_number", 0)),
            user_preview=user_preview,
            user_message=user_message,
            created_at=str(data.get("created_at", "")),
            touched_files=tuple(str(x) for x in touched),
        )

    @staticmethod
    def _normalize_manifest(raw_manifest: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw_manifest, dict):
            return {}
        manifest: dict[str, dict[str, Any]] = {}
        for key, raw in raw_manifest.items():
            relpath = str(key).replace("\\", "/").strip("/")
            if not relpath:
                continue
            state = raw if isinstance(raw, dict) else {}
            manifest[relpath] = {
                "exists": bool(state.get("exists", False)),
                "binary": bool(state.get("binary", False)),
                "sha256": state.get("sha256"),
            }
        return manifest

    def _normalize_relpath(self, raw_path: Any) -> str | None:
        if raw_path is None:
            return None
        text = str(raw_path).strip()
        if not text:
            return None
        path = Path(text)
        abs_path = path.resolve() if path.is_absolute() else (self._project_root / path).resolve()
        if not self._is_under(abs_path, self._project_root):
            return None
        return abs_path.relative_to(self._project_root).as_posix()

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except Exception:
            return False

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "schema_version": _SCHEMA_VERSION,
                "session_id": self._session.session_id,
                "next_checkpoint_id": 1,
                "checkpoints": [],
            }
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(f"failed to load rewind index: {self.index_path}")
            return {
                "schema_version": _SCHEMA_VERSION,
                "session_id": self._session.session_id,
                "next_checkpoint_id": 1,
                "checkpoints": [],
            }

        if not isinstance(data, dict):
            return {
                "schema_version": _SCHEMA_VERSION,
                "session_id": self._session.session_id,
                "next_checkpoint_id": 1,
                "checkpoints": [],
            }

        checkpoints = data.get("checkpoints")
        if not isinstance(checkpoints, list):
            checkpoints = []
        data["checkpoints"] = checkpoints
        data["next_checkpoint_id"] = int(data.get("next_checkpoint_id", len(checkpoints) + 1))
        data["schema_version"] = _SCHEMA_VERSION
        data["session_id"] = self._session.session_id
        return data

    def _save_index(self, index: dict[str, Any]) -> None:
        self.rewind_root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8")
        self._atomic_write_bytes(self.index_path, payload)

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            try:
                os.remove(tmp_name)
            except FileNotFoundError:
                pass
