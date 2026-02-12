from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class SessionRunController:
    """Session-scoped controller for cooperative interruption."""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _interrupted: bool = field(default=False, init=False, repr=False)
    _reason: str = field(default="", init=False, repr=False)
    _interrupt_count: int = field(default=0, init=False, repr=False)

    def interrupt(self, reason: str = "user") -> None:
        normalized_reason = reason.strip() if isinstance(reason, str) else "user"
        if not normalized_reason:
            normalized_reason = "user"
        with self._lock:
            self._interrupted = True
            self._reason = normalized_reason
            self._interrupt_count += 1

    def clear(self) -> None:
        with self._lock:
            self._interrupted = False
            self._reason = ""
            self._interrupt_count = 0

    @property
    def is_interrupted(self) -> bool:
        with self._lock:
            return self._interrupted

    @property
    def reason(self) -> str | None:
        with self._lock:
            if not self._interrupted:
                return None
            return self._reason or "user"

    @property
    def interrupt_count(self) -> int:
        with self._lock:
            return self._interrupt_count
