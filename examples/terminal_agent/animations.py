from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from enum import Enum

from rich.console import RenderableType
from rich.text import Text

from comate_agent_sdk.agent.events import StopEvent, TextEvent, ToolCallEvent, ToolResultEvent, UserQuestionEvent

DEFAULT_STATUS_PHRASES: tuple[str, ...] = (
    "Vibing...",
    "Thinking...",
    "Reasoning...",
    "Planning next move...",
    "Reading context...",
    "Connecting dots...",
    "Synthesizing signal...",
    "Spotting edge cases...",
    "Checking assumptions...",
    "Tracing dependencies...",
    "Drafting response...",
    "Polishing details...",
    "Validating flow...",
    "Cross-checking facts...",
    "Refining intent...",
    "Mapping tools...",
    "Building confidence...",
    "Stitching answer...",
    "Finalizing output...",
    "Almost there...",
)

PULSE_COLORS: tuple[str, ...] = (
    "#6B7280",
    "#9CA3AF",
    "#D1D5DB",
    "#9CA3AF",
)
PULSE_GLYPHS: tuple[str, ...] = ("◐", "◓", "◑", "◒")


def _lerp_rgb(
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, ratio))
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * clamped)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * clamped)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * clamped)
    return r, g, b


def _cyan_sweep_text(
    content: str,
    frame: int,
) -> Text:
    """Create a cyan sweep (moving highlight) effect over text."""
    text = Text()
    if not content:
        return text
    total = len(content)
    base_rgb = (95, 155, 190)
    mid_rgb = (120, 200, 235)
    high_rgb = (210, 245, 255)

    window = max(3, total // 5)
    cycle = max(total + window * 2, 16)
    center = (frame % cycle) - window

    for idx, ch in enumerate(content):
        distance = abs(idx - center)
        if distance <= window:
            glow = 1.0 - (distance / window)
            if glow >= 0.6:
                r, g, b = _lerp_rgb(mid_rgb, high_rgb, (glow - 0.6) / 0.4)
            else:
                r, g, b = _lerp_rgb(base_rgb, mid_rgb, glow / 0.6)
        else:
            r, g, b = base_rgb
        text.append(ch, style=f"bold rgb({r},{g},{b})")
    return text


class SubmissionAnimator:
    """Animated status line shown after user submits a message."""

    def __init__(
        self,
        console: object | None = None,
        phrases: Sequence[str] | None = None,
        refresh_interval: float = 0.12,
        min_phrase_seconds: float = 2.4,
        max_phrase_seconds: float = 3.0,
    ) -> None:
        del console
        self._phrases = tuple(phrases) if phrases else DEFAULT_STATUS_PHRASES
        self._refresh_interval = refresh_interval
        self._min_phrase_seconds = max(0.6, min_phrase_seconds)
        self._max_phrase_seconds = max(self._min_phrase_seconds, max_phrase_seconds)
        self._status_hint: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._frame = 0
        self._phrase_idx = 0
        self._phrase_started_at_monotonic = 0.0
        self._phrase_duration_seconds = 0.0
        self._dirty = False
        self._is_active = False

    def _compute_phrase_duration(self, phrase_idx: int) -> float:
        # Deterministic per phrase, bounded by [min, max], and never exceeds 3s by default.
        span = self._max_phrase_seconds - self._min_phrase_seconds
        if span <= 0:
            return self._max_phrase_seconds
        step = (phrase_idx * 17 + 11) % 100
        ratio = step / 100.0
        return self._min_phrase_seconds + span * ratio

    async def start(self) -> None:
        if self._task is not None:
            return
        self._frame = 0
        self._phrase_idx = 0
        self._phrase_started_at_monotonic = time.monotonic()
        self._phrase_duration_seconds = self._phrase_duration_for_idx(0)
        self._is_active = True
        self._dirty = True
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="submission-animator")

    def set_status_hint(self, hint: str | None) -> None:
        if hint is None:
            if self._status_hint is not None:
                self._dirty = True
            self._status_hint = None
            return
        normalized = hint.strip()
        new_value = normalized or None
        if new_value != self._status_hint:
            self._dirty = True
        self._status_hint = new_value

    async def stop(self) -> None:
        if self._task is None:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            self._stop_event = None
            self._is_active = False
            self._dirty = True

    @property
    def is_active(self) -> bool:
        return self._is_active

    def consume_dirty(self) -> bool:
        dirty = self._dirty
        self._dirty = False
        return dirty

    def renderable(self) -> RenderableType:
        if not self._is_active:
            return Text("")

        pulse_idx = self._frame % len(PULSE_COLORS)
        phrase = self._status_hint if self._status_hint else self._phrases[self._phrase_idx]
        dot = Text(
            f"{PULSE_GLYPHS[pulse_idx]} ",
            style=f"bold {PULSE_COLORS[pulse_idx]}",
        )
        sweep = _cyan_sweep_text(phrase, frame=self._frame)
        return Text.assemble(dot, sweep)

    async def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - self._phrase_started_at_monotonic >= self._phrase_duration_seconds:
                self._phrase_idx = (self._phrase_idx + 1) % len(self._phrases)
                self._phrase_started_at_monotonic = now
                self._phrase_duration_seconds = self._phrase_duration_for_idx(self._phrase_idx)
            self._frame += 1
            self._dirty = True
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._refresh_interval)
            except TimeoutError:
                continue

    def _phrase_duration_for_idx(self, phrase_idx: int) -> float:
        return self._compute_phrase_duration(phrase_idx)


class AnimationPhase(str, Enum):
    IDLE = "idle"
    SUBMITTING = "submitting"
    TOOL_RUNNING = "tool_running"
    ASSISTANT_STREAMING = "assistant_streaming"
    DONE = "done"


class StreamAnimationController:
    """Controls submission animation lifecycle across stream events."""

    def __init__(
        self,
        animator: SubmissionAnimator,
        *,
        min_visible_seconds: float = 0.35,
    ) -> None:
        self._animator = animator
        self._min_visible_seconds = max(0.0, float(min_visible_seconds))
        self._phase = AnimationPhase.IDLE
        self._started_at_monotonic = 0.0
        self._stopped = True
        self._active_tool_call_ids: set[str] = set()

    @property
    def phase(self) -> AnimationPhase:
        return self._phase

    async def start(self) -> None:
        self._active_tool_call_ids.clear()
        self._animator.set_status_hint(None)
        self._started_at_monotonic = time.monotonic()
        self._phase = AnimationPhase.SUBMITTING
        self._stopped = False
        await self._animator.start()

    async def shutdown(self) -> None:
        await self._stop_if_needed(AnimationPhase.DONE)

    async def on_event(self, event: object) -> None:
        if self._stopped:
            return

        if isinstance(event, ToolCallEvent):
            self._active_tool_call_ids.add(event.tool_call_id)
            await self._stop_if_needed(AnimationPhase.TOOL_RUNNING)
            return

        if isinstance(event, ToolResultEvent):
            self._active_tool_call_ids.discard(event.tool_call_id)
            return

        if isinstance(event, TextEvent):
            await self._stop_if_needed(AnimationPhase.ASSISTANT_STREAMING)
            return

        if isinstance(event, UserQuestionEvent) or isinstance(event, StopEvent):
            await self._stop_if_needed(AnimationPhase.DONE)

    async def _stop_if_needed(self, next_phase: AnimationPhase) -> None:
        if self._stopped:
            return
        elapsed = time.monotonic() - self._started_at_monotonic
        if elapsed < self._min_visible_seconds:
            await asyncio.sleep(self._min_visible_seconds - elapsed)
        self._animator.set_status_hint(None)
        await self._animator.stop()
        self._stopped = True
        self._phase = next_phase
