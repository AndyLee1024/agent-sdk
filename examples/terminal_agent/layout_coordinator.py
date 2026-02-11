from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.segment import Segment, Segments
from rich.text import Text

_TODO_MIN_LINES = 6


def _is_empty_text(renderable: RenderableType | None) -> bool:
    return isinstance(renderable, Text) and not renderable.plain


def _renderable_to_lines(console: Console, renderable: RenderableType | None) -> list[list[Segment]]:
    if renderable is None:
        return []
    if _is_empty_text(renderable):
        return []
    return console.render_lines(renderable, console.options, pad=False, new_lines=False)


def _lines_to_renderable(lines: list[list[Segment]]) -> RenderableType | None:
    if not lines:
        return None
    segments: list[Segment] = []
    last_index = len(lines) - 1
    for idx, line in enumerate(lines):
        segments.extend(line)
        if idx != last_index:
            segments.append(Segment.line())
    return Segments(segments, new_lines=False)


def _total_height(*layers: list[list[Segment]]) -> int:
    non_empty = [layer for layer in layers if layer]
    if not non_empty:
        return 0
    return sum(len(layer) for layer in non_empty) + max(0, len(non_empty) - 1)


class TerminalLayoutCoordinator:
    """Single Live owner for loading/message/todo layers."""

    def __init__(
        self,
        console: Console,
        *,
        target_fps: int = 12,
    ) -> None:
        normalized_fps = max(4, min(int(target_fps), 24))
        self._console = console
        self._target_fps = normalized_fps
        self._turn_active = False
        self._live: Live | None = None
        self._dirty = False
        self._loading_layer: RenderableType | None = None
        self._message_layer: RenderableType | None = Text("")
        self._todo_layer: RenderableType | None = None

    def start_turn(self) -> None:
        self._turn_active = True
        self._dirty = True
        self._ensure_live()
        self.refresh(force=True)

    def stop_turn(self) -> None:
        if not self._turn_active:
            return
        self.refresh(force=True)
        self._stop_live()
        self._turn_active = False

    def close(self) -> None:
        self._stop_live()
        self._turn_active = False

    def update_layers(
        self,
        *,
        loading: RenderableType | None,
        message: RenderableType | None,
        todo: RenderableType | None,
    ) -> None:
        self._loading_layer = loading
        self._message_layer = message
        self._todo_layer = todo
        self._dirty = True

    def refresh(self, *, force: bool = False) -> None:
        if not self._turn_active:
            return
        if not force and not self._dirty:
            return

        self._ensure_live()
        if self._live is None:
            return
        self._live.update(self._compose(), refresh=True)
        self._dirty = False

    def _ensure_live(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            Text(""),
            console=self._console,
            transient=False,
            refresh_per_second=self._target_fps,
            vertical_overflow="crop",
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()

    def _stop_live(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None

    def _compose(self) -> RenderableType:
        message_lines = _renderable_to_lines(self._console, self._message_layer)
        todo_lines = _renderable_to_lines(self._console, self._todo_layer)
        loading_lines = _renderable_to_lines(self._console, self._loading_layer)

        max_height = max(int(self._console.size.height), 1)
        while True:
            total = _total_height(message_lines, todo_lines, loading_lines)
            if total <= max_height:
                break

            overflow = total - max_height
            progressed = False

            if message_lines and overflow > 0:
                drop = min(overflow, len(message_lines))
                message_lines = message_lines[drop:]
                progressed = progressed or drop > 0

            total = _total_height(message_lines, todo_lines, loading_lines)
            overflow = max(0, total - max_height)
            if todo_lines and overflow > 0 and len(todo_lines) > _TODO_MIN_LINES:
                max_drop = len(todo_lines) - _TODO_MIN_LINES
                drop = min(overflow, max_drop)
                todo_lines = todo_lines[: len(todo_lines) - drop]
                progressed = progressed or drop > 0

            total = _total_height(message_lines, todo_lines, loading_lines)
            overflow = max(0, total - max_height)
            if loading_lines and overflow > 0:
                original_len = len(loading_lines)
                keep = max(original_len - overflow, 1)
                dropped = original_len - keep
                loading_lines = loading_lines[-keep:]
                progressed = progressed or dropped > 0

            total = _total_height(message_lines, todo_lines, loading_lines)
            overflow = max(0, total - max_height)
            if todo_lines and overflow > 0:
                original_len = len(todo_lines)
                keep = max(original_len - overflow, 1)
                dropped = original_len - keep
                todo_lines = todo_lines[:keep]
                progressed = progressed or dropped > 0

            if not progressed:
                break

        message = _lines_to_renderable(message_lines)
        todo = _lines_to_renderable(todo_lines)
        loading = _lines_to_renderable(loading_lines)

        parts: list[RenderableType] = []
        if message is not None:
            parts.append(message)
        if todo is not None:
            if parts:
                parts.append(Text(""))
            parts.append(todo)
        if loading is not None:
            if parts:
                parts.append(Text(""))
            parts.append(loading)

        if not parts:
            return Text("")
        return Group(*parts)
