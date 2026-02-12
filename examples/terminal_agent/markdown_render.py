from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.markdown import Markdown


def render_markdown_to_plain(content: str, *, width: int) -> str:
    normalized_width = max(int(width), 40)
    try:
        sink = StringIO()
        console = Console(
            file=sink,
            width=normalized_width,
            force_terminal=False,
            color_system=None,
            soft_wrap=True,
        )
        console.print(Markdown(content, code_theme="monokai", hyperlinks=False))
        rendered = sink.getvalue().rstrip("\n")
        return rendered
    except Exception:
        return content
