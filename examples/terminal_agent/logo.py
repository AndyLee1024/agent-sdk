from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from rich.console import Console
from rich.text import Text

_LOGO_LINES: tuple[str, ...] = (
    "   ____   ___  __  __    _    _____ _____ ",
    "  / ___| / _ \\|  \\/  |  / \\  |_   _| ____|",
    " | |    | | | | |\\/| | / _ \\   | | |  _|  ",
    " | |___ | |_| | |  | |/ ___ \\  | | | |___ ",
    "  \\____| \\___/|_|  |_/_/   \\_\\ |_| |_____|",
)


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


def _resolve_version() -> str:
    try:
        return f"v{version('comate-agent-sdk')}"
    except PackageNotFoundError:
        return "v0.0.1-dev"


def print_logo(console: Console) -> None:
    start_rgb = (67, 114, 240)
    end_rgb = (54, 214, 220)
    line_count = len(_LOGO_LINES)

    logo_text = Text()
    for idx, line in enumerate(_LOGO_LINES):
        ratio = idx / max(line_count - 1, 1)
        r, g, b = _lerp_rgb(start_rgb, end_rgb, ratio)
        logo_text.append(line, style=f"bold rgb({r},{g},{b})")
        if idx < line_count - 1:
            logo_text.append("\n")

    console.print(logo_text)
    console.print(
        Text(f"  {_resolve_version()}  Product-grade terminal agent UI", style="dim")
    )
    console.print()
