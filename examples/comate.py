"""Strong terminal agent entrypoint.

该入口仅负责启动 `examples/terminal_agent/app.py`。
为避免系统环境中同名第三方包覆盖，这里会优先注入当前 examples 目录到 `sys.path`。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _ensure_local_examples_on_path() -> None:
    examples_dir = Path(__file__).resolve().parent
    examples_dir_str = str(examples_dir)
    if examples_dir_str not in sys.path:
        sys.path.insert(0, examples_dir_str)


def main() -> None:
    _ensure_local_examples_on_path()
    from terminal_agent.app import run

    asyncio.run(run())


if __name__ == "__main__":
    main()
