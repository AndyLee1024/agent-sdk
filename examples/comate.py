# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "comate-agent-sdk>=0.0.1",
#     "rich>=14.0",
#     "prompt-toolkit>=3.0",
# ]
# ///

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


def _parse_args(argv: list[str]) -> tuple[bool, str | None]:
    rpc_stdio = False
    session_id: str | None = None
    for arg in argv:
        if arg == "--rpc-stdio":
            rpc_stdio = True
            continue
        if arg.startswith("-"):
            continue
        if session_id is None:
            session_id = arg
    return rpc_stdio, session_id


def main() -> None:
    _ensure_local_examples_on_path()
    from terminal_agent.app import run

    rpc_stdio, session_id = _parse_args(sys.argv[1:])
    asyncio.run(run(rpc_stdio=rpc_stdio, session_id=session_id))


if __name__ == "__main__":
    main()
