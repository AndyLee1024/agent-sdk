"""Strong terminal agent entrypoint."""

from terminal_agent.app import run


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())

