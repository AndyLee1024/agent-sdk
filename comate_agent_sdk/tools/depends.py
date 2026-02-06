"""
Dependency injection system inspired by FastAPI's Depends.

Usage:
    from typing import Annotated

    def get_db_session() -> DatabaseSession:
        return DatabaseSession()

    @tool("Query the database")
    async def query_db(sql: str, db: Annotated[DatabaseSession, Depends(get_db_session)]) -> str:
        return await db.execute(sql)

    # With dependency overrides (useful for testing or scoped contexts):
    overrides = {get_db_session: lambda: mock_db}
    await tool.execute(sql="SELECT 1", _overrides=overrides)
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

T = TypeVar('T')

# Type alias for dependency overrides
DependencyOverrides = dict[Callable[[], Any], Callable[[], Any]]


class Depends(Generic[T]):
	"""
	Dependency injection marker, similar to FastAPI's Depends.

	The dependency callable will be invoked when the tool is called,
	and its return value will be injected into the tool function.

	Supports dependency overrides for scoped injection (like FastAPI's
	app.dependency_overrides).

	Use with typing.Annotated for proper type safety:

	    from typing import Annotated

	    def get_browser() -> Browser:
	        return Browser()

	    @tool("Navigate to a URL")
	    async def navigate(url: str, browser: Annotated[Browser, Depends(get_browser)]) -> str:
	        await browser.goto(url)

	Args:
	    dependency: A callable (sync or async) that returns the dependency value.
	"""

	__slots__ = ('dependency',)

	def __init__(self, dependency: Callable[[], T | Awaitable[T]]) -> None:
		self.dependency = dependency

	async def resolve(self, overrides: DependencyOverrides | None = None) -> T:
		"""
		Resolve the dependency, handling both sync and async callables.

		Args:
		    overrides: Optional dict mapping original dependency functions to
		              replacement functions. If the dependency is in overrides,
		              the override is called instead.

		Returns:
		    The resolved dependency value.
		"""
		# Check for override
		func = self.dependency
		if overrides and func in overrides:
			func = overrides[func]

		result = func()
		if asyncio.iscoroutine(result):
			return await result
		return result  # type: ignore[return-value]
