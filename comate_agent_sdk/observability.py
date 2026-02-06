"""
Observability module for comate_agent_sdk with optional Laminar (lmnr) integration.

This module provides:
- `observe` decorator for tracing functions
- `observe_debug` decorator that only traces in debug mode
- Manual span helpers via `Laminar.start_as_current_span` and `Laminar.set_span_output`

If lmnr is not installed, all decorators become no-ops.

Usage:
    from comate_agent_sdk.observability import observe, observe_debug, Laminar

    @observe(name="my_function")
    async def my_function():
        ...

    # Manual spans (with null safety):
    if Laminar is not None:
        with Laminar.start_as_current_span(name="action", input={...}, span_type='TOOL'):
            result = await do_something()
            Laminar.set_span_output(result)
"""

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, TypeVar, cast

logger = logging.getLogger(__name__)

# Type definitions
F = TypeVar("F", bound=Callable[..., Any])

_LMNR_AVAILABLE = False
_lmnr_observe = None
Laminar = None

# Try to import lmnr
try:
    from lmnr import Laminar  # type: ignore
    from lmnr import observe as _lmnr_observe  # type: ignore

    _LMNR_AVAILABLE = True
    logger.debug("Laminar (lmnr) is available for observability")
except ImportError:
    logger.debug(
        "Laminar (lmnr) not installed - observability decorators will be no-ops"
    )
    _LMNR_AVAILABLE = False


def _is_debug_mode() -> bool:
    """Check if we're in debug mode based on environment variables."""
    lmnr_debug_mode = os.getenv("LMNR_LOGGING_LEVEL", "").lower()
    if lmnr_debug_mode == "debug":
        return True
    # Also check BU_DEBUG for convenience
    if os.getenv("BU_DEBUG", "").lower() in ("1", "true", "yes"):
        return True
    return False


def _create_no_op_decorator(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Create a no-op decorator that accepts all lmnr observe parameters but does nothing."""
    import asyncio
    import inspect

    def decorator(func: F) -> F:
        # Check for async generators first (async def with yield)
        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                async for item in func(*args, **kwargs):
                    yield item

            return cast(F, async_gen_wrapper)
        elif asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return cast(F, async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return cast(F, sync_wrapper)

    return decorator


def _create_async_gen_observe_decorator(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    metadata: dict[str, Any] | None = None,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    **kwargs: Any,
) -> Callable[[F], F]:
    """Create a decorator for async generators that wraps them in a Laminar span."""
    from contextlib import asynccontextmanager

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_gen_wrapper(*args, **inner_kwargs):
            span_name = name or func.__name__

            # Build input for span (respecting ignore_input)
            span_input = None
            if not ignore_input:
                span_input = {
                    "args": str(args)[:500],
                    "kwargs": str(inner_kwargs)[:500],
                }

            # Start a span that will encompass the entire generator
            span_ctx = Laminar.start_as_current_span(
                name=span_name,
                input=span_input,
                span_type=span_type,
                metadata=metadata,
            )

            with span_ctx:
                try:
                    async for item in func(*args, **inner_kwargs):
                        yield item
                except Exception as e:
                    # Set error output before re-raising
                    if not ignore_output:
                        Laminar.set_span_output({"error": str(e)})
                    raise

        return cast(F, async_gen_wrapper)

    return decorator


def observe(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    metadata: dict[str, Any] | None = None,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    **kwargs: Any,
) -> Callable[[F], F]:
    """
    Observability decorator that traces function execution when lmnr is available.

    This decorator will use lmnr's observe decorator if lmnr is installed,
    otherwise it will be a no-op.

    Args:
        name: Name of the span/trace
        ignore_input: Whether to ignore function input parameters in tracing
        ignore_output: Whether to ignore function output in tracing
        metadata: Additional metadata to attach to the span
        span_type: Type of span ('DEFAULT', 'LLM', or 'TOOL')
        **kwargs: Additional parameters passed to lmnr observe

    Example:
        @observe(name="my_function", metadata={"version": "1.0"})
        async def my_function(param1, param2):
            return param1 + param2
    """
    import inspect

    decorator_kwargs = {
        "name": name,
        "ignore_input": ignore_input,
        "ignore_output": ignore_output,
        "metadata": metadata,
        "span_type": span_type,
        **kwargs,
    }

    def decorator(func: F) -> F:
        # Async generators need special handling - use manual span wrapper
        if inspect.isasyncgenfunction(func):
            if _LMNR_AVAILABLE and Laminar is not None:
                return _create_async_gen_observe_decorator(**decorator_kwargs)(func)
            else:
                return _create_no_op_decorator(**decorator_kwargs)(func)

        if _LMNR_AVAILABLE and _lmnr_observe:
            return cast(F, _lmnr_observe(**decorator_kwargs)(func))
        else:
            return _create_no_op_decorator(**decorator_kwargs)(func)

    return decorator


def observe_debug(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    metadata: dict[str, Any] | None = None,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    **kwargs: Any,
) -> Callable[[F], F]:
    """
    Debug-only observability decorator that only traces when in debug mode.

    This decorator will use lmnr's observe decorator if both lmnr is installed
    AND we're in debug mode, otherwise it will be a no-op.

    Debug mode is enabled by:
    - LMNR_LOGGING_LEVEL=debug
    - BU_DEBUG=1/true/yes

    Args:
        name: Name of the span/trace
        ignore_input: Whether to ignore function input parameters in tracing
        ignore_output: Whether to ignore function output in tracing
        metadata: Additional metadata to attach to the span
        span_type: Type of span ('DEFAULT', 'LLM', or 'TOOL')
        **kwargs: Additional parameters passed to lmnr observe

    Example:
        @observe_debug(name="debug_function")
        async def debug_function():
            ...
    """
    import inspect

    decorator_kwargs = {
        "name": name,
        "ignore_input": ignore_input,
        "ignore_output": ignore_output,
        "metadata": metadata,
        "span_type": span_type,
        **kwargs,
    }

    def decorator(func: F) -> F:
        # Async generators need special handling - use manual span wrapper
        if inspect.isasyncgenfunction(func):
            if _LMNR_AVAILABLE and Laminar is not None and _is_debug_mode():
                return _create_async_gen_observe_decorator(**decorator_kwargs)(func)
            else:
                return _create_no_op_decorator(**decorator_kwargs)(func)

        if _LMNR_AVAILABLE and _lmnr_observe and _is_debug_mode():
            return cast(F, _lmnr_observe(**decorator_kwargs)(func))
        else:
            return _create_no_op_decorator(**decorator_kwargs)(func)

    return decorator


# Convenience functions
def is_lmnr_available() -> bool:
    """Check if lmnr is available for tracing."""
    return _LMNR_AVAILABLE


def is_debug_mode() -> bool:
    """Check if we're currently in debug mode."""
    return _is_debug_mode()


def get_observability_status() -> dict[str, bool]:
    """Get the current status of observability features."""
    return {
        "lmnr_available": _LMNR_AVAILABLE,
        "debug_mode": _is_debug_mode(),
        "observe_active": _LMNR_AVAILABLE,
        "observe_debug_active": _LMNR_AVAILABLE and _is_debug_mode(),
    }
