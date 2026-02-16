"""Unified error formatter for terminal agent"""
from __future__ import annotations


def format_error(exc: Exception) -> tuple[str, str | None]:
    """Convert exception to user-friendly message.

    Returns:
        (error_message, suggestion) - Error message and optional suggestion
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)

    # LLM Provider errors
    if exc_type == "ModelRateLimitError":
        return "⚠️ Rate limit exceeded", "Wait a moment, or use /model to switch"

    if exc_type == "ModelProviderError":
        code = getattr(exc, 'status_code', None)
        if code == 404:
            return "⚠️ Model not found or invalid API path", "Check .agent/settings.json"
        if code == 401:
            return "⚠️ Invalid or expired API key", "Check api_key in .agent/settings.json"
        if code == 403:
            return "⚠️ Access denied to this model", "Check API key permissions"
        if code and code >= 500:
            return f"⚠️ Server error ({code})", "Try again later"
        return f"⚠️ API error: {_truncate(exc_msg, 80)}", None

    # Session errors
    if exc_type == "ChatSessionClosedError":
        return "⚠️ Session closed", "Please restart the CLI"

    # Network errors (generic detection)
    lower_msg = exc_msg.lower()
    if "timeout" in lower_msg or "timed out" in lower_msg:
        return "⚠️ Request timed out", "Check network connection, or try again"
    if "connection" in lower_msg:
        return "⚠️ Connection failed", "Check network and API endpoint"

    # Generic fallback
    return f"⚠️ Error: {_truncate(exc_msg, 60)}", "You can continue typing"


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
