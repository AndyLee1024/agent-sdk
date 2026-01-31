"""
SDK 内置系统工具（Claude Code 风格）。

注意：
- 这些工具仅在 Agent(tools=None) 时会通过默认 registry 启用
- 当用户显式传入 tools=[...] 时，不会自动注入这些工具
"""

from bu_agent_sdk.system_tools.registry import get_system_tools, register_system_tools

__all__ = ["get_system_tools", "register_system_tools"]

