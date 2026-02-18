"""
Compaction subservice for managing conversation context.

Automatically summarizes and compresses conversation history when token usage
approaches model's context window
"""

from comate_agent_sdk.agent.compaction.models import (
    CompactionConfig,
    CompactionResult,
)
from comate_agent_sdk.agent.compaction.service import CompactionService

__all__ = [
    "CompactionConfig",
    "CompactionResult",
    "CompactionService",
]
