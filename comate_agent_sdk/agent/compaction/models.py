"""
Models for the compaction subservice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.llm.views import ChatInvokeUsage

# Default ratio of context window to use before triggering compaction
DEFAULT_THRESHOLD_RATIO = 0.80


DEFAULT_SUMMARY_SYSTEM_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.\nThis summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.\n\nBefore providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:\n\n1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:\n   - The user's explicit requests and intents\n   - Your approach to addressing the user's requests\n   - Key decisions, technical concepts and code patterns\n   - Specific details like:\n     - file names\n     - full code snippets\n     - function signatures\n     - file edits\n  - Errors that you ran into and how you fixed them\n  - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.\n2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.\n\nYour summary should include the following sections:\n\n1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail\n2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.\n3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.\n4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.\n5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.\n6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.\n6. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.\n7. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.\n8. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.\n                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.\n\nHere's an example of how your output should be structured:\n\n<example>\n<analysis>\n[Your thought process, ensuring all points are covered thoroughly and accurately]\n</analysis>\n\n<summary>\n1. Primary Request and Intent:\n   [Detailed description]\n\n2. Key Technical Concepts:\n   - [Concept 1]\n   - [Concept 2]\n   - [...]\n\n3. Files and Code Sections:\n   - [File Name 1]\n      - [Summary of why this file is important]\n      - [Summary of the changes made to this file, if any]\n      - [Important Code Snippet]\n   - [File Name 2]\n      - [Important Code Snippet]\n   - [...]\n\n4. Errors and fixes:\n    - [Detailed description of error 1]:\n      - [How you fixed the error]\n      - [User feedback on the error if any]\n    - [...]\n\n5. Problem Solving:\n   [Description of solved problems and ongoing troubleshooting]\n\n6. All user messages: \n    - [Detailed non tool use user message]\n    - [...]\n\n7. Pending Tasks:\n   - [Task 1]\n   - [Task 2]\n   - [...]\n\n8. Current Work:\n   [Precise description of current work]\n\n9. Optional Next Step:\n   [Optional Next step to take]\n\n</summary>\n</example>\n\nPlease provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response. \n\nThere may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:\n<example>\n## Compact Instructions\nWhen summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.\n</example>\n\n<example>\n# Summary instructions\nWhen you are using compact - please focus on test output and code changes. Include file reads verbatim.\n</example>\n\n\nIMPORTANT: Do NOT use any tools. You MUST respond with ONLY the <summary>...</summary> block as your text output."""


@dataclass
class CompactionConfig:
    """Configuration for the compaction service.

    The compaction service monitors token usage and automatically summarizes
    conversation history when approaching the model's context window limit.

    Attributes:
            enabled: Whether compaction is enabled. Defaults to True.
            threshold_ratio: Ratio of context window at which compaction triggers (0.0-1.0).
                    E.g., 0.80 means compact when context reaches 80% of model's limit.
            model: Optional model to use for generating summaries. If None, uses the agent's model.
            summary_system_prompt: System prompt used for summary generation.
            summary_prompt: Custom prompt for summary generation.
    """

    enabled: bool = True
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO
    model: str | None = None
    summary_system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT


@dataclass
class CompactionResult:
    """Result of a compaction operation.

    Attributes:
            compacted: Whether compaction was performed.
            original_tokens: Token count before compaction.
            new_tokens: Token count after compaction (estimated from summary output tokens).
            summary: The generated summary text (if compaction was performed).
    """

    compacted: bool
    original_tokens: int = 0
    new_tokens: int = 0
    summary: str | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None
    stop_reason: str | None = None
    raw_content_length: int = 0


@dataclass
class TokenUsage:
    """Token usage tracking for compaction decisions.

    Attributes:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cache_creation_tokens: Number of tokens used to create cache (Anthropic).
            cache_read_tokens: Number of cached tokens read.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens for compaction threshold check.

        This matches the Anthropic SDK's calculation:
        input_tokens + cache_creation_input_tokens + cache_read_input_tokens + output_tokens
        """
        return (
            self.input_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
            + self.output_tokens
        )

    @classmethod
    def from_usage(cls, usage: ChatInvokeUsage | None) -> TokenUsage:
        """Create TokenUsage from ChatInvokeUsage."""
        if usage is None:
            return cls()

        return cls(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_creation_tokens=usage.prompt_cache_creation_tokens or 0,
            cache_read_tokens=usage.prompt_cached_tokens or 0,
        )
