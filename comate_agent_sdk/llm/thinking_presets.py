"""Model-level thinking/reasoning presets.

Maps model names to ThinkingPreset, which determines:
- whether the model should send thinking parameters to the API
- how to handle thinking blocks in message history

Matching order (first match wins):
1. Exact match
2. Suffix match  (e.g. "*-thinking")
3. Prefix match  (e.g. "gemini-2.5-flash*")
4. Default       (supports_thinking=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ThinkingPreset:
    supports_thinking: bool = False
    """Whether this model uses extended thinking/reasoning."""

    budget_tokens: int | None = None
    """Anthropic/Google: token budget to send in the thinking param.
    None  = do not send a thinking param (model self-thinks, e.g. MiniMax/DeepSeek).
    int   = send thinking={type:enabled, budget_tokens:X} (Claude -thinking series).
    0     = explicitly disable Google thinking (gemini-2.5-flash default)."""

    effort: Literal["low", "medium", "high"] | None = None
    """OpenAI: reasoning_effort level. None = not applicable."""

    history_policy: Literal["preserve", "strip", "auto"] = "auto"
    """How to handle thinking blocks in serialized history.
    preserve = always keep (MiniMax-M2.5 — chain must not break).
    strip    = always remove (reserved; DeepSeek already handled by serializer).
    auto     = strip when supports_thinking=False, preserve otherwise."""


# ---------------------------------------------------------------------------
# Preset table
# ---------------------------------------------------------------------------
# Keys:
#   exact strings  → exact match
#   starts with "~" → suffix match (strip "~" prefix, match model ending)
#   starts with "^" → prefix match (strip "^" prefix, match model beginning)
# ---------------------------------------------------------------------------

_PRESETS: dict[str, ThinkingPreset] = {
    # ── Suffix: Claude -thinking virtual model names ────────────────────────
    "~-thinking": ThinkingPreset(
        supports_thinking=True,
        budget_tokens=4096,
        history_policy="auto",
    ),

    "gpt-5.2": ThinkingPreset(
        supports_thinking=True,
        effort="medium", 
        history_policy="auto"
    ),
    # ── Exact: MiniMax ───────────────────────────────────────────────────────
    "MiniMax-M2.5": ThinkingPreset(
        supports_thinking=True,
        budget_tokens=8192,   # send thinking param; model returns thinking blocks
        history_policy="preserve",
    ),
    # ── Prefix: Gemini flash (disable thinking by default) ──────────────────
    "^gemini-2.5-flash": ThinkingPreset(
        supports_thinking=False,
        budget_tokens=0,      # send thinking_budget=0 to explicitly disable
        history_policy="auto",
    ),
    # ── Prefix: Gemini Pro (enable thinking) ────────────────────────────────
    "^gemini-2.5-pro": ThinkingPreset(
        supports_thinking=True,
        budget_tokens=16_000,
        history_policy="auto",
    ),
    # ── Prefix: gemini-flash-latest / gemini-flash-lite-latest ──────────────
    "^gemini-flash": ThinkingPreset(
        supports_thinking=False,
        budget_tokens=0,
        history_policy="auto",
    ),
    # ── Exact: DeepSeek reasoner ─────────────────────────────────────────────
    "deepseek-reasoner": ThinkingPreset(
        supports_thinking=True,
        budget_tokens=None,
        history_policy="strip",  # serializer already handles; preserve as strip
    ),
    # ── Exact: OpenAI reasoning models ───────────────────────────────────────
    "o1": ThinkingPreset(supports_thinking=True, effort="medium", history_policy="auto"),
    "o1-pro": ThinkingPreset(supports_thinking=True, effort="medium", history_policy="auto"),
    "o3": ThinkingPreset(supports_thinking=True, effort="medium", history_policy="auto"),
    "o3-mini": ThinkingPreset(supports_thinking=True, effort="medium", history_policy="auto"),
    "o3-pro": ThinkingPreset(supports_thinking=True, effort="medium", history_policy="auto"),
    "o4-mini": ThinkingPreset(supports_thinking=True, effort="medium", history_policy="auto"),
}

_DEFAULT_PRESET = ThinkingPreset()  # supports_thinking=False, everything None/auto


def get_thinking_preset(model: str) -> ThinkingPreset:
    """Return the ThinkingPreset for *model*.

    Lookup order:
    1. Exact match in _PRESETS
    2. Suffix match  (~<suffix> keys)
    3. Prefix match  (^<prefix> keys)
    4. Default preset (supports_thinking=False)
    """
    # 1. Exact
    if model in _PRESETS:
        return _PRESETS[model]

    # 2. Suffix
    for key, preset in _PRESETS.items():
        if key.startswith("~") and model.endswith(key[1:]):
            return preset

    # 3. Prefix
    for key, preset in _PRESETS.items():
        if key.startswith("^") and model.startswith(key[1:]):
            return preset

    return _DEFAULT_PRESET


def resolve_api_model_name(model: str) -> str:
    """Strip virtual suffixes before sending model name to the API.

    Currently only strips the ``-thinking`` suffix used for Claude thinking
    variants (e.g. ``claude-sonnet-4-5-thinking`` → ``claude-sonnet-4-5``).
    """
    if model.endswith("-thinking"):
        return model[: -len("-thinking")]
    return model
