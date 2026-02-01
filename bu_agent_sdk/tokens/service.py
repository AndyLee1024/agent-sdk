"""
Token cost service that tracks LLM token usage and costs.

Fetches pricing data from LiteLLM repository and caches it for 1 day.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import httpx

from bu_agent_sdk.llm.views import ChatInvokeUsage
from bu_agent_sdk.tokens.custom_pricing import CUSTOM_MODEL_PRICING
from bu_agent_sdk.tokens.mappings import MODEL_TO_LITELLM
from bu_agent_sdk.tokens.views import (
    CachedPricingData,
    ModelPricing,
    ModelUsageStats,
    ModelUsageTokens,
    TokenCostCalculated,
    TokenUsageEntry,
    UsageSummary,
)

logger = logging.getLogger(__name__)


def xdg_cache_home() -> Path:
    default = Path.home() / ".cache"
    xdg_path = os.getenv("XDG_CACHE_HOME")
    if xdg_path and (path := Path(xdg_path)).is_absolute():
        return path
    return default


class TokenCost:
    """Service for tracking token usage and calculating costs"""

    CACHE_DIR_NAME = "bu_agent_sdk/token_cost"
    CACHE_DURATION = timedelta(days=1)
    PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

    def __init__(self, include_cost: bool = False):
        env_val = os.getenv("bu_agent_sdk_CALCULATE_COST")
        if env_val is not None:
            self.include_cost = env_val.lower() == "true"
        else:
            self.include_cost = include_cost

        self.usage_history: list[TokenUsageEntry] = []
        self._pricing_data: dict[str, Any] | None = None
        self._initialized = False
        self._cache_dir = xdg_cache_home() / self.CACHE_DIR_NAME

    async def initialize(self) -> None:
        """Initialize the service by loading pricing data"""
        if not self._initialized:
            if self.include_cost:
                await self._load_pricing_data()
            self._initialized = True

    async def _load_pricing_data(self) -> None:
        """Load pricing data from cache or fetch from GitHub"""
        # Try to find a valid cache file
        cache_file = await self._find_valid_cache()

        if cache_file:
            await self._load_from_cache(cache_file)
        else:
            await self._fetch_and_cache_pricing_data()

    async def _find_valid_cache(self) -> Path | None:
        """Find the most recent valid cache file"""
        try:
            # Ensure cache directory exists
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # List all JSON files in the cache directory
            cache_files = list(self._cache_dir.glob("*.json"))

            if not cache_files:
                return None

            # Sort by modification time (most recent first)
            cache_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Check each file until we find a valid one
            for cache_file in cache_files:
                if await self._is_cache_valid(cache_file):
                    return cache_file
                else:
                    # Clean up old cache files
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass

            return None
        except Exception:
            return None

    async def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if a specific cache file is valid and not expired"""
        try:
            if not cache_file.exists():
                return False

            # Read the cached data
            cached = CachedPricingData.model_validate_json(
                await anyio.Path(cache_file).read_text()
            )

            # Check if cache is still valid
            return datetime.now() - cached.timestamp < self.CACHE_DURATION
        except Exception:
            return False

    async def _load_from_cache(self, cache_file: Path) -> None:
        """Load pricing data from a specific cache file"""
        try:
            content = await anyio.Path(cache_file).read_text()
            cached = CachedPricingData.model_validate_json(content)
            self._pricing_data = cached.data
        except Exception as e:
            logger.debug(f"Error loading cached pricing data from {cache_file}: {e}")
            # Fall back to fetching
            await self._fetch_and_cache_pricing_data()

    async def _fetch_and_cache_pricing_data(self) -> None:
        """Fetch pricing data from LiteLLM GitHub and cache it with timestamp"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.PRICING_URL, timeout=30)
                response.raise_for_status()

                self._pricing_data = response.json()

            # Create cache object with timestamp
            cached = CachedPricingData(
                timestamp=datetime.now(), data=self._pricing_data or {}
            )

            # Ensure cache directory exists
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Create cache file with timestamp in filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = self._cache_dir / f"pricing_{timestamp_str}.json"

            await anyio.Path(cache_file).write_text(cached.model_dump_json(indent=2))
        except Exception as e:
            logger.debug(f"Error fetching pricing data: {e}")
            # Fall back to empty pricing data
            self._pricing_data = {}

    def _find_model_in_pricing_data(self, model_name: str) -> dict | None:
        """Try to find model in pricing data using various name formats."""
        if not self._pricing_data:
            return None

        # Try exact match first
        if model_name in self._pricing_data:
            return self._pricing_data[model_name]

        # Try mapped name
        mapped_name = MODEL_TO_LITELLM.get(model_name)
        if mapped_name and mapped_name in self._pricing_data:
            return self._pricing_data[mapped_name]

        # Try with common provider prefixes
        prefixes = ["anthropic/", "openai/", "google/", "azure/", "bedrock/"]
        for prefix in prefixes:
            prefixed_name = f"{prefix}{model_name}"
            if prefixed_name in self._pricing_data:
                return self._pricing_data[prefixed_name]

        # Try without provider prefix if model_name has one
        if "/" in model_name:
            bare_name = model_name.split("/", 1)[1]
            if bare_name in self._pricing_data:
                return self._pricing_data[bare_name]

        return None

    async def get_model_pricing(self, model_name: str) -> ModelPricing | None:
        """Get pricing information for a specific model"""
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()

        # Check custom pricing first
        if model_name in CUSTOM_MODEL_PRICING:
            data = CUSTOM_MODEL_PRICING[model_name]
            return ModelPricing(
                model=model_name,
                input_cost_per_token=data.get("input_cost_per_token"),
                output_cost_per_token=data.get("output_cost_per_token"),
                max_tokens=data.get("max_tokens"),
                max_input_tokens=data.get("max_input_tokens"),
                max_output_tokens=data.get("max_output_tokens"),
                cache_read_input_token_cost=data.get("cache_read_input_token_cost"),
                cache_creation_input_token_cost=data.get(
                    "cache_creation_input_token_cost"
                ),
            )

        # Find model in pricing data using flexible lookup
        data = self._find_model_in_pricing_data(model_name)
        if data is None:
            return None

        return ModelPricing(
            model=model_name,
            input_cost_per_token=data.get("input_cost_per_token"),
            output_cost_per_token=data.get("output_cost_per_token"),
            max_tokens=data.get("max_tokens"),
            max_input_tokens=data.get("max_input_tokens"),
            max_output_tokens=data.get("max_output_tokens"),
            cache_read_input_token_cost=data.get("cache_read_input_token_cost"),
            cache_creation_input_token_cost=data.get("cache_creation_input_token_cost"),
        )

    async def calculate_cost(
        self, model: str, usage: ChatInvokeUsage
    ) -> TokenCostCalculated | None:
        if not self.include_cost:
            return None

        data = await self.get_model_pricing(model)
        if data is None:
            return None

        uncached_prompt_tokens = usage.prompt_tokens - (usage.prompt_cached_tokens or 0)

        return TokenCostCalculated(
            new_prompt_tokens=usage.prompt_tokens,
            new_prompt_cost=uncached_prompt_tokens * (data.input_cost_per_token or 0),
            # Cached tokens
            prompt_read_cached_tokens=usage.prompt_cached_tokens,
            prompt_read_cached_cost=usage.prompt_cached_tokens
            * data.cache_read_input_token_cost
            if usage.prompt_cached_tokens and data.cache_read_input_token_cost
            else None,
            # Cache creation tokens
            prompt_cached_creation_tokens=usage.prompt_cache_creation_tokens,
            prompt_cache_creation_cost=usage.prompt_cache_creation_tokens
            * data.cache_creation_input_token_cost
            if data.cache_creation_input_token_cost
            and usage.prompt_cache_creation_tokens
            else None,
            # Completion tokens
            completion_tokens=usage.completion_tokens,
            completion_cost=usage.completion_tokens
            * float(data.output_cost_per_token or 0),
        )

    def add_usage(
        self,
        model: str,
        usage: ChatInvokeUsage,
        *,
        level: str | None = None,
        source: str | None = None,
    ) -> TokenUsageEntry:
        """Add token usage entry to history (without calculating cost)"""
        entry = TokenUsageEntry(
            model=model,
            timestamp=datetime.now(),
            usage=usage,
            level=level,
            source=source,
        )

        self.usage_history.append(entry)

        return entry

    def get_usage_tokens_for_model(self, model: str) -> ModelUsageTokens:
        """Get usage tokens for a specific model"""
        filtered_usage = [u for u in self.usage_history if u.model == model]

        return ModelUsageTokens(
            model=model,
            prompt_tokens=sum(u.usage.prompt_tokens for u in filtered_usage),
            prompt_cached_tokens=sum(
                u.usage.prompt_cached_tokens or 0 for u in filtered_usage
            ),
            completion_tokens=sum(u.usage.completion_tokens for u in filtered_usage),
            total_tokens=sum(
                u.usage.prompt_tokens + u.usage.completion_tokens
                for u in filtered_usage
            ),
        )

    async def get_usage_summary(
        self, model: str | None = None, since: datetime | None = None
    ) -> UsageSummary:
        """Get summary of token usage and costs (costs calculated on-the-fly)"""
        filtered_usage = self.usage_history

        if model:
            filtered_usage = [u for u in filtered_usage if u.model == model]

        if since:
            filtered_usage = [u for u in filtered_usage if u.timestamp >= since]

        if not filtered_usage:
            return UsageSummary(
                total_prompt_tokens=0,
                total_prompt_cost=0.0,
                total_prompt_cached_tokens=0,
                total_prompt_cached_cost=0.0,
                total_completion_tokens=0,
                total_completion_cost=0.0,
                total_tokens=0,
                total_cost=0.0,
                entry_count=0,
            )

        # Calculate totals
        total_prompt = sum(u.usage.prompt_tokens for u in filtered_usage)
        total_completion = sum(u.usage.completion_tokens for u in filtered_usage)
        total_tokens = total_prompt + total_completion
        total_prompt_cached = sum(
            u.usage.prompt_cached_tokens or 0 for u in filtered_usage
        )

        # Calculate per-model stats with record-by-record cost calculation
        model_stats: dict[str, ModelUsageStats] = {}
        level_stats: dict[str, ModelUsageStats] = {}

        total_new_prompt_cost = 0.0
        total_completion_cost = 0.0
        total_prompt_cached_cost = 0.0
        total_cache_creation_cost = 0.0

        for entry in filtered_usage:
            if entry.model not in model_stats:
                model_stats[entry.model] = ModelUsageStats(model=entry.model)

            stats = model_stats[entry.model]
            stats.prompt_tokens += entry.usage.prompt_tokens
            stats.completion_tokens += entry.usage.completion_tokens
            stats.total_tokens += (
                entry.usage.prompt_tokens + entry.usage.completion_tokens
            )
            stats.invocations += 1

            if entry.level:
                if entry.level not in level_stats:
                    level_stats[entry.level] = ModelUsageStats(model=entry.level)
                ls = level_stats[entry.level]
                ls.prompt_tokens += entry.usage.prompt_tokens
                ls.completion_tokens += entry.usage.completion_tokens
                ls.total_tokens += entry.usage.prompt_tokens + entry.usage.completion_tokens
                ls.invocations += 1

            if self.include_cost:
                # Calculate cost record by record using the updated calculate_cost function
                cost = await self.calculate_cost(entry.model, entry.usage)
                if cost:
                    stats.cost += cost.total_cost
                    total_new_prompt_cost += cost.new_prompt_cost
                    total_completion_cost += cost.completion_cost
                    total_prompt_cached_cost += cost.prompt_read_cached_cost or 0.0
                    total_cache_creation_cost += cost.prompt_cache_creation_cost or 0.0

                    if entry.level:
                        level_stats[entry.level].cost += cost.total_cost

        def _fill_averages(stats_map: dict[str, ModelUsageStats]) -> None:
            for s in stats_map.values():
                if s.invocations > 0:
                    s.average_tokens_per_invocation = s.total_tokens / s.invocations

        _fill_averages(model_stats)
        _fill_averages(level_stats)

        total_prompt_cost = (
            total_new_prompt_cost + total_prompt_cached_cost + total_cache_creation_cost
        )
        total_cost = total_prompt_cost + total_completion_cost

        return UsageSummary(
            total_prompt_tokens=total_prompt,
            total_prompt_cost=total_prompt_cost,
            total_prompt_cached_tokens=total_prompt_cached,
            total_prompt_cached_cost=total_prompt_cached_cost,
            total_completion_tokens=total_completion,
            total_completion_cost=total_completion_cost,
            total_tokens=total_tokens,
            total_cost=total_cost,
            entry_count=len(filtered_usage),
            by_model=model_stats,
            by_level=level_stats,
        )

    def _format_tokens(self, tokens: int) -> str:
        """Format token count with k suffix for thousands"""
        if tokens >= 1000000000:
            return f"{tokens / 1000000000:.1f}B"
        if tokens >= 1000000:
            return f"{tokens / 1000000:.1f}M"
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}k"
        return str(tokens)

    async def get_cost_by_model(self) -> dict[str, ModelUsageStats]:
        """Get cost breakdown by model"""
        summary = await self.get_usage_summary()
        return summary.by_model

    def clear_history(self) -> None:
        """Clear usage history"""
        self.usage_history = []

    async def refresh_pricing_data(self) -> None:
        """Force refresh of pricing data from GitHub"""
        if self.include_cost:
            await self._fetch_and_cache_pricing_data()

    async def clean_old_caches(self, keep_count: int = 3) -> None:
        """Clean up old cache files, keeping only the most recent ones"""
        try:
            # List all JSON files in the cache directory
            cache_files = list(self._cache_dir.glob("*.json"))

            if len(cache_files) <= keep_count:
                return

            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_mtime)

            # Remove all but the most recent files
            for cache_file in cache_files[:-keep_count]:
                try:
                    os.remove(cache_file)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Error cleaning old cache files: {e}")

    async def ensure_pricing_loaded(self) -> None:
        """Ensure pricing data is loaded in the background. Call this after creating the service."""
        if not self._initialized and self.include_cost:
            # This will run in the background and won't block
            await self.initialize()
