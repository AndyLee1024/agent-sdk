from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(f"Invalid env var {name}={raw!r}; using default {default}.")
        return default
    if value <= 0:
        logger.warning(f"Invalid env var {name}={raw!r}; using default {default}.")
        return default
    return value


def read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning(f"Invalid env var {name}={raw!r}; using default {default}.")
        return default
    if value <= 0:
        logger.warning(f"Invalid env var {name}={raw!r}; using default {default}.")
        return default
    return value

