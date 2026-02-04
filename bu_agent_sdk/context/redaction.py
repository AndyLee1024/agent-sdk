from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("bu_agent_sdk.context.redaction")


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redacted: bool


_REDACTED_VALUE = "***REDACTED***"

# 固定规则：按 key 名与常见 token 形态脱敏
_SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
    "authorization",
    "cookie",
)

_TOKEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Authorization: Bearer <token>
    (re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-\._~\+/]+=*)"), "Bearer ***REDACTED***"),
    # Common API key prefixes
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), _REDACTED_VALUE),
    (re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{10,}\b"), _REDACTED_VALUE),
    # AWS Access Key ID (best-effort)
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _REDACTED_VALUE),
]


def _is_sensitive_key(key: str) -> bool:
    k = (key or "").lower().strip()
    k = k.replace("-", "_")
    return any(s in k for s in _SENSITIVE_KEY_SUBSTRINGS)


def _redact_obj(obj: Any) -> tuple[Any, bool]:
    """对 JSON 对象做递归脱敏，返回 (new_obj, redacted)。"""
    if isinstance(obj, dict):
        changed = False
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if _is_sensitive_key(str(k)):
                out[str(k)] = _REDACTED_VALUE
                changed = True
                continue
            new_v, c = _redact_obj(v)
            out[str(k)] = new_v
            changed = changed or c
        return out, changed

    if isinstance(obj, list):
        changed = False
        out_list: list[Any] = []
        for v in obj:
            new_v, c = _redact_obj(v)
            out_list.append(new_v)
            changed = changed or c
        return out_list, changed

    if isinstance(obj, str):
        rr = redact_text(obj)
        return rr.text, rr.redacted

    return obj, False


def redact_text(text: str) -> RedactionResult:
    """对普通文本做固定规则脱敏。"""
    if not text:
        return RedactionResult(text=text, redacted=False)

    redacted = False
    out = text
    for pattern, replacement in _TOKEN_PATTERNS:
        new_out, n = pattern.subn(replacement, out)
        if n:
            redacted = True
            out = new_out

    return RedactionResult(text=out, redacted=redacted)


def redact_json_text(text: str) -> RedactionResult:
    """对 JSON 字符串做 key/regex 双通道脱敏（失败则回退到纯文本）。"""
    if not text:
        return RedactionResult(text=text, redacted=False)

    try:
        parsed = json.loads(text)
    except Exception:
        return redact_text(text)

    redacted_obj, changed = _redact_obj(parsed)
    if not changed:
        # 即使 key 没命中，也做一次 regex 兜底（避免 token 嵌在字符串里）
        rr = redact_text(text)
        return rr

    try:
        dumped = json.dumps(redacted_obj, ensure_ascii=False)
    except Exception:
        logger.debug("JSON dumps failed after redaction; fallback to text redaction")
        return redact_text(text)

    rr = redact_text(dumped)
    return RedactionResult(text=rr.text, redacted=bool(changed or rr.redacted))
