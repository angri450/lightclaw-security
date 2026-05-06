"""Lightweight token counting using tiktoken.

This module provides a **single, project-wide** token counting
implementation so that every call-site uses the same encoder and the
same fallback logic.

Design choices
--------------
* **Encoder**: ``cl100k_base`` — the encoding used by GPT-4 / GPT-3.5-turbo
  and a reasonable approximation for most modern LLMs (including Qwen,
  Claude, etc.) when the exact provider tokeniser is unavailable.
* **Lazy singleton**: the encoder is loaded once on first call and
  reused thereafter (~2 MB memory, no PyTorch dependency).
* **Graceful fallback**: if ``tiktoken`` is somehow unavailable at
  runtime, falls back to ``len(text) // 4`` so the application never
  crashes due to token counting alone.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model special token cleaning
# ---------------------------------------------------------------------------

MODEL_SPECIAL_TOKEN_RE = re.compile(r"<[|｜][^|｜\r\n]{0,200}[|｜]>")

FALLBACK_MODEL_SPECIAL_TOKENS = {
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|endofprompt|>",
    "<|tool_call_result_begin|>",
    "<|tool_call_result_end|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|tool|>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
}


def strip_model_special_tokens(text):
    """Remove model-reserved special token strings from normal text.

    This function only removes complete token-shaped strings,
    such as <|endoftext|> or <｜begin▁of▁sentence｜>.

    It must NOT remove ordinary characters: <, >, |, ｜
    """
    if text is None:
        return text

    if not isinstance(text, str):
        text = str(text)

    if not text:
        return text

    for token in FALLBACK_MODEL_SPECIAL_TOKENS:
        if token in text:
            text = text.replace(token, " ")

    text = MODEL_SPECIAL_TOKEN_RE.sub(" ", text)

    # Collapse spaces and tabs, but do not flatten newlines.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def sanitize_text_payload(value):
    """Recursively sanitize strings inside common payload containers."""
    if value is None:
        return value

    if isinstance(value, str):
        return strip_model_special_tokens(value)

    if isinstance(value, list):
        return [sanitize_text_payload(item) for item in value]

    if isinstance(value, tuple):
        return tuple(sanitize_text_payload(item) for item in value)

    if isinstance(value, dict):
        return {key: sanitize_text_payload(val) for key, val in value.items()}

    return value


# ---------------------------------------------------------------------------
# Lazy-loaded tiktoken encoder
# ---------------------------------------------------------------------------

_encoder: Any | None = None
_tiktoken_available: bool | None = None


def _get_encoder() -> Any | None:
    """Return a cached tiktoken encoder, or *None* if unavailable."""
    global _encoder, _tiktoken_available

    if _tiktoken_available is not None:
        return _encoder

    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
        _tiktoken_available = True
        logger.debug("tiktoken encoder loaded (cl100k_base)")
    except Exception:
        _encoder = None
        _tiktoken_available = False
        logger.warning(
            "tiktoken is not available — falling back to len//4 estimate. Install it with: pip install tiktoken",
        )
    return _encoder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def count_str_tokens(text: str) -> int:
    """Count the number of tokens in a plain string.

    Uses tiktoken ``cl100k_base`` when available; otherwise falls back
    to ``len(text) // 4``.
    """
    enc = _get_encoder()
    if enc is not None:
        text = strip_model_special_tokens(text)
        return len(enc.encode(text, disallowed_special=()))
    return max(len(text) // 4, 1) if text else 0


def count_message_tokens(messages: list[dict] | Sequence[Any]) -> int:
    """Count tokens across a list of message-like objects.

    Accepts both:
    * ``list[dict]`` — plain dict messages with ``"content"`` key
    * ``Sequence[BaseMessage]`` — LangChain messages with ``.content`` attr

    Each message's content is resolved to text(s) and counted via
    :func:`count_str_tokens`.
    """
    total = 0
    for msg in messages:
        content = _resolve_content(msg)
        if isinstance(content, str):
            total += count_str_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    total += count_str_tokens(part)
                elif isinstance(part, dict):
                    # Content blocks: {"type": "text", "text": "..."}
                    text = part.get("text") or part.get("output") or ""
                    if text:
                        total += count_str_tokens(str(text))
                    else:
                        total += count_str_tokens(str(part))
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_content(msg: Any) -> Any:
    """Extract content from either a dict or an object with .content."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")
