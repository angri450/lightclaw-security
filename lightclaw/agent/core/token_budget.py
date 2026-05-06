"""Token budget step — truncate tool outputs and drop old messages to fit the context window."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from lightclaw.agent.utils.token_counting import count_message_tokens as _count_tokens
from lightclaw.agent.utils.tool_message_utils import _truncate_text as _truncate_text_impl

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOOL_OUTPUT_LENGTH = 8_000


def _clone_message(message: BaseMessage) -> BaseMessage:
    """Clone a LangChain message without triggering Pydantic v2 deprecation warnings."""
    if hasattr(message, "model_copy"):
        return message.model_copy(deep=True)
    if hasattr(message, "copy"):
        return message.copy(deep=True)
    raise TypeError(f"Message type does not support copying: {type(message)!r}")


@dataclass(frozen=True, slots=True)
class ToolOutputTruncationStats:
    """Aggregate diagnostics for tool-output truncation."""

    truncated_messages: int = 0
    truncated_chars: int = 0


class TokenBudgetEnforcer:
    """Two-phase token management: truncate oversized tool outputs, then drop oldest messages.

    ``truncate_tool_outputs`` is a pure transformation (no state) and can be
    called multiple times.  ``enforce`` uses the shared token caches from
    ``MemoryCompactor`` so that the same message is never re-encoded twice
    within a session.
    """

    def __init__(self, max_input_length: int) -> None:
        self._max_input_length = max_input_length

    @property
    def max_input_length(self) -> int:
        """Expose the configured context-window limit for diagnostics."""
        return self._max_input_length

    def truncate_tool_outputs(
        self,
        messages: list[BaseMessage],
        max_length: int | None = None,
    ) -> list[BaseMessage]:
        """Return a new list with oversized ToolMessage content truncated.

        Creates new message objects only when truncation is needed — never
        mutates the originals.
        """
        result, _ = self.truncate_tool_outputs_with_stats(messages, max_length=max_length)
        return result

    def truncate_tool_outputs_with_stats(
        self,
        messages: list[BaseMessage],
        max_length: int | None = None,
    ) -> tuple[list[BaseMessage], ToolOutputTruncationStats]:
        """Return truncated messages plus aggregate truncation diagnostics."""
        if max_length is None:
            max_length = int(os.environ.get(
                "LIGHTCLAW_TOOL_MESSAGE_SINGLE_MAX_CHARS",
                os.environ.get("MAX_FORMATTER_TEXT_LENGTH", str(_DEFAULT_MAX_TOOL_OUTPUT_LENGTH))
            ))

        result: list[BaseMessage] = []
        truncated_messages = 0
        truncated_chars = 0
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                result.append(msg)
                continue
            content = msg.content
            if isinstance(content, str) and len(content) > max_length:
                truncated = _truncate_text_impl(content, max_length)
                new_msg = _clone_message(msg)
                new_msg.content = truncated
                result.append(new_msg)
                truncated_messages += 1
                truncated_chars += max(len(content) - len(truncated), 0)
            elif isinstance(content, list):
                new_parts = []
                changed = False
                removed_chars = 0
                for part in content:
                    if isinstance(part, str) and len(part) > max_length:
                        truncated = _truncate_text_impl(part, max_length)
                        new_parts.append(truncated)
                        changed = True
                        removed_chars += max(len(part) - len(truncated), 0)
                    elif isinstance(part, dict):
                        text = part.get("text", "")
                        if isinstance(text, str) and len(text) > max_length:
                            truncated = _truncate_text_impl(text, max_length)
                            new_parts.append({**part, "text": truncated})
                            changed = True
                            removed_chars += max(len(text) - len(truncated), 0)
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                if changed:
                    new_msg = _clone_message(msg)
                    new_msg.content = new_parts
                    result.append(new_msg)
                    truncated_messages += 1
                    truncated_chars += removed_chars
                else:
                    result.append(msg)
            else:
                result.append(msg)
        return result, ToolOutputTruncationStats(
            truncated_messages=truncated_messages,
            truncated_chars=truncated_chars,
        )

    def enforce(
        self,
        messages: list[BaseMessage],
        token_cache: dict[str, int] | None = None,
        sys_token_cache: list[int] | None = None,
        reserved_tokens: int = 0,
    ) -> list[BaseMessage]:
        """Drop oldest non-system messages until total tokens fit within max_input_length.

        Args:
            token_cache: Shared per-message token-count cache (mutated in place).
                Pass MemoryCompactor.token_cache so repeated messages aren't re-encoded.
            sys_token_cache: Single-element list caching the system-messages token total.
                Pass MemoryCompactor.sys_token_cache to avoid re-counting static prompts.
            reserved_tokens: Additional token budget to reserve for extra content (e.g.
                bootstrap blocks, memory recall, facts) injected into the system message
                *after* enforcement. The available budget is reduced by this amount.
        """
        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_sys = [m for m in messages if not isinstance(m, SystemMessage)]

        if sys_token_cache:
            sys_tokens = sys_token_cache[0]
        else:
            sys_tokens = _count_tokens(sys_msgs)
            if sys_token_cache is not None:
                sys_token_cache.append(sys_tokens)

        budget = self._max_input_length - sys_tokens - reserved_tokens
        if budget <= 0:
            logger.warning(
                "System messages alone exceed max_input_length (%d); returning only system messages + last message",
                self._max_input_length,
            )
            return sys_msgs + (non_sys[-1:] if non_sys else [])

        # v2: tool message total budget check — hard truncate, not just warning
        tool_total_chars = 0
        tool_msg_indices: list[tuple[int, int]] = []  # (index, chars)
        tool_total_max = int(os.environ.get("LIGHTCLAW_TOOL_MESSAGE_TOTAL_MAX_CHARS", "24000"))
        placeholder_max = int(os.environ.get(
            "LIGHTCLAW_TOOL_MESSAGE_TRUNCATED_PLACEHOLDER_MAX_CHARS", "800"
        ))
        for i, msg in enumerate(non_sys):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                clen = len(msg.content)
                tool_total_chars += clen
                tool_msg_indices.append((i, clen))

        if tool_total_chars > tool_total_max:
            # Build placeholder for replaced tool outputs
            placeholder = (
                f"\n\n[Tool output truncated by LightClaw context guard.\n"
                f"Reason: total tool output exceeded {tool_total_max} chars.\n"
                f"Original chars: <orig>.\n"
                f"This older tool output was removed before model call.]\n"
            )
            # Truncate from oldest tool messages first, keep latest ones
            remaining_budget = tool_total_max
            truncated_count = 0
            truncated_chars_removed = 0
            # Process in reverse: latest tool msgs first (they get budget priority)
            for idx, clen in reversed(tool_msg_indices):
                if remaining_budget <= 0:
                    # No budget left — replace entire content with short placeholder
                    orig_chars = clen
                    short_placeholder = (
                        f"[Tool output removed by context guard. "
                        f"Original chars: {orig_chars}. "
                        f"Total tool budget {tool_total_max} exceeded.]"
                    )[:placeholder_max]
                    new_msg = _clone_message(non_sys[idx])
                    new_msg.content = short_placeholder
                    non_sys[idx] = new_msg
                    truncated_count += 1
                    truncated_chars_removed += orig_chars
                elif clen > remaining_budget:
                    # Partial truncation: keep head, append note
                    keep_chars = max(remaining_budget - placeholder_max, 100)
                    kept = non_sys[idx].content[:keep_chars]
                    note = (
                        f"\n\n[Tool output truncated: {clen - keep_chars} chars removed. "
                        f"Total tool budget {tool_total_max} exceeded.]"
                    )[:placeholder_max]
                    new_msg = _clone_message(non_sys[idx])
                    new_msg.content = kept + note
                    non_sys[idx] = new_msg
                    truncated_count += 1
                    truncated_chars_removed += clen - len(new_msg.content)
                    remaining_budget = 0
                else:
                    remaining_budget -= clen

            logger.warning(
                "tool_message_total_truncated: before_chars=%d after_chars<=%d "
                "truncated_messages=%d max_chars=%d",
                tool_total_chars, tool_total_max, truncated_count, tool_total_max,
            )

        kept: list[BaseMessage] = []
        running = 0
        for msg in reversed(non_sys):
            if token_cache is not None:
                msg_id = str(getattr(msg, "id", None) or id(msg))
                if msg_id not in token_cache:
                    token_cache[msg_id] = _count_tokens([msg])
                msg_tokens = token_cache[msg_id]
            else:
                msg_tokens = _count_tokens([msg])

            if running + msg_tokens > budget:
                logger.info(
                    "Token budget reached (~%d / %d tokens). Dropping %d older message(s).",
                    running,
                    self._max_input_length,
                    len(non_sys) - len(kept),
                )
                break
            running += msg_tokens
            kept.append(msg)

        kept.reverse()
        return sys_msgs + kept
