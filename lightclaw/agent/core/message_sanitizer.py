"""Sanitize LangChain messages to ensure tool_calls / ToolMessages are paired.

LLM APIs (OpenAI, Anthropic, etc.) require that every ``AIMessage.tool_calls``
entry has a corresponding ``ToolMessage`` with matching ``tool_call_id``, and
vice-versa.  In practice, orphans appear due to:

* User cancellation mid-tool-call
* Token-budget trimming that drops one half of a pair
* Corrupt / migrated session data

This module provides :func:`sanitize_tool_messages` which detects and removes
unpaired messages so the LLM never receives an invalid sequence.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger(__name__)


def _collect_ids(
    messages: Sequence[BaseMessage],
) -> tuple[set[str], set[str]]:
    """Return (tool_call_ids, tool_result_ids) from messages."""
    call_ids: set[str] = set()
    result_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tid = tc.get("id")
                if tid:
                    call_ids.add(tid)
        elif isinstance(msg, ToolMessage) and msg.tool_call_id:
            result_ids.add(msg.tool_call_id)

    return call_ids, result_ids


def _strip_ai_message(
    msg: AIMessage,
    valid_ids: set[str],
) -> AIMessage | None:
    """Return a cleaned AIMessage, or None if it should be dropped."""
    surviving_calls = [tc for tc in msg.tool_calls if tc.get("id") in valid_ids]

    if len(surviving_calls) == len(msg.tool_calls):
        return msg

    if surviving_calls:
        return AIMessage(
            content=msg.content,
            tool_calls=surviving_calls,
            additional_kwargs=msg.additional_kwargs,
            response_metadata=msg.response_metadata,
            id=msg.id,
        )

    if msg.content:
        return AIMessage(
            content=msg.content,
            additional_kwargs=msg.additional_kwargs,
            response_metadata=msg.response_metadata,
            id=msg.id,
        )

    logger.debug("Dropping empty AIMessage after removing orphan tool_calls")
    return None


def sanitize_tool_messages(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Ensure every tool_call has a matching ToolMessage, and vice-versa.

    The algorithm:
    1. Collect all ``tool_call`` IDs declared by ``AIMessage.tool_calls``.
    2. Collect all ``tool_call_id`` values from ``ToolMessage`` instances.
    3. Compute the set of *valid* IDs — those present on **both** sides.
    4. Strip orphan ``tool_calls`` entries from ``AIMessage`` objects and
       drop orphan ``ToolMessage`` objects entirely.

    An ``AIMessage`` whose ``tool_calls`` list becomes empty after
    stripping is kept only if it still has non-empty ``content``.

    Returns a **new** list; the original messages are not mutated.
    """
    call_ids, result_ids = _collect_ids(messages)
    valid_ids = call_ids & result_ids

    if call_ids == valid_ids and result_ids == valid_ids:
        return list(messages)

    logger.info(
        "Sanitizing tool messages: %d orphan tool_call(s), %d orphan ToolMessage(s)",
        len(call_ids - valid_ids),
        len(result_ids - valid_ids),
    )

    cleaned: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id in valid_ids:
                cleaned.append(msg)
            else:
                logger.debug(
                    "Dropping orphan ToolMessage (tool_call_id=%s)",
                    msg.tool_call_id,
                )
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            stripped = _strip_ai_message(msg, valid_ids)
            if stripped is not None:
                cleaned.append(stripped)
        else:
            cleaned.append(msg)

    return cleaned
