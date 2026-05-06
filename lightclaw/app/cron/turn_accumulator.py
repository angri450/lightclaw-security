"""Assistant turn accumulator for proactive outbound message construction.

Provides ``AssistantTurnAccumulator`` which collects streaming deltas and
final completed text during one agent turn, then produces an
``AccumulatedTurn`` with a clear authority chain:

- ``completed_text`` (from ASSISTANT_COMPLETED / final AIMessage) is the
  **only authoritative source** for outbound text.
- ``delta_text`` is streaming display-only and is NOT used for outbound
  unless ``completed_text`` is missing (with a WARNING).
- ``thinking_text`` is NEVER included in outbound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Markers that indicate the agent decided nothing is worth pushing.
_SKIP_MARKERS = {"[SKIP]", "HEARTBEAT_OK"}


@dataclass
class AccumulatedTurn:
    """Complete产物 of a single agent turn.

    Separates streaming display text (``delta_text``) from authoritative
    final text (``completed_text``) so that double-collection cannot occur
    even when the bridge's ``drop_completed_text`` mechanism fails.
    """

    turn_id: str
    call_id: str | None = None
    run_id: str | None = None

    # Streaming text — for real-time display ONLY, NOT authoritative for outbound.
    delta_text: str = ""

    # Authoritative final text from ASSISTANT_COMPLETED / final AIMessage.
    completed_text: str = ""

    # Thinking / reasoning text — MUST NEVER enter outbound.
    thinking_text: str = ""

    # Tool call records (for audit only; do not affect outbound).
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

    # Source label for completed_text.
    completed_source: str = ""  # "assistant_completed" | "final_ai_message"

    @property
    def outbound_text(self) -> str:
        """Return the text that should be sent to the user.

        Rules:
        1. Prefer ``completed_text`` (from ASSISTANT_COMPLETED / final AIMessage).
        2. Fallback to ``delta_text`` only if ``completed_text`` is empty,
           and emit a WARNING log.
        3. ``thinking_text`` is NEVER returned.
        """
        if self.completed_text:
            return self.completed_text
        if self.delta_text:
            logger.warning(
                "completed_text is empty, falling back to delta_text "
                "(len=%d). This indicates ASSISTANT_COMPLETED did not "
                "provide text content (bridge may have dropped it).",
                len(self.delta_text),
            )
            return self.delta_text
        return ""

    @property
    def is_skip(self) -> bool:
        """Check whether this turn should be skipped (no outbound send).

        Returns ``True`` when ``outbound_text`` is empty or contains a
        recognised skip marker (``[SKIP]`` / ``HEARTBEAT_OK``).
        """
        text = self.outbound_text.strip()
        if not text:
            return True
        if any(m in text for m in _SKIP_MARKERS):
            return True
        return False


class AssistantTurnAccumulator:
    """Collects streaming deltas and final text during one agent turn.

    Intended usage inside ``_run_and_maybe_dispatch()``::

        acc = AssistantTurnAccumulator(turn_id="t1")
        # During streaming:
        acc.add_delta("hello ")
        acc.add_thinking("let me think...")
        # On completion:
        acc.set_completed("hello world", source="assistant_completed")
        # Finalize:
        turn = acc.assemble()
        text = turn.outbound_text  # "hello world" (prefers completed)
    """

    def __init__(
        self,
        turn_id: str,
        *,
        call_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._turn = AccumulatedTurn(
            turn_id=turn_id,
            call_id=call_id,
            run_id=run_id,
        )

    # -- public mutation methods --

    def add_delta(self, text: str) -> None:
        """Append streaming display text (ASSISTANT_DELTA, non-thinking)."""
        self._turn.delta_text += text

    def add_thinking(self, text: str) -> None:
        """Append thinking/reasoning text (NEVER enters outbound)."""
        self._turn.thinking_text += text

    def set_completed(self, text: str, *, source: str = "assistant_completed") -> None:
        """Set the authoritative final text from ASSISTANT_COMPLETED.

        This becomes the preferred source for ``outbound_text``.
        """
        self._turn.completed_text = text
        self._turn.completed_source = source

    def record_tool_call(self, record: dict) -> None:
        """Record a tool call (audit only; does NOT affect outbound text)."""
        self._turn.tool_calls.append(record)

    def record_tool_result(self, record: dict) -> None:
        """Record a tool result (audit only; does NOT affect outbound text)."""
        self._turn.tool_results.append(record)

    # -- accessors for inline use during streaming --

    @property
    def turn_id(self) -> str:
        return self._turn.turn_id

    @turn_id.setter
    def turn_id(self, value: str) -> None:
        self._turn.turn_id = value

    @property
    def call_id(self) -> str | None:
        return self._turn.call_id

    @call_id.setter
    def call_id(self, value: str | None) -> None:
        self._turn.call_id = value

    # -- finalisation --

    def assemble(self) -> AccumulatedTurn:
        """Return the completed ``AccumulatedTurn``."""
        return self._turn
