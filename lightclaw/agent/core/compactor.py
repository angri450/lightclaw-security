"""Memory compaction step — summarise old messages when the token threshold is exceeded."""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from lightclaw.agent.utils.text_sanitizer import sanitize_summary_text
from lightclaw.agent.utils.token_counting import count_message_tokens as _count_tokens

if TYPE_CHECKING:
    from lightclaw.agent.memory import LightClawInMemoryMemory, MemoryManager
    from lightclaw.agent.memory.compaction_config import CompactionConfig

logger = logging.getLogger(__name__)


class MemoryCompactor:
    """Compact old messages into a rolling summary when the token budget is near exhausted.

    Also owns the compressed-summary state and exposes helper methods for
    building the summary and flush-reminder blocks that go into the system prompt.
    """

    def __init__(
        self,
        *,
        memory_manager: MemoryManager | None,
        memory: LightClawInMemoryMemory | None,
        compact_threshold: int,
        keep_recent: int,
        channel: str,
        session_id: str,
        compaction_config: CompactionConfig | None,
        compressed_summary: str = "",
        language: str = "zh",
    ) -> None:
        self._memory_manager = memory_manager
        self._memory = memory
        self._compact_threshold = compact_threshold
        self._keep_recent = keep_recent
        self._channel = channel
        self._session_id = session_id
        self._compaction_config = compaction_config
        self._language = language
        self._compressed_summary = compressed_summary
        self._summary_version: int | None = None
        self._summary_message_count: int | None = None
        self._summary_timestamp: str | None = None
        # Set to True after maybe_compact fires so awrap_model_call can inject flush reminder.
        self._imminent = False
        # Set to True when threshold is exceeded; cleared after do_deferred_compact runs.
        self._needs_compaction = False
        # Per-session token-count caches shared with TokenBudgetEnforcer.
        self._token_cache: dict[str, int] = {}
        self._sys_token_cache: list[int] = []

    # --- State exposed to the middleware coordinator ---

    @property
    def compressed_summary(self) -> str:
        return self._compressed_summary

    @compressed_summary.setter
    def compressed_summary(self, value: str) -> None:
        self._compressed_summary = value

    @property
    def is_imminent(self) -> bool:
        """True when compaction threshold was crossed; cleared after flush reminder is sent."""
        return self._imminent

    def reset_imminent(self) -> None:
        self._imminent = False

    @property
    def needs_compaction(self) -> bool:
        """True when threshold was exceeded this turn; deferred compaction is pending."""
        return self._needs_compaction

    @property
    def token_cache(self) -> dict[str, int]:
        """Shared message-token cache — passed to TokenBudgetEnforcer to avoid re-encoding."""
        return self._token_cache

    @property
    def sys_token_cache(self) -> list[int]:
        """Cached system-message token total (length-1 list, empty = not yet cached)."""
        return self._sys_token_cache

    # --- Core compaction ---

    def maybe_compact(self, messages: list[BaseMessage]) -> tuple[bool, list[BaseMessage]]:
        """Detect whether compaction is needed and schedule it for post-turn execution.

        Compaction is deferred to ``do_deferred_compact`` so it runs after the
        response has been delivered to the user, keeping the current turn's
        critical path free of blocking LLM calls.  ``TokenBudgetEnforcer`` in
        ``awrap_model_call`` handles the token overflow for the current turn.

        Returns:
            Always (False, messages) — graph state is never mutated here.
        """
        if self._memory_manager is None:
            return False, messages

        rest = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(rest) <= self._keep_recent:
            return False, messages

        compactable = rest[: -self._keep_recent] if self._keep_recent > 0 else rest
        token_count = _count_tokens(compactable)

        if token_count <= self._compact_threshold:
            return False, messages

        self._imminent = True
        self._needs_compaction = True
        logger.info(
            "Memory compaction deferred: ~%d tokens (threshold: %d), %d messages pending compaction",
            token_count,
            self._compact_threshold,
            len(compactable),
        )
        return False, messages

    async def do_deferred_compact(self, messages: list[BaseMessage]) -> list[BaseMessage] | None:
        """Run the pending compaction on the final post-turn message list.

        Called by the middleware coordinator after the turn's response has been
        delivered.  Takes the complete final message list (including the new AI
        response) so the rolling summary is always up to date.

        Returns:
            The compacted message list (sys_msgs + recent) on success, or None
            if compaction was not needed or failed.
        """
        if not self._needs_compaction or self._memory_manager is None:
            return None

        self._needs_compaction = False

        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        rest = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(rest) <= self._keep_recent:
            return None

        compactable = rest[: -self._keep_recent] if self._keep_recent > 0 else rest
        recent = rest[-self._keep_recent :] if self._keep_recent > 0 else []
        token_count = _count_tokens(compactable)

        logger.info(
            "Running deferred compaction: ~%d tokens, compacting %d messages, keeping %d recent",
            token_count,
            len(compactable),
            len(recent),
        )

        _lf_span = None
        try:
            from langfuse import get_client as _lf_get_client

            _lf_span = _lf_get_client().span(
                name="memory_compaction",
                input={
                    "token_count": token_count,
                    "threshold": self._compact_threshold,
                    "messages_to_compact": len(compactable),
                    "messages_to_keep": len(recent),
                    "has_previous_summary": bool(self._compressed_summary),
                },
            )
        except Exception:
            pass

        try:
            self._memory_manager.add_async_summary_task(
                messages=compactable,
                channel=self._channel,
                session_id=self._session_id,
            )
            compact_content = await self._memory_manager.compact_memory(
                messages_to_summarize=compactable,
                previous_summary=self._compressed_summary,
                config=self._compaction_config,
            )
            self._compressed_summary = compact_content
            self._summary_timestamp = datetime.now(UTC).isoformat(timespec="seconds")
            self._summary_message_count = len(compactable) if compactable else None
            # Invalidate caches — compacted messages are gone.
            self._token_cache.clear()
            self._sys_token_cache.clear()
            logger.info("Deferred compaction done. Summary: %s...", compact_content[:100])

            if _lf_span is not None:
                with contextlib.suppress(Exception):
                    _lf_span.end(output={"summary_length": len(compact_content)})

            if self._memory is not None:
                compacted_ids = [
                    m.additional_kwargs.get("id", "")
                    for m in compactable
                    if "id" in getattr(m, "additional_kwargs", {})
                ]
                await self._memory.update_compressed_summary(
                    summary=compact_content,
                    message_ids=compacted_ids or None,
                    trigger_source="middleware",
                    tokens_before=token_count,
                )
                self._summary_version = self._memory._summary_version

            return sys_msgs + recent
        except Exception:
            if _lf_span is not None:
                with contextlib.suppress(Exception):
                    _lf_span.end(output={"error": "compaction failed"})
            logger.exception("Deferred compaction failed, full history will be saved")
            return None

    # --- System-prompt block helpers ---

    def get_summary_block(self, messages: list[BaseMessage]) -> str:
        """Return the compressed summary as a system-prompt block string."""
        if not self._compressed_summary:
            return ""

        summary = sanitize_summary_text(self._compressed_summary)
        if not summary:
            return ""
        # v2: summary block budget
        max_chars = int(os.environ.get("LIGHTCLAW_SUMMARY_BLOCK_MAX_CHARS", "12000"))
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n[... summary truncated ...]"
            logger.info(
                "Summary block truncated: original=%d chars, capped=%d chars",
                len(self._compressed_summary), max_chars,
            )
        self._compressed_summary = summary

        # Skip if summary marker already present — avoid duplicate injection.
        for m in messages:
            text = m.content if isinstance(m.content, str) else ""
            if isinstance(m, SystemMessage | HumanMessage) and (
                "<context-summary" in text or "<previous-summary>" in text
            ):
                return ""

        try:
            from lightclaw.agent.memory.context_assembly import build_summary_message

            summary_dict = build_summary_message(
                summary_text=summary,
                version=self._summary_version,
                messages_compressed=self._summary_message_count,
                compressed_at=self._summary_timestamp,
                language=self._language,
            )
            blocks = summary_dict.get("content", [])
            return blocks[0]["text"] if blocks and isinstance(blocks[0], dict) else summary
        except Exception:
            logger.debug("Failed to build summary block, using raw summary text")
            return summary

    def build_flush_reminder(self) -> str:
        """One-shot reminder for the agent to persist important info before compaction."""
        if self._language == "zh":
            return (
                "<memory-flush-reminder>\n"
                "上下文即将被压缩，较旧的消息将被摘要替代。\n"
                "如果对话历史中有尚未保存的重要信息，请现在用 write_file/edit_file 保存。\n"
                "重点关注：用户偏好、决策结论、项目上下文、性格反馈。\n"
                "</memory-flush-reminder>"
            )
        return (
            "<memory-flush-reminder>\n"
            "Context is about to be compressed. Older messages will be replaced by a summary.\n"
            "If there is anything important in the conversation history that you haven't\n"
            "already saved to memory files, save it now using write_file/edit_file.\n"
            "Focus on: user preferences, decisions, project context, and personality feedback.\n"
            "</memory-flush-reminder>"
        )
