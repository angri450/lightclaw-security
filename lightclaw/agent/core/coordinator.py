"""LangGraph middleware coordinator — assembles per-turn context before each LLM call.

Six single-responsibility steps are composed here in two hook methods:

``abefore_model`` (permanent state changes):
    ToolOutputTruncator → MessageSanitizer → MemoryCompactor

``awrap_model_call`` (ephemeral input shaping, no state mutation):
    ToolOutputTruncator → BootstrapInjector.get_block()
    → MemoryCompactor.get_summary_block
    → AutoRecallInjector → PersonalityBlockProvider
    → TokenBudgetEnforcer → MessageSanitizer → BrowserTaskGuard
    → merge extra blocks into system_message
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.factory import ExtendedModelResponse, ModelResponse
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from lightclaw.agent.core.engines.langgraph.middleware.message_sanitizer import sanitize_tool_messages
from lightclaw.agent.core.engines.langgraph.middleware.steps import (
    AutoRecallInjector,
    BootstrapInjector,
    BrowserTaskGuard,
    FactsBlockProvider,
    MemoryCompactor,
    PersonalityBlockProvider,
    TokenBudgetEnforcer,
)
from lightclaw.agent.core.interfaces import BaseMiddleware
from lightclaw.agent.utils.text_sanitizer import sanitize_messages
from lightclaw.agent.utils.token_counting import count_message_tokens as _count_tokens
from lightclaw.agent.utils.token_counting import count_str_tokens as _count_str_tokens
from lightclaw.agent.utils.token_counting import sanitize_text_payload, strip_model_special_tokens
from lightclaw.agent.utils.token_estimation import count_message_tool_call_tokens, count_tool_schema_tokens

if TYPE_CHECKING:
    from lightclaw.agent.memory import LightClawInMemoryMemory, MemoryManager
    from lightclaw.agent.memory.compaction_config import CompactionConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _VisibleTokenStats:
    """Token and count breakdown for a visible message list."""

    total_tokens: int
    estimated_total_tokens: int
    system_tokens: int
    non_system_tokens: int
    tool_tokens: int
    tool_call_tokens: int
    count: int
    split_by_type: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _TokenLogRecord:
    """Diagnostics snapshot for a single model call."""

    call_index: int
    max_input_length: int
    base_system_tokens: int
    extra_system_tokens: int
    final_system_tokens: int
    visible_pre: _VisibleTokenStats
    budget_post: _VisibleTokenStats
    visible_post: _VisibleTokenStats
    actual_total_pre_tokens: int
    actual_total_post_tokens: int
    over_budget_pre_tokens: int
    over_budget_post_tokens: int
    truncated_tool_messages: int
    truncated_tool_chars: int
    tool_schema_tokens: int
    tool_schema_count: int
    browser_guard_tokens: int
    provider_usage: dict[str, Any] | None = None
    flags: dict[str, bool] = field(default_factory=dict)
    extra_breakdown: dict[str, int] = field(default_factory=dict)


class LightClawMiddleware(BaseMiddleware, AgentMiddleware):
    """Middleware coordinator: composes 6 single-responsibility steps into a pipeline.

    Inherits from ``AgentMiddleware`` so it can be passed directly in the
    ``middleware`` sequence of ``create_agent``.

    Usage::

        mw = LightClawMiddleware(...)
        graph = create_agent(model=..., tools=..., middleware=[mw])
    """

    # AgentMiddleware expects a ``tools`` attribute.
    tools: Sequence = ()

    def __init__(
        self,
        *,
        working_dir: Path,
        language: str = "zh",
        memory_manager: MemoryManager | None = None,
        memory: LightClawInMemoryMemory | None = None,
        memory_compact_threshold: int = 90000,
        max_input_length: int = 131072,
        keep_recent: int = 3,
        compaction_config: CompactionConfig | None = None,
        fact_injector: Any | None = None,
    ) -> None:
        self._cancelled = False
        self._language = language

        self._bootstrap = BootstrapInjector(working_dir=working_dir, language=language)
        self._compactor = MemoryCompactor(
            memory_manager=memory_manager,
            memory=memory,
            compact_threshold=memory_compact_threshold,
            keep_recent=keep_recent,
            channel="",
            session_id="",
            compaction_config=compaction_config,
            compressed_summary="",
            language=language,
        )
        self._recall = AutoRecallInjector(memory_manager=memory_manager, language=language)
        self._personality = PersonalityBlockProvider(working_dir=working_dir, language=language)
        self._browser_guard = BrowserTaskGuard()
        self._token_budget = TokenBudgetEnforcer(max_input_length=max_input_length)
        self._facts_block = FactsBlockProvider(fact_injector=fact_injector)
        self._model_call_index = 0
        self._token_log_records: list[_TokenLogRecord] = []

    # --- Public API (unchanged for external callers) ---

    @property
    def compressed_summary(self) -> str:
        return self._compactor.compressed_summary

    @compressed_summary.setter
    def compressed_summary(self, value: str) -> None:
        self._compactor.compressed_summary = value

    def cancel(self, runtime_context: Any | None = None) -> None:
        """Signal cancellation — the next abefore_model call will raise CancelledError."""
        if runtime_context is not None:
            try:
                if isinstance(runtime_context, dict):
                    runtime_context["cancelled"] = True
                else:
                    runtime_context.cancelled = True
                return
            except Exception:
                logger.debug("Failed to set runtime_context.cancelled, falling back to middleware flag", exc_info=True)
        self._cancelled = True

    @property
    def needs_compaction(self) -> bool:
        """True when the token threshold was exceeded this turn and deferred compaction is pending."""
        return self._compactor.needs_compaction

    def needs_compaction_for(self, runtime_context: Any | None = None) -> bool:
        """Return deferred-compaction status for the current runtime context."""
        return self._resolve_compactor(runtime_context).needs_compaction

    async def do_deferred_compaction(
        self,
        messages: list[BaseMessage],
        *,
        runtime_context: Any | None = None,
    ) -> list[BaseMessage] | None:
        """Run post-turn compaction on the final message list.

        Returns the compacted message list, or None if compaction was not needed or failed.
        """
        compactor = self._resolve_compactor(runtime_context)
        compacted = await compactor.do_deferred_compact(messages)
        self._sync_summary_to_context(runtime_context, compactor.compressed_summary)
        return compacted

    def commit_bootstrap_completion(self, *, turn_id: str = "", session_id: str = "") -> None:
        """Persist the bootstrap completion flag after a successful run."""
        state_before = self._bootstrap.debug_state()
        if (
            state_before.get("pending")
            or state_before.get("bootstrap_file_exists")
            or state_before.get("bootstrap_completed")
        ):
            logger.info(
                "[Bootstrap][lifecycle][commit_request] turn=%s session=%s state_before=%s",
                turn_id or "-",
                session_id or "-",
                state_before,
            )
        self._bootstrap.commit()
        state_after = self._bootstrap.debug_state()
        if (
            state_before.get("pending")
            or state_before.get("bootstrap_file_exists")
            or state_before.get("bootstrap_completed")
            or state_after.get("bootstrap_completed")
        ):
            logger.info(
                "[Bootstrap][lifecycle][commit_done] turn=%s session=%s state_after=%s",
                turn_id or "-",
                session_id or "-",
                state_after,
            )

    def ensure_weibo_onboarding(self, *, turn_id: str = "", session_id: str = "") -> None:
        """Unconditionally attempt to arm the Weibo onboarding hook.

        Called from the run controller's ``finally`` block so it runs on
        every turn regardless of success/failure.
        """
        state_before = self._bootstrap.debug_state()
        weibo_before = state_before.get("weibo_onboarding", {})
        if (
            state_before.get("bootstrap_completed")
            or state_before.get("bootstrap_file_exists")
            or weibo_before.get("armed_flag_exists")
            or weibo_before.get("identity_exists")
        ):
            logger.info(
                "[Bootstrap][lifecycle][ensure_weibo_request] turn=%s session=%s state_before=%s",
                turn_id or "-",
                session_id or "-",
                state_before,
            )
        self._bootstrap.ensure_weibo_onboarding()
        state_after = self._bootstrap.debug_state()
        weibo_after = state_after.get("weibo_onboarding", {})
        if (
            state_before != state_after
            or state_after.get("bootstrap_completed")
            or weibo_after.get("armed_flag_exists")
            or weibo_after.get("identity_exists")
        ):
            logger.info(
                "[Bootstrap][lifecycle][ensure_weibo_done] turn=%s session=%s state_after=%s",
                turn_id or "-",
                session_id or "-",
                state_after,
            )

    def _maybe_get_bootstrap_block(self, messages: list[BaseMessage]) -> str | None:
        """Return bootstrap guidance block for the current turn, if applicable."""
        return self._bootstrap.get_block(messages)

    @staticmethod
    def _runtime_context_from_runtime(runtime: Any | None) -> Any | None:
        if runtime is None:
            return None
        return getattr(runtime, "context", None)

    @classmethod
    def _runtime_context_from_request(cls, request: Any) -> Any | None:
        return cls._runtime_context_from_runtime(getattr(request, "runtime", None))

    @staticmethod
    def _context_value(context: Any | None, attr: str, default: Any = "") -> Any:
        if context is None:
            return default
        if isinstance(context, dict):
            return context.get(attr, default)
        return getattr(context, attr, default)

    @staticmethod
    def _set_context_value(context: Any | None, attr: str, value: Any) -> None:
        if context is None:
            return
        if isinstance(context, dict):
            context[attr] = value
            return
        try:
            setattr(context, attr, value)
        except Exception:
            logger.debug("Failed to set runtime context field %s", attr, exc_info=True)

    @classmethod
    def _runtime_ids(cls, context: Any | None) -> tuple[str, str]:
        turn_id = str(cls._context_value(context, "turn_id", "") or "")
        session_id = str(cls._context_value(context, "session_id", "") or "")
        return turn_id, session_id

    def _resolve_compactor(self, runtime_context: Any | None) -> MemoryCompactor:
        compactor = self._context_value(runtime_context, "compactor", None)
        if isinstance(compactor, MemoryCompactor):
            return compactor
        return self._compactor

    def _sync_summary_to_context(self, runtime_context: Any | None, summary: str) -> None:
        if runtime_context is not None:
            self._set_context_value(runtime_context, "compressed_summary", summary)

    def _token_log_records_for_context(self, runtime_context: Any | None) -> list[_TokenLogRecord]:
        records = self._context_value(runtime_context, "token_log_records", None)
        if isinstance(records, list):
            return records
        return self._token_log_records

    def _model_call_index_for_context(self, runtime_context: Any | None) -> int:
        value = self._context_value(runtime_context, "model_call_index", self._model_call_index)
        try:
            return int(value)
        except Exception:
            return self._model_call_index

    def _set_model_call_index_for_context(self, runtime_context: Any | None, value: int) -> None:
        if runtime_context is not None:
            self._set_context_value(runtime_context, "model_call_index", value)
            return
        self._model_call_index = value

    def _is_cancelled(self, runtime_context: Any | None) -> bool:
        return bool(self._context_value(runtime_context, "cancelled", False) or self._cancelled)

    def get_token_log_summary(self, runtime_context: Any | None = None) -> dict[str, Any] | None:
        """Return a turn-level summary of per-call token diagnostics."""
        records = self._token_log_records_for_context(runtime_context)
        if not records:
            return None

        first = records[0]
        return {
            "model_calls": len(records),
            "first_call_actual_tokens": first.actual_total_post_tokens,
            "estimated_input_tokens": sum(record.actual_total_post_tokens for record in records),
            "tool_schema_tokens": sum(record.tool_schema_tokens for record in records),
            "max_actual_tokens": max(record.actual_total_post_tokens for record in records),
            "max_over_budget": max(record.over_budget_post_tokens for record in records),
            "bootstrap_first_call": first.flags.get("bootstrap", False),
            "later_tool_result_seen": any(
                record.call_index > 1 and record.visible_pre.tool_tokens > 0 for record in records
            ),
            "later_large_tool_result_seen": any(
                record.call_index > 1 and record.truncated_tool_messages > 0 for record in records
            ),
        }

    def _log_bootstrap_stage(
        self,
        *,
        turn_id: str,
        session_id: str,
        call_index: int,
        messages: list[BaseMessage],
        bootstrap_block: str | None,
        weibo_onboarding_block: str | None,
    ) -> None:
        """Emit a focused DEBUG log for bootstrap-related context shaping."""
        bootstrap_state = self._bootstrap.debug_state()
        weibo_state = bootstrap_state.get("weibo_onboarding", {})
        last_message_type, last_message_preview = self._last_visible_message_snapshot(messages)
        logger.debug(
            "[Bootstrap][middleware] turn=%s session=%s call_index=%d messages=%d has_human=%s "
            "bootstrap_block=%s weibo_block=%s pending=%s completed=%s bootstrap_file=%s "
            "weibo_state=%s last_message_type=%s last_message=%r",
            turn_id or "-",
            session_id or "-",
            call_index,
            len(messages),
            any(isinstance(message, HumanMessage) for message in messages),
            bool(bootstrap_block),
            bool(weibo_onboarding_block),
            bootstrap_state.get("pending", False),
            bootstrap_state.get("bootstrap_completed", False),
            bootstrap_state.get("bootstrap_file_exists", False),
            weibo_state,
            last_message_type,
            last_message_preview,
        )

    @classmethod
    def _log_bootstrap_resolution(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_index: int,
        messages: list[BaseMessage],
        bootstrap_block: str | None,
        weibo_onboarding_block: str | None,
        bootstrap_state: dict[str, object],
    ) -> None:
        """Emit an INFO log when bootstrap-related dynamic blocks are resolved."""
        last_message_type, last_message_preview = cls._last_visible_message_snapshot(messages)
        logger.info(
            "[Bootstrap][stage][resolved] turn=%s session=%s call_index=%d bootstrap_block=%s "
            "bootstrap_chars=%d weibo_block=%s weibo_chars=%d pending=%s completed=%s "
            "bootstrap_file=%s last_message_type=%s last_message=%r",
            turn_id or "-",
            session_id or "-",
            call_index,
            bool(bootstrap_block),
            len(bootstrap_block or ""),
            bool(weibo_onboarding_block),
            len(weibo_onboarding_block or ""),
            bootstrap_state.get("pending", False),
            bootstrap_state.get("bootstrap_completed", False),
            bootstrap_state.get("bootstrap_file_exists", False),
            last_message_type,
            last_message_preview,
        )

    @classmethod
    def _log_bootstrap_merge(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_id: str,
        call_index: int,
        merge_mode: str,
        bootstrap_state: dict[str, object],
        flags: dict[str, bool],
        final_system_chars: int,
        messages: list[BaseMessage],
    ) -> None:
        """Emit an INFO log when bootstrap context is merged into the request."""
        last_message_type, last_message_preview = cls._last_visible_message_snapshot(messages)
        logger.info(
            "[Middleware][onboarding_merged] turn=%s session=%s call_id=%s call_index=%d merge_mode=%s "
            "active_blocks=%s final_system_chars=%d pending=%s completed=%s bootstrap_file=%s "
            "weibo_state=%s last_message_type=%s last_message=%r",
            turn_id or "-",
            session_id or "-",
            call_id,
            call_index,
            merge_mode,
            cls._format_active_blocks(flags),
            final_system_chars,
            bootstrap_state.get("pending", False),
            bootstrap_state.get("bootstrap_completed", False),
            bootstrap_state.get("bootstrap_file_exists", False),
            bootstrap_state.get("weibo_onboarding", {}),
            last_message_type,
            last_message_preview,
        )

    @classmethod
    def _log_io_step_start(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_index: int,
        step: str,
    ) -> None:
        """Emit an INFO log immediately before a middleware IO step begins."""
        frame = inspect.stack()[1]
        caller_loc = f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}"
        logger.info(
            "[Middleware][io_start] turn=%s session=%s call_index=%d step=%s caller=%s",
            turn_id or "-",
            session_id or "-",
            call_index,
            step,
            caller_loc,
        )

    @classmethod
    def _log_io_step(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_index: int,
        step: str,
        duration_ms: float,
        result: str | None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Emit an INFO log for middleware steps that may block on IO."""
        frame = inspect.stack()[1]
        caller_loc = f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}"
        logger.info(
            "[Middleware][io] turn=%s session=%s call_index=%d step=%s duration_ms=%.2f "
            "result_present=%s result_chars=%d caller=%s details=%s",
            turn_id or "-",
            session_id or "-",
            call_index,
            step,
            duration_ms,
            bool(result),
            len(result or ""),
            caller_loc,
            details or {},
        )

    @classmethod
    def _log_context_ready(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_index: int,
        messages: list[BaseMessage],
        flags: dict[str, bool],
        extra_breakdown: dict[str, int],
        extra_char_counts: dict[str, int],
        base_system_chars: int,
        tool_schema_count: int,
        tool_schema_tokens: int,
    ) -> None:
        """Emit an INFO log after dynamic context blocks are assembled."""
        last_message_type, last_message_preview = cls._last_visible_message_snapshot(messages)
        logger.info(
            "[Middleware][context_ready] turn=%s session=%s call_index=%d active_blocks=%s "
            "extra_tokens=%s extra_chars=%s base_system_chars=%d tool_schema_count=%d "
            "tool_schema_tokens=%d last_message_type=%s last_message=%r",
            turn_id or "-",
            session_id or "-",
            call_index,
            cls._format_active_blocks(flags),
            extra_breakdown,
            extra_char_counts,
            base_system_chars,
            tool_schema_count,
            tool_schema_tokens,
            last_message_type,
            last_message_preview,
        )

    @classmethod
    def _log_request_ready(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_id: str,
        call_index: int,
        merge_mode: str,
        visible_pre: _VisibleTokenStats,
        budget_post: _VisibleTokenStats,
        visible_post: _VisibleTokenStats,
        final_system_chars: int,
        browser_guard_tokens: int,
        actual_total_pre_tokens: int,
        actual_total_post_tokens: int,
        over_budget_pre_tokens: int,
        over_budget_post_tokens: int,
        flags: dict[str, bool],
        messages: list[BaseMessage],
    ) -> None:
        """Emit an INFO log after the final model request payload is shaped."""
        last_message_type, last_message_preview = cls._last_visible_message_snapshot(messages)
        logger.info(
            "[Middleware][request_ready] turn=%s session=%s call_id=%s call_index=%d merge_mode=%s "
            "active_blocks=%s visible_pre=%d budget_post=%d visible_post=%d final_system_chars=%d "
            "browser_guard_tokens=%d actual_pre_tokens=%d actual_post_tokens=%d "
            "over_budget_pre=%d over_budget_post=%d last_message_type=%s last_message=%r",
            turn_id or "-",
            session_id or "-",
            call_id,
            call_index,
            merge_mode,
            cls._format_active_blocks(flags),
            visible_pre.estimated_total_tokens,
            budget_post.estimated_total_tokens,
            visible_post.estimated_total_tokens,
            final_system_chars,
            browser_guard_tokens,
            actual_total_pre_tokens,
            actual_total_post_tokens,
            over_budget_pre_tokens,
            over_budget_post_tokens,
            last_message_type,
            last_message_preview,
        )

    # --- LangGraph AgentMiddleware hooks ---

    async def abefore_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        """Called before every LLM invocation inside the ReAct loop.

        Handles operations that may permanently rewrite the graph state
        (compaction).  Ephemeral input shaping happens in awrap_model_call.

        Returns:
            A state-update dict when compaction rewrites the message history,
            or None for ordinary turns.
        """
        runtime_context = self._runtime_context_from_runtime(runtime)
        if self._is_cancelled(runtime_context):
            raise asyncio.CancelledError("Agent interrupted by user")

        compactor = self._resolve_compactor(runtime_context)
        context_summary = self._context_value(runtime_context, "compressed_summary", None)
        if isinstance(context_summary, str):
            compactor.compressed_summary = context_summary

        messages: list[BaseMessage] = list(state.get("messages", []))
        messages, _ = self._token_budget.truncate_tool_outputs_with_stats(messages)
        messages = sanitize_tool_messages(messages)

        compacted, messages = compactor.maybe_compact(messages)
        logger.debug(
            "Middleware before_model: messages=%d compacted=%s needs_compaction=%s",
            len(messages),
            compacted,
            compactor.needs_compaction,
        )
        self._sync_summary_to_context(runtime_context, compactor.compressed_summary)
        if compacted:
            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]}
        return None

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Shape the effective LLM input without mutating graph state.

        Assembles all context blocks (facts, summary, recall, personality) and
        merges them into the system message so that providers that reject
        multiple system messages receive exactly one.
        """
        runtime_context = self._runtime_context_from_request(request)
        if self._is_cancelled(runtime_context):
            raise asyncio.CancelledError("Agent interrupted by user")

        turn_id, session_id = self._runtime_ids(runtime_context)
        compactor = self._resolve_compactor(runtime_context)
        context_summary = self._context_value(runtime_context, "compressed_summary", None)
        if isinstance(context_summary, str):
            compactor.compressed_summary = context_summary
        token_log_records = self._token_log_records_for_context(runtime_context)
        model_call_index = self._model_call_index_for_context(runtime_context)

        messages: list[BaseMessage] = list(request.messages)
        next_call_index = model_call_index + 1
        messages, truncation_stats = self._token_budget.truncate_tool_outputs_with_stats(messages)
        self._log_io_step_start(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="bootstrap.get_block",
        )
        step_started_at = time.perf_counter()
        bootstrap_block = self._bootstrap.get_block(messages)
        bootstrap_duration_ms = (time.perf_counter() - step_started_at) * 1000
        self._log_io_step(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="bootstrap.get_block",
            duration_ms=bootstrap_duration_ms,
            result=bootstrap_block,
            details=self._bootstrap.debug_state(),
        )
        self._log_io_step_start(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="bootstrap.weibo_onboarding.get_block",
        )
        step_started_at = time.perf_counter()
        weibo_onboarding_block = self._bootstrap.weibo_onboarding.get_block(messages)
        weibo_duration_ms = (time.perf_counter() - step_started_at) * 1000
        bootstrap_state = self._bootstrap.debug_state()
        self._log_io_step(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="bootstrap.weibo_onboarding.get_block",
            duration_ms=weibo_duration_ms,
            result=weibo_onboarding_block,
            details=bootstrap_state.get("weibo_onboarding", {}),
        )
        self._log_bootstrap_stage(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            messages=messages,
            bootstrap_block=bootstrap_block,
            weibo_onboarding_block=weibo_onboarding_block,
        )
        if (
            bootstrap_block
            or weibo_onboarding_block
            or bootstrap_state.get("pending")
            or bootstrap_state.get("bootstrap_file_exists")
            or bootstrap_state.get("bootstrap_completed")
        ):
            self._log_bootstrap_resolution(
                turn_id=turn_id,
                session_id=session_id,
                call_index=next_call_index,
                messages=messages,
                bootstrap_block=bootstrap_block,
                weibo_onboarding_block=weibo_onboarding_block,
                bootstrap_state=bootstrap_state,
            )

        # Collect extra context as strings; merge into system_message at the end
        # to guarantee a single system message in the final payload.
        extra_sys_parts: list[str] = []
        extra_parts_map: dict[str, str] = {
            "bootstrap": "",
            "weibo": "",
            "facts": "",
            "summary": "",
            "recall": "",
            "personality": "",
            "flush_reminder": "",
            "memory_write_reminder": "",
        }
        if bootstrap_block:
            extra_sys_parts.append(bootstrap_block)
            extra_parts_map["bootstrap"] = bootstrap_block
        if weibo_onboarding_block:
            extra_sys_parts.append(weibo_onboarding_block)
            extra_parts_map["weibo"] = weibo_onboarding_block

        self._log_io_step_start(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="facts_block.get_block",
        )
        step_started_at = time.perf_counter()
        facts_block = await self._facts_block.get_block(messages)
        facts_duration_ms = (time.perf_counter() - step_started_at) * 1000
        self._log_io_step(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="facts_block.get_block",
            duration_ms=facts_duration_ms,
            result=facts_block,
        )
        if facts_block:
            extra_sys_parts.append(facts_block)
            extra_parts_map["facts"] = facts_block

        summary = compactor.get_summary_block(messages)
        self._sync_summary_to_context(runtime_context, compactor.compressed_summary)
        if summary:
            extra_sys_parts.append(summary)
            extra_parts_map["summary"] = summary

        self._log_io_step_start(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="recall.get_block",
        )
        step_started_at = time.perf_counter()
        # Build ActorContext for this turn — prefer runtime_context (from channel adapter).
        # Fallback policy: only dashboard channel gets dashboard_owner;
        # everything else gets unknown (minimal privileges, no owner_memory access).
        actor_ctx = getattr(runtime_context, 'actor_context', None)
        if actor_ctx is None:
            from lightclaw.agent.core.actor_context import ActorContext

            _fallback_channel = getattr(self, '_channel', '') or ''
            if _fallback_channel == 'dashboard':
                actor_ctx = ActorContext.dashboard_owner(
                    session_id=session_id,
                    agent_id=getattr(self, '_agent_id', 'default'),
                )
                logger.info(
                    "[ActorContext][fallback] reason=dashboard_legacy "
                    "fallback_trust=owner session_id=%s",
                    session_id,
                )
            else:
                actor_ctx = ActorContext.unknown(
                    session_id=session_id,
                    channel=_fallback_channel,
                    agent_id=getattr(self, '_agent_id', 'default'),
                )
                logger.warning(
                    "[ActorContext][fallback] reason=missing_actor_context "
                    "fallback_trust=unknown channel=%s session_id=%s",
                    _fallback_channel or 'unknown',
                    session_id,
                )
        recall = await self._recall.get_block(messages, actor_context=actor_ctx)
        recall_duration_ms = (time.perf_counter() - step_started_at) * 1000
        self._log_io_step(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="recall.get_block",
            duration_ms=recall_duration_ms,
            result=recall,
            details={
                "bootstrap_pending": getattr(self._recall, "_bootstrap_pending", False),
                "memory_md_enabled": getattr(self._recall, "_memory_md_injection_enabled", False),
                "daily_talk_enabled": getattr(self._recall, "_daily_talk_injection_enabled", False),
                "semantic_recall_enabled": getattr(self._recall, "_enabled", False),
            },
        )
        if recall:
            extra_sys_parts.append(recall)
            extra_parts_map["recall"] = recall

        self._log_io_step_start(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="personality.get_block",
        )
        step_started_at = time.perf_counter()
        personality = self._personality.get_block(messages)
        personality_duration_ms = (time.perf_counter() - step_started_at) * 1000
        self._log_io_step(
            turn_id=turn_id,
            session_id=session_id,
            call_index=next_call_index,
            step="personality.get_block",
            duration_ms=personality_duration_ms,
            result=personality,
            details={"injector_available": getattr(self._personality, "_injector", None) is not None},
        )
        if personality:
            extra_sys_parts.append(personality)
            extra_parts_map["personality"] = personality

        flush_reminder_injected = False
        if compactor.is_imminent:
            flush_reminder = compactor.build_flush_reminder()
            extra_sys_parts.append(flush_reminder)
            extra_parts_map["flush_reminder"] = flush_reminder
            compactor.reset_imminent()
            flush_reminder_injected = True

        # Append a lightweight memory-write reminder at the tail of the system
        # prompt on user-message turns only.  Skipped on tool-call intermediate
        # turns (where the last non-system message is an AIMessage/ToolMessage)
        # to avoid interrupting the tool chain.  Also skipped when a
        # flush-reminder was already injected — their content overlaps.
        # Additionally skipped when the last user message is too short or
        # contains no letters/digits — such messages (e.g. "ok", "👍", "继续")
        # almost never warrant a memory write, so injecting the reminder wastes
        # tokens without any benefit.
        if (
            not flush_reminder_injected
            and self._is_user_turn(messages)
            and self._last_user_message_is_meaningful(messages)
        ):
            memory_write_reminder = self._build_memory_write_reminder()
            extra_sys_parts.append(memory_write_reminder)
            extra_parts_map["memory_write_reminder"] = memory_write_reminder

        call_index = model_call_index + 1
        self._set_model_call_index_for_context(runtime_context, call_index)
        existing_system_blocks = self._extract_request_system_blocks(request)
        existing_system_text = self._extract_request_system_text(request)
        visible_pre = self._summarize_visible_tokens(messages)
        base_system_chars = len(existing_system_text or "")
        tool_schema_count = len(getattr(request, "tools", []) or [])
        tool_schema_tokens = count_tool_schema_tokens(getattr(request, "tools", []) or [])
        extra_parts_map = self._clean_extra_parts_map(extra_parts_map)
        # ── P2a: heartbeat context slimming ──────────────────────────────────
        # Heartbeat agent only needs the sensor summary in its prompt.  Skip
        # recall_block, memory_write_reminder, and flush_reminder to save tokens.
        _profile = getattr(runtime_context, "runtime_profile", None)
        if _profile in ("heartbeat", "readonly_worker"):
            removed: list[str] = []
            for _key in ("recall", "memory_write_reminder", "flush_reminder"):
                if extra_parts_map.get(_key):
                    extra_parts_map[_key] = ""
                    removed.append(_key)
            if removed:
                extra_parts_map = self._clean_extra_parts_map(extra_parts_map)
                logger.info(
                    "heartbeat_context_slimming: removed_blocks=%s",
                    ",".join(removed),
                )
        extra_sys_parts = [v for v in extra_parts_map.values() if isinstance(v, str) and v]
        extra_breakdown = {name: _count_str_tokens(text) if text else 0 for name, text in extra_parts_map.items()}
        extra_char_counts = {name: len(text) if text else 0 for name, text in extra_parts_map.items()}
        extra_system_tokens = sum(extra_breakdown.values())
        actual_total_pre_tokens = (
            visible_pre.estimated_total_tokens
            + (_count_str_tokens(existing_system_text) if existing_system_text else 0)
            + extra_system_tokens
            + tool_schema_tokens
        )
        over_budget_pre_tokens = max(actual_total_pre_tokens - self._token_budget.max_input_length, 0)
        # ── P2a: update flags for heartbeat-slimmed blocks ──────────────────
        if _profile == "heartbeat":
            recall = ""  # type: ignore[assignment]
            flush_reminder_injected = False
        flags = {
            "bootstrap": bool(bootstrap_block),
            "weibo": bool(weibo_onboarding_block),
            "facts": bool(facts_block),
            "summary": bool(summary),
            "recall": bool(recall),
            "personality": bool(personality),
            "flush_reminder": flush_reminder_injected,
            "memory_write_reminder": bool(extra_parts_map["memory_write_reminder"]),
            "browser_guard": False,
        }
        self._log_context_ready(
            turn_id=turn_id,
            session_id=session_id,
            call_index=call_index,
            messages=messages,
            flags=flags,
            extra_breakdown=extra_breakdown,
            extra_char_counts=extra_char_counts,
            base_system_chars=base_system_chars,
            tool_schema_count=tool_schema_count,
            tool_schema_tokens=tool_schema_tokens,
        )
        self._log_token_preflight(
            call_index=call_index,
            turn_id=turn_id,
            session_id=session_id,
            base_system_tokens=_count_str_tokens(existing_system_text) if existing_system_text else 0,
            extra_system_tokens=extra_system_tokens,
            visible_pre=visible_pre,
            actual_total_pre_tokens=actual_total_pre_tokens,
            over_budget_pre_tokens=over_budget_pre_tokens,
            truncation_stats=truncation_stats,
            tool_schema_tokens=tool_schema_tokens,
            tool_schema_count=tool_schema_count,
            flags=flags,
        )

        budget_messages = self._token_budget.enforce(
            messages,
            token_cache=compactor.token_cache,
            sys_token_cache=compactor.sys_token_cache,
            reserved_tokens=actual_total_pre_tokens - visible_pre.estimated_total_tokens,
        )
        budget_post = self._summarize_visible_tokens(budget_messages)
        # Final sanitisation after token budget may have dropped messages.
        messages = sanitize_tool_messages(budget_messages)
        messages = sanitize_messages(messages)
        before_browser_messages = list(messages)
        messages = self._browser_guard.inject(messages)
        browser_guard_tokens = (
            _count_tokens(messages[len(before_browser_messages) :])
            if len(messages) > len(before_browser_messages)
            else 0
        )
        flags["browser_guard"] = browser_guard_tokens > 0
        visible_post = self._summarize_visible_tokens(messages)

        override_kwargs: dict = {"messages": messages}
        final_system_text = existing_system_text
        merge_mode = "unchanged"
        if extra_sys_parts and hasattr(request, "system_message") and request.system_message is not None:
            merged_blocks = list(existing_system_blocks)
            merged_blocks.extend({"type": "text", "text": part} for part in extra_sys_parts if part)
            override_kwargs["system_message"] = SystemMessage(content=merged_blocks)
            final_system_text = self._extract_text_from_content(merged_blocks)
            merge_mode = "override_system_message_blocks"
        elif extra_sys_parts:
            # No request.system_message — fall back to prepending in messages list.
            messages = [SystemMessage(content="\n\n".join(extra_sys_parts)), *messages]
            override_kwargs["messages"] = messages
            visible_post = self._summarize_visible_tokens(messages)
            final_system_text = ""
            merge_mode = "prepend_system_message"

        final_system_tokens = _count_str_tokens(final_system_text) if final_system_text else 0
        call_id = self._build_model_call_id(call_index, turn_id=turn_id)
        record = _TokenLogRecord(
            call_index=call_index,
            max_input_length=self._token_budget.max_input_length,
            base_system_tokens=_count_str_tokens(existing_system_text) if existing_system_text else 0,
            extra_system_tokens=extra_system_tokens,
            final_system_tokens=final_system_tokens,
            visible_pre=visible_pre,
            budget_post=budget_post,
            visible_post=visible_post,
            actual_total_pre_tokens=actual_total_pre_tokens,
            actual_total_post_tokens=visible_post.estimated_total_tokens + final_system_tokens + tool_schema_tokens,
            over_budget_pre_tokens=over_budget_pre_tokens,
            over_budget_post_tokens=max(
                (visible_post.estimated_total_tokens + final_system_tokens + tool_schema_tokens)
                - self._token_budget.max_input_length,
                0,
            ),
            truncated_tool_messages=truncation_stats.truncated_messages,
            truncated_tool_chars=truncation_stats.truncated_chars,
            tool_schema_tokens=tool_schema_tokens,
            tool_schema_count=tool_schema_count,
            browser_guard_tokens=browser_guard_tokens,
            flags=flags,
            extra_breakdown=extra_breakdown,
        )
        token_log_records.append(record)
        self._sync_summary_to_context(runtime_context, compactor.compressed_summary)
        bootstrap_state = self._bootstrap.debug_state()
        if (
            flags.get("bootstrap")
            or flags.get("weibo")
            or bootstrap_state.get("pending")
            or bootstrap_state.get("bootstrap_file_exists")
        ):
            self._log_bootstrap_merge(
                turn_id=turn_id,
                session_id=session_id,
                call_id=call_id,
                call_index=call_index,
                merge_mode=merge_mode,
                bootstrap_state=bootstrap_state,
                flags=flags,
                final_system_chars=len(final_system_text or ""),
                messages=messages,
            )
        self._log_request_ready(
            turn_id=turn_id,
            session_id=session_id,
            call_id=call_id,
            call_index=call_index,
            merge_mode=merge_mode,
            visible_pre=visible_pre,
            budget_post=budget_post,
            visible_post=visible_post,
            final_system_chars=len(final_system_text or ""),
            browser_guard_tokens=browser_guard_tokens,
            actual_total_pre_tokens=actual_total_pre_tokens,
            actual_total_post_tokens=record.actual_total_post_tokens,
            over_budget_pre_tokens=over_budget_pre_tokens,
            over_budget_post_tokens=record.over_budget_post_tokens,
            flags=flags,
            messages=messages,
        )
        response: Any = None
        model_call_started_at = time.monotonic()
        # Inject per-turn model from runtime context when one was routed
        # (e.g. a vision model selected because the request contains images).
        # Falls back to the graph's default model when turn_model is absent.
        turn_model = self._context_value(runtime_context, "turn_model", None)
        if turn_model is not None:
            override_kwargs["model"] = turn_model
        effective_model = turn_model or getattr(request, "model", None)
        effective_tools = list(getattr(request, "tools", []) or [])
        effective_model_settings = dict(getattr(request, "model_settings", {}) or {})
        # ── P1b: tool whitelist filtering based on runtime_context.allowed_tools ─
        allowed_tools: list[str] | None = self._context_value(runtime_context, "allowed_tools", None)
        tool_policy_source = self._context_value(runtime_context, "tool_policy_source", None)
        if allowed_tools is not None:
            before_count = len(effective_tools)
            allowed_set = set(allowed_tools)
            filtered_tools = [t for t in effective_tools if getattr(t, "name", "") in allowed_set]
            blocked_count = before_count - len(filtered_tools)
            effective_tools = filtered_tools
            override_kwargs["tools"] = effective_tools
            logger.info(
                "tool_policy: source=%s allowed=%d before=%d after=%d blocked=%d",
                tool_policy_source or "unknown",
                len(allowed_tools), before_count, len(effective_tools), blocked_count,
            )
        self._log_model_call_start(
            turn_id=turn_id,
            session_id=session_id,
            call_id=call_id,
            call_index=call_index,
            model=effective_model,
            messages=messages,
            system_text=final_system_text,
            tools=effective_tools,
            flags=flags,
            model_settings=effective_model_settings,
        )
        # ── v3: multi-stage final input guard ──────────────────────────────
        # Stage 0: prepare helpers
        from lightclaw.agent.utils.token_counting import count_message_tokens as _final_count
        from lightclaw.constant import (
            EXTRA_PARTS_MAX_TOTAL_CHARS,
            FINAL_INPUT_GUARD_HARD_FAIL,
            FINAL_INPUT_MAX_TRIM_ROUNDS,
        )
        import re as _guard_re

        max_input = self._token_budget.max_input_length
        # ── P1b: heartbeat runtime budget override ──────────────────────────
        runtime_profile = self._context_value(runtime_context, "runtime_profile", None)
        max_input_override = self._context_value(runtime_context, "max_input_length_override", None)
        if max_input_override is not None and isinstance(max_input_override, int) and max_input_override > 0:
            max_input = max_input_override
            logger.debug(
                "final_input_guard: budget_override profile=%s max=%d (shared_default=%d)",
                runtime_profile or "unknown", max_input, self._token_budget.max_input_length,
            )
        trim_actions: list[str] = []
        trim_round = 0

        def _recount() -> int:
            return _final_count(messages)

        # Find the last HumanMessage (must be preserved)
        def _last_human_idx() -> int | None:
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    return i
            return None

        # ── Stage 1: Guard against empty non-system messages ──
        final_total = _recount()
        final_non_sys = _final_count([m for m in messages if not isinstance(m, SystemMessage)])
        lidx = _last_human_idx()

        if final_non_sys == 0 and lidx is not None:
            logger.warning("final_input_guard: all_non_sys_dropped — keeping last HumanMessage")
            messages = [m for m in messages if isinstance(m, SystemMessage)] + [messages[lidx]]
            trim_actions.append("keep_last_human")
            final_total = _recount()

        # ── Stage 2: Multi-round trimming if over max_input_length ──
        while final_total > max_input and trim_round < FINAL_INPUT_MAX_TRIM_ROUNDS:
            trim_round += 1
            before = final_total

            # 2a: Trim recall_block from system message (hardest hitter)
            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
                    if "<auto-recall-context>" in msg.content:
                        msg.content = _guard_re.sub(
                            r"<auto-recall-context>.*?</auto-recall-context>",
                            '<auto-recall-context>\n[recall trimmed by budget guard]\n</auto-recall-context>',
                            msg.content,
                            flags=_guard_re.DOTALL,
                        )
                        trim_actions.append("trim_recall")
                        break

            # 2b: Compress large extra system parts (summary, flush_reminder, memory_write_reminder)
            extra_total_chars = sum(len(p) for p in extra_sys_parts) if extra_sys_parts else 0
            if extra_total_chars > EXTRA_PARTS_MAX_TOTAL_CHARS:
                # Replace non-essential blocks with short placeholders
                non_essential = {"summary", "flush_reminder", "memory_write_reminder", "recall"}
                for i, msg in enumerate(messages):
                    if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
                        for block_name in non_essential:
                            tag = f"<{block_name}>" if block_name != "recall" else "<auto-recall-context>"
                            close = f"</{block_name}>" if block_name != "recall" else "</auto-recall-context>"
                            if tag in msg.content:
                                msg.content = _guard_re.sub(
                                    f"{_guard_re.escape(tag)}.*?{_guard_re.escape(close)}",
                                    f"{tag}\n[{block_name} trimmed by budget guard]\n{close}",
                                    msg.content,
                                    flags=_guard_re.DOTALL,
                                )
                        trim_actions.append("trim_extra_sys_parts")
                        break

            # 2c: Drop oldest non-system messages (preserve last HumanMessage)
            non_sys_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
            sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
            if len(non_sys_msgs) > 1:
                lidx2 = _last_human_idx()
                # Drop oldest ~20% of non-system messages, but never the last HumanMessage
                drop_count = max(1, len(non_sys_msgs) // 5)
                kept_non_sys = []
                for idx, m in enumerate(non_sys_msgs):
                    orig_idx = len(sys_msgs) + idx
                    is_last_human = (lidx2 is not None and orig_idx == lidx2)
                    is_in_drop_window = idx < drop_count
                    if is_in_drop_window and not is_last_human:
                        # Replace old ToolMessage with short placeholder instead of deleting
                        if isinstance(m, ToolMessage):
                            if hasattr(m, "model_copy"):
                                short = m.model_copy(deep=True)
                            elif hasattr(m, "copy"):
                                short = m.copy(deep=True)
                            else:
                                short = m
                            short.content = (
                                f"[Older tool output removed by context guard. "
                                f"Original tool: {getattr(m, 'name', 'unknown')}. "
                                f"Budget: {max_input} tokens.]"
                            )
                            kept_non_sys.append(short)
                        # Skip other old non-essential messages
                        continue
                    kept_non_sys.append(m)
                messages = sys_msgs + kept_non_sys
                trim_actions.append(f"drop_old_messages:{drop_count}")
                # Re-find last human index since messages shifted
                lidx = _last_human_idx()

            final_total = _recount()
            logger.warning(
                "final_input_guard: over_limit round=%d tokens_before=%d tokens_after=%d max=%d actions=%s",
                trim_round, before, final_total, max_input, ",".join(trim_actions[-3:]),
            )

        # ── Stage 3: Final check — hard fail or forced last-resort trim ──
        if final_total > max_input:
            # Last resort: keep only system + last HumanMessage
            lidx3 = _last_human_idx()
            if lidx3 is not None:
                messages = sys_msgs + [messages[lidx3]]
                trim_actions.append("last_resort_keep_system_plus_last_human")
                final_total = _recount()

        if final_total > max_input:
            # Always fail closed — never send an over-budget request regardless of HARD_FAIL setting
            if FINAL_INPUT_GUARD_HARD_FAIL:
                logger.error(
                    "final_input_guard: hard_fail tokens=%d max=%d rounds=%d actions=%s",
                    final_total, max_input, trim_round, ",".join(trim_actions),
                )
                raise RuntimeError(
                    f"Final input guard: token budget exceeded after {trim_round} trim rounds. "
                    f"final_tokens={final_total} max_input_length={max_input}. "
                    f"Actions taken: {','.join(trim_actions)}. "
                    f"The model call was blocked to prevent context overflow."
                )
            else:
                logger.error(
                    "final_input_guard: fail_closed tokens=%d max=%d rounds=%d actions=%s",
                    final_total, max_input, trim_round, ",".join(trim_actions),
                )
                raise RuntimeError(
                    f"Final input guard: token budget exceeded after {trim_round} trim rounds. "
                    f"final_tokens={final_total} max_input_length={max_input}. "
                    f"Actions taken: {','.join(trim_actions)}. "
                    f"The model call was blocked (fail_closed mode)."
                )
        else:
            logger.info(
                "final_input_guard: ok profile=%s tokens=%d max=%d rounds=%d actions=%s",
                runtime_profile or "default", final_total, max_input, trim_round,
                ",".join(trim_actions) if trim_actions else "none",
            )

        # v2: thinking cleanup for non-DeepSeek providers
        provider = getattr(handler, 'provider', '') if hasattr(handler, 'provider') else ''
        model_name_val = getattr(handler, 'model_name', '') if hasattr(handler, 'model_name') else ''
        is_deepseek = 'deepseek' in str(provider).lower() or 'deepseek' in str(model_name_val).lower()

        if not is_deepseek:
            for msg in messages:
                if hasattr(msg, 'additional_kwargs'):
                    msg.additional_kwargs.pop('reasoning_content', None)
                    msg.additional_kwargs.pop('thinking', None)
                    msg.additional_kwargs.pop('reasoning', None)

        try:
            response = await handler(request.override(**override_kwargs))
            # Detect content_filter finish_reason and replace the generic
            # refusal with a user-friendly message so the user understands
            # the model's safety policy triggered, not a bot malfunction.
            response = self._handle_content_filter(response, call_index, turn_id=turn_id, session_id=session_id)
            # Detect network_error finish_reason and inject a user-visible
            # error notice when the model returns an empty reply due to a
            # transient network/provider error.
            response = self._handle_network_error(response, call_index, turn_id=turn_id, session_id=session_id)
            return response
        except asyncio.CancelledError:
            logger.warning(
                "[ModelCall][cancelled] turn=%s session=%s call_id=%s call_index=%d duration=%.2fs",
                turn_id or "-",
                session_id or "-",
                call_id,
                call_index,
                time.monotonic() - model_call_started_at,
            )
            raise
        except Exception:
            logger.exception(
                "[ModelCall][error] turn=%s session=%s call_id=%s call_index=%d duration=%.2fs",
                turn_id or "-",
                session_id or "-",
                call_id,
                call_index,
                time.monotonic() - model_call_started_at,
            )
            raise
        finally:
            record.provider_usage = self._extract_provider_usage(response)
            if response is not None:
                self._log_model_call_end(
                    turn_id=turn_id,
                    session_id=session_id,
                    call_id=call_id,
                    call_index=call_index,
                    model=effective_model,
                    response=response,
                    duration_s=time.monotonic() - model_call_started_at,
                    provider_usage=record.provider_usage,
                )
            self._log_token_final(
                record,
                turn_id=turn_id,
                session_id=session_id,
                provider_usage=record.provider_usage,
            )

            logger.debug(
                "Middleware wrap_model_call: turn_id=%s session_id=%s call_index=%d messages=%d extra_blocks=%d "
                "bootstrap=%s weibo=%s facts=%s summary=%s recall=%s personality=%s flush_reminder=%s "
                "base_system_tokens=%d tool_schema_tokens=%d tool_schema_count=%d extra_breakdown=%s "
                "visible_pre_split=%s budget_post_tokens=%d visible_post_split=%s browser_guard_tokens=%d "
                "truncated_tool_messages=%d truncated_tool_chars=%d "
                "actual_pre_tokens=%d actual_post_tokens=%d over_budget_pre=%d over_budget_post=%d "
                "provider_usage=%s",
                turn_id or "-",
                session_id or "-",
                call_index,
                len(messages),
                len(extra_sys_parts),
                bool(bootstrap_block),
                bool(weibo_onboarding_block),
                bool(facts_block),
                bool(summary),
                bool(recall),
                bool(personality),
                flush_reminder_injected,
                record.base_system_tokens,
                record.tool_schema_tokens,
                record.tool_schema_count,
                record.extra_breakdown,
                record.visible_pre.split_by_type,
                record.budget_post.total_tokens,
                record.visible_post.split_by_type,
                record.browser_guard_tokens,
                record.truncated_tool_messages,
                record.truncated_tool_chars,
                record.actual_total_pre_tokens,
                record.actual_total_post_tokens,
                record.over_budget_pre_tokens,
                record.over_budget_post_tokens,
                record.provider_usage or {},
            )

    @staticmethod
    def _clean_extra_parts_map(extra_parts_map):
        """Strip model special tokens from extra_parts_map values before token counting."""
        clean_extra_parts_map = {}

        for name, text in extra_parts_map.items():
            if isinstance(text, str):
                if "<|" in text or "<｜" in text:
                    try:
                        logger.warning(
                            "model special token-like text found in extra_parts_map field=%s len=%s preview=%r",
                            name,
                            len(text),
                            strip_model_special_tokens(text[:200]),
                        )
                    except Exception:
                        pass

                clean_extra_parts_map[name] = strip_model_special_tokens(text)

            elif isinstance(text, (list, tuple, dict)):
                clean_extra_parts_map[name] = sanitize_text_payload(text)

            else:
                clean_extra_parts_map[name] = text

        return clean_extra_parts_map

    @classmethod
    def _extract_request_system_text(cls, request: Any) -> str:
        """Return the request system message content as text for diagnostics."""
        system_message = getattr(request, "system_message", None)
        content = getattr(system_message, "content", "")
        return cls._extract_text_from_content(content)

    @classmethod
    def _extract_request_system_blocks(cls, request: Any) -> list[dict[str, Any]]:
        """Return request system prompt as normalized text blocks."""
        system_message = getattr(request, "system_message", None)
        if system_message is None:
            return []

        content_blocks = getattr(system_message, "content_blocks", None)
        blocks: list[dict[str, Any]] = []
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict):
                    blocks.append(dict(block))
                elif isinstance(block, str):
                    blocks.append({"type": "text", "text": block})
            if blocks:
                return blocks

        content = getattr(system_message, "content", "")
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
            return blocks
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    blocks.append(dict(block))
                elif isinstance(block, str):
                    blocks.append({"type": "text", "text": block})
        return blocks

    def _build_model_call_id(self, call_index: int, *, turn_id: str) -> str:
        """Return a stable per-turn model-call identifier for logs."""
        normalized_turn_id = turn_id or "turn-unknown"
        return f"{normalized_turn_id}:model_call:{call_index}"

    @staticmethod
    def _describe_model(model: Any) -> tuple[str, str]:
        """Return a stable model name and class name for logs."""
        for attr in ("model_name", "model", "model_id"):
            value = getattr(model, attr, None)
            if value:
                return str(value), type(model).__name__
        return type(model).__name__, type(model).__name__

    @staticmethod
    def _truncate_preview(text: str, *, limit: int = 160) -> str:
        """Return a single-line bounded preview string for logs."""
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    @classmethod
    def _extract_text_from_content(cls, content: Any) -> str:
        """Extract visible text from a message content payload."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            for key in ("text", "thinking", "output"):
                value = block.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)
        return "\n".join(parts)

    @classmethod
    def _message_preview(cls, message: Any) -> str:
        """Return a bounded preview of message content for logs."""
        return cls._truncate_preview(cls._extract_text_from_content(getattr(message, "content", "")))

    @classmethod
    def _last_visible_message_snapshot(cls, messages: list[BaseMessage]) -> tuple[str, str]:
        """Return the last non-system message type and preview."""
        for message in reversed(messages):
            if isinstance(message, SystemMessage):
                continue
            return cls._message_type_name(message), cls._message_preview(message)
        return "none", ""

    @staticmethod
    def _tool_name(tool: Any) -> str:
        """Return a stable tool name for logs."""
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    return name
            name = tool.get("name")
            if isinstance(name, str) and name:
                return name
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            return name
        return type(tool).__name__

    @classmethod
    def _summarize_tool_names(cls, tools: list[Any], *, limit: int = 10) -> str:
        """Return a bounded tool-name summary for logs."""
        if not tools:
            return "-"
        names = [cls._tool_name(tool) for tool in tools[:limit]]
        if len(tools) > limit:
            names.append(f"+{len(tools) - limit} more")
        return ",".join(names)

    @classmethod
    def _response_messages(cls, response: Any) -> list[Any]:
        """Extract model-response messages for lifecycle logs."""
        if response is None:
            return []
        if isinstance(response, ExtendedModelResponse):
            response = response.model_response
        if isinstance(response, ModelResponse):
            return list(response.result)
        return [response]

    @classmethod
    def _log_model_call_start(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_id: str,
        call_index: int,
        model: Any,
        messages: list[BaseMessage],
        system_text: str,
        tools: list[Any],
        flags: dict[str, bool],
        model_settings: dict[str, Any],
    ) -> None:
        """Emit a model-call start log with the effective request shape."""
        model_name, model_class = cls._describe_model(model)
        last_message_type, last_message_preview = cls._last_visible_message_snapshot(messages)
        logger.info(
            "[ModelCall][start] turn=%s session=%s call_id=%s call_index=%d model=%s class=%s "
            "messages=%d system_chars=%d tools=%d tool_names=%s active_blocks=%s last_message_type=%s "
            "last_message=%r model_settings=%s",
            turn_id or "-",
            session_id or "-",
            call_id,
            call_index,
            model_name,
            model_class,
            len(messages),
            len(system_text or ""),
            len(tools),
            cls._summarize_tool_names(tools),
            cls._format_active_blocks(flags),
            last_message_type,
            last_message_preview,
            model_settings or {},
        )

    # Known finish_reason values that indicate content safety filtering.
    _CONTENT_FILTER_REASONS = frozenset({"content_filter", "content_filtered", "sensitive"})

    # User-friendly message appended when content_filter is detected.
    _CONTENT_FILTER_NOTICE = (
        "\n\n---\n⚠️ 当前回复受到模型内容安全策略限制，可能无法正常回答该问题。建议换一种方式提问，或尝试切换其他模型。"
    )

    # Known finish_reason values that indicate a network/provider error.
    _NETWORK_ERROR_REASONS = frozenset({"network_error", "error", "timeout"})

    # User-friendly message used when a network error causes an empty reply.
    _NETWORK_ERROR_NOTICE = "⚠️ 模型请求遇到网络错误，未能获取到回复。请稍后重试，或检查网络连接和模型服务状态。"

    def _handle_content_filter(
        self,
        response: Any,
        call_index: int,
        *,
        turn_id: str,
        session_id: str,
    ) -> Any:
        """Detect content_filter finish_reason and enrich the AI reply.

        When the model provider returns ``finish_reason='content_filter'``,
        the original reply is typically a generic refusal like "你好，我无法
        给到相关内容。".  This method appends a user-friendly notice so the
        end-user understands the limitation comes from the model's safety
        policy, not from the bot itself.
        """
        response_messages = self._response_messages(response)
        if not response_messages:
            return response

        last_message = response_messages[-1]
        if not isinstance(last_message, AIMessage):
            return response

        metadata = self._coerce_dict_like(getattr(last_message, "response_metadata", None))
        if not metadata:
            return response

        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason") or ""
        if finish_reason not in self._CONTENT_FILTER_REASONS:
            return response

        # Log a clear warning so operators can diagnose quickly.
        logger.warning(
            "[ModelCall][content_filter] turn=%s session=%s call_index=%d "
            "finish_reason=%s — model safety policy triggered, "
            "enriching reply with user notice. original=%r",
            turn_id or "-",
            session_id or "-",
            call_index,
            finish_reason,
            self._message_preview(last_message),
        )

        # Append the notice to the AI message content.
        original_content = last_message.content or ""
        enriched_content = original_content + self._CONTENT_FILTER_NOTICE
        enriched_message = last_message.model_copy(update={"content": enriched_content})

        # Replace the last message in the response.
        if hasattr(response, "messages"):
            new_messages = list(response.messages)
            new_messages[-1] = enriched_message
            response = response.model_copy(update={"messages": new_messages})
        elif isinstance(response, dict) and "messages" in response:
            new_messages = list(response["messages"])
            new_messages[-1] = enriched_message
            response = {**response, "messages": new_messages}

        return response

    def _handle_network_error(
        self,
        response: Any,
        call_index: int,
        *,
        turn_id: str,
        session_id: str,
    ) -> Any:
        """Detect network_error finish_reason and inject a user-friendly error message.

        When the model provider returns ``finish_reason='network_error'`` (or
        similar transient error reasons), the response content is typically
        empty.  This method replaces the empty reply with a user-visible error
        notice so the user is not left with zero feedback.
        """
        response_messages = self._response_messages(response)
        if not response_messages:
            return response

        last_message = response_messages[-1]
        if not isinstance(last_message, AIMessage):
            return response

        metadata = self._coerce_dict_like(getattr(last_message, "response_metadata", None))
        if not metadata:
            return response

        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason") or ""
        if finish_reason not in self._NETWORK_ERROR_REASONS:
            return response

        # Only inject when the reply is actually empty — don't overwrite a
        # partial response that happened to carry a network_error reason.
        existing_content = last_message.content or ""
        if isinstance(existing_content, str) and existing_content.strip():
            return response
        if isinstance(existing_content, list) and existing_content:
            return response

        logger.warning(
            "[ModelCall][network_error] turn=%s session=%s call_index=%d "
            "finish_reason=%s — model returned empty content due to network error, "
            "injecting user-visible error notice.",
            turn_id or "-",
            session_id or "-",
            call_index,
            finish_reason,
        )

        enriched_message = last_message.model_copy(update={"content": self._NETWORK_ERROR_NOTICE})

        # Replace the last message in the response.
        if hasattr(response, "messages"):
            new_messages = list(response.messages)
            new_messages[-1] = enriched_message
            response = response.model_copy(update={"messages": new_messages})
        elif isinstance(response, dict) and "messages" in response:
            new_messages = list(response["messages"])
            new_messages[-1] = enriched_message
            response = {**response, "messages": new_messages}

        return response

    @classmethod
    def _log_model_call_end(
        cls,
        *,
        turn_id: str,
        session_id: str,
        call_id: str,
        call_index: int,
        model: Any,
        response: Any,
        duration_s: float,
        provider_usage: dict[str, Any] | None,
    ) -> None:
        """Emit a model-call completion log with output summary."""
        model_name, model_class = cls._describe_model(model)
        response_messages = cls._response_messages(response)
        last_message = response_messages[-1] if response_messages else None
        preview = cls._message_preview(last_message) if last_message is not None else ""
        tool_calls = len(getattr(last_message, "tool_calls", []) or []) if last_message is not None else 0
        response_metadata = (
            cls._coerce_dict_like(getattr(last_message, "response_metadata", None))
            if last_message is not None
            else None
        )

        # Emit a dedicated warning when the model's safety filter triggered.
        finish_reason = (response_metadata or {}).get("finish_reason", "")
        if finish_reason in cls._CONTENT_FILTER_REASONS:
            logger.warning(
                "[ModelCall][content_filter_detected] turn=%s session=%s call_id=%s "
                "call_index=%d model=%s finish_reason=%s — the model refused to "
                "answer due to its content safety policy. preview=%r",
                turn_id or "-",
                session_id or "-",
                call_id,
                call_index,
                model_name,
                finish_reason,
                preview,
            )

        logger.info(
            "[ModelCall][end] turn=%s session=%s call_id=%s call_index=%d model=%s class=%s duration=%.2fs "
            "response_messages=%d last_message_type=%s tool_calls=%d preview=%r usage=%s response_metadata=%s",
            turn_id or "-",
            session_id or "-",
            call_id,
            call_index,
            model_name,
            model_class,
            duration_s,
            len(response_messages),
            cls._message_type_name(last_message)
            if isinstance(last_message, BaseMessage)
            else type(last_message).__name__,
            tool_calls,
            preview,
            provider_usage or {},
            response_metadata or {},
        )

    @staticmethod
    def _message_type_name(message: BaseMessage) -> str:
        """Return a stable token-bucket type label for a message."""
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, HumanMessage):
            return "human"
        if isinstance(message, ToolMessage):
            return "tool"
        return type(message).__name__.removesuffix("Message").lower() or "other"

    def _summarize_visible_tokens(self, messages: list[BaseMessage]) -> _VisibleTokenStats:
        """Summarize tokens for the visible message list."""
        split_by_type: dict[str, int] = {}
        system_tokens = 0
        non_system_tokens = 0
        tool_tokens = 0
        tool_call_tokens = 0

        for message in messages:
            msg_tokens = _count_tokens([message])
            msg_tool_call_tokens = count_message_tool_call_tokens(message)
            kind = self._message_type_name(message)
            split_by_type[kind] = split_by_type.get(kind, 0) + msg_tokens
            if msg_tool_call_tokens:
                key = f"{kind}_tool_calls"
                split_by_type[key] = split_by_type.get(key, 0) + msg_tool_call_tokens
            if isinstance(message, SystemMessage):
                system_tokens += msg_tokens
            else:
                non_system_tokens += msg_tokens
            if isinstance(message, ToolMessage):
                tool_tokens += msg_tokens
            tool_call_tokens += msg_tool_call_tokens

        return _VisibleTokenStats(
            total_tokens=system_tokens + non_system_tokens,
            estimated_total_tokens=system_tokens + non_system_tokens + tool_call_tokens,
            system_tokens=system_tokens,
            non_system_tokens=non_system_tokens,
            tool_tokens=tool_tokens,
            tool_call_tokens=tool_call_tokens,
            count=len(messages),
            split_by_type=split_by_type,
        )

    def _log_token_preflight(
        self,
        *,
        call_index: int,
        turn_id: str,
        session_id: str,
        base_system_tokens: int,
        extra_system_tokens: int,
        visible_pre: _VisibleTokenStats,
        actual_total_pre_tokens: int,
        over_budget_pre_tokens: int,
        truncation_stats: Any,
        tool_schema_tokens: int,
        tool_schema_count: int,
        flags: dict[str, bool],
    ) -> None:
        """Emit a readable INFO log before token budgeting."""
        logger.info(
            self._format_token_log_block(
                phase="preflight",
                call_index=call_index,
                turn_id=turn_id,
                session_id=session_id,
                max_input_length=self._token_budget.max_input_length,
                actual_total_tokens=actual_total_pre_tokens,
                visible_tokens=visible_pre.total_tokens,
                estimated_visible_tokens=visible_pre.estimated_total_tokens,
                over_budget_tokens=over_budget_pre_tokens,
                base_system_tokens=base_system_tokens,
                extra_system_tokens=extra_system_tokens,
                final_system_tokens=None,
                tool_tokens=visible_pre.tool_tokens,
                tool_call_tokens=visible_pre.tool_call_tokens,
                message_count=visible_pre.count,
                truncated_tool_messages=truncation_stats.truncated_messages,
                truncated_tool_chars=truncation_stats.truncated_chars,
                tool_schema_tokens=tool_schema_tokens,
                tool_schema_count=tool_schema_count,
                browser_guard_tokens=0,
                flags=flags,
            )
        )

    def _log_token_final(
        self,
        record: _TokenLogRecord,
        *,
        turn_id: str,
        session_id: str,
        provider_usage: dict[str, Any] | None = None,
    ) -> None:
        """Emit a readable INFO log after final payload shaping."""
        logger.info(
            self._format_token_log_block(
                phase="final",
                call_index=record.call_index,
                turn_id=turn_id,
                session_id=session_id,
                max_input_length=record.max_input_length,
                actual_total_tokens=record.actual_total_post_tokens,
                visible_tokens=record.visible_post.total_tokens,
                estimated_visible_tokens=record.visible_post.estimated_total_tokens,
                over_budget_tokens=record.over_budget_post_tokens,
                base_system_tokens=record.base_system_tokens,
                extra_system_tokens=record.extra_system_tokens,
                final_system_tokens=record.final_system_tokens,
                tool_tokens=record.visible_post.tool_tokens,
                tool_call_tokens=record.visible_post.tool_call_tokens,
                message_count=record.visible_post.count,
                truncated_tool_messages=record.truncated_tool_messages,
                truncated_tool_chars=record.truncated_tool_chars,
                tool_schema_tokens=record.tool_schema_tokens,
                tool_schema_count=record.tool_schema_count,
                browser_guard_tokens=record.browser_guard_tokens,
                flags=record.flags,
                provider_usage=provider_usage,
            )
        )

    def _format_token_log_block(
        self,
        *,
        phase: str,
        call_index: int,
        turn_id: str,
        session_id: str,
        max_input_length: int,
        actual_total_tokens: int,
        visible_tokens: int,
        estimated_visible_tokens: int,
        over_budget_tokens: int,
        base_system_tokens: int,
        extra_system_tokens: int,
        final_system_tokens: int | None,
        tool_tokens: int,
        tool_call_tokens: int,
        message_count: int,
        truncated_tool_messages: int,
        truncated_tool_chars: int,
        tool_schema_tokens: int,
        tool_schema_count: int,
        browser_guard_tokens: int,
        flags: dict[str, bool],
        provider_usage: dict[str, Any] | None = None,
    ) -> str:
        """Render a multi-line token diagnostics block for INFO logs."""
        active_blocks = self._format_active_blocks(flags)
        lines = [
            f"[Token][Call #{call_index}][{phase}]",
            f"turn={turn_id or '-'} session={session_id or '-'}",
            f"call_id={self._build_model_call_id(call_index, turn_id=turn_id)}",
            "token_source=estimated(local_token_counting)",
            f"{'actual_total':<18}{actual_total_tokens:>12,}",
            f"{'budget_visible':<18}{visible_tokens:>12,}",
            f"{'estimated_visible':<18}{estimated_visible_tokens:>12,}",
            f"{'over_budget':<18}{over_budget_tokens:>12,} / {max_input_length:,}",
            "system:",
            f"{'  base':<18}{base_system_tokens:>12,}",
            f"{'  extra':<18}{extra_system_tokens:>12,}  [{active_blocks}]",
        ]
        if final_system_tokens is not None:
            lines.append(f"{'  final':<18}{final_system_tokens:>12,}")
            lines.append(f"{'  browser_guard':<18}{browser_guard_tokens:>12,}")
        lines.extend(
            [
                "tools:",
                f"{'  schemas':<18}{tool_schema_tokens:>12,}",
                f"{'  count':<18}{tool_schema_count:>12,}",
                "messages:",
                f"{'  total':<18}{visible_tokens:>12,}",
                f"{'  estimated':<18}{estimated_visible_tokens:>12,}",
                f"{'  tool':<18}{tool_tokens:>12,}",
                f"{'  tool_calls':<18}{tool_call_tokens:>12,}",
                f"{'  count':<18}{message_count:>12,}",
                "truncation:",
                f"{'  tool_messages':<18}{truncated_tool_messages:>12,}",
                f"{'  chars':<18}{truncated_tool_chars:>12,}",
            ]
        )
        if phase == "final":
            lines.extend(
                [
                    "provider_usage:",
                    f"{'  source':<18}{self._provider_usage_source(provider_usage):>12}",
                    f"{'  input':<18}{self._format_usage_metric(provider_usage, 'input_tokens'):>12}",
                    f"{'  output':<18}{self._format_usage_metric(provider_usage, 'output_tokens'):>12}",
                    f"{'  total':<18}{self._format_usage_metric(provider_usage, 'total_tokens'):>12}",
                    f"{'input_vs_provider':<18}{self._format_input_delta(actual_total_tokens, provider_usage):>12}",
                ]
            )
        return "\n" + "\n".join(lines)

    @staticmethod
    def _coerce_dict_like(value: Any) -> dict[str, Any] | None:
        """Convert pydantic-ish metadata objects to plain dicts when possible."""
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        elif hasattr(value, "dict"):
            value = value.dict(exclude_none=True)
        if isinstance(value, dict):
            return dict(value)
        return None

    @classmethod
    def _extract_message_usage(cls, message: Any) -> dict[str, Any] | None:
        """Extract LangChain-standard usage metadata from a message-like object."""
        return cls._coerce_dict_like(getattr(message, "usage_metadata", None))

    @classmethod
    def _extract_provider_usage(cls, response: Any) -> dict[str, Any] | None:
        """Extract provider-reported usage metadata from a model response."""
        if response is None:
            return None
        if isinstance(response, ExtendedModelResponse):
            response = response.model_response
        if isinstance(response, ModelResponse):
            for message in reversed(response.result):
                usage = cls._extract_message_usage(message)
                if usage:
                    return usage
            return None
        if isinstance(response, AIMessage):
            return cls._extract_message_usage(response)
        return cls._extract_message_usage(response)

    @staticmethod
    def _provider_usage_source(provider_usage: dict[str, Any] | None) -> str:
        """Return the origin label for the provider-usage section."""
        return "provider" if provider_usage else "missing"

    @staticmethod
    def _format_usage_metric(provider_usage: dict[str, Any] | None, key: str) -> str:
        """Format a provider-usage metric for human-readable logs."""
        value = provider_usage.get(key) if provider_usage else None
        if isinstance(value, Real) and not isinstance(value, bool):
            return f"{int(value):,}"
        return "-"

    @staticmethod
    def _format_input_delta(actual_total_tokens: int, provider_usage: dict[str, Any] | None) -> str:
        """Format the gap between local input estimate and provider input usage."""
        provider_input = provider_usage.get("input_tokens") if provider_usage else None
        if isinstance(provider_input, Real) and not isinstance(provider_input, bool):
            return f"{actual_total_tokens - int(provider_input):+,}"
        return "-"

    @staticmethod
    def _format_active_blocks(flags: dict[str, bool]) -> str:
        """Return active dynamic system blocks in a stable order."""
        ordered_keys = (
            "bootstrap",
            "weibo",
            "facts",
            "summary",
            "recall",
            "personality",
            "flush_reminder",
            "memory_write_reminder",
        )
        active = [key for key in ordered_keys if flags.get(key)]
        return ", ".join(active) if active else "none"

    @staticmethod
    def _is_user_turn(messages: list[BaseMessage]) -> bool:
        """Return True when the last non-system message is a HumanMessage.

        Used to gate the memory-write-reminder so it is only injected on
        user-initiated turns, not on intermediate tool-call rounds inside the
        ReAct loop where the last message is an AIMessage or ToolMessage.
        """
        for msg in reversed(messages):
            if not isinstance(msg, SystemMessage):
                return isinstance(msg, HumanMessage)
        return False

    @staticmethod
    def _last_user_message_is_meaningful(messages: list[BaseMessage]) -> bool:
        """Return True when the last HumanMessage carries enough content to warrant a memory-write check.

        Two cheap heuristics — no LLM required:
        1. Stripped length >= 5 characters.
        2. Contains at least one Unicode letter or digit.

        Messages that fail either check (e.g. "ok", "👍", pure punctuation) almost
        never contain information worth persisting, so skipping the reminder saves
        tokens without any meaningful loss of recall quality.
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                text = ""
                if isinstance(msg.content, str):
                    text = msg.content.strip()
                elif isinstance(msg.content, list):
                    parts = [block.get("text", "") if isinstance(block, dict) else "" for block in msg.content]
                    text = " ".join(parts).strip()
                if len(text) < 5:
                    return False
                import unicodedata

                return any(unicodedata.category(ch)[0] in {"L", "N"} for ch in text)
        return False

    def _build_memory_write_reminder(self) -> str:
        """Build a per-turn reminder to persist noteworthy information.

        Appended last in the system prompt to exploit recency bias — the model
        is more likely to act on instructions it sees immediately before
        generating its reply.  Only injected on user-message turns.

        Provides concrete trigger signals so the model has an unambiguous
        checklist rather than a vague self-assessment prompt.  The full
        decision tree still lives in AGENTS.md; this reminder is the
        last-mile enforcement layer.
        """
        if self._language == "zh":
            return (
                "<memory-write-reminder>\n"
                "回复前先对照以下信号逐条检查，命中任意一条就必须先写文件再回复：\n"
                "- 用户提到了人名、宠物名、项目名、地点、组织\n"
                "- 用户表达了偏好、习惯、风格要求（喜欢/不喜欢/以后要/不要）\n"
                "- 做出了技术决策、方案选择、架构约定\n"
                "- 约定了规则、流程、工作方式\n"
                "- 用户分享了情绪、状态、近况、重要事件\n"
                "- 出现了值得长期记住的教训、洞察、结论\n"
                "以上均不符合 → 直接回复，无需写文件。\n"
                "</memory-write-reminder>"
            )
        return (
            "<memory-write-reminder>\n"
            "Before replying, check each signal below — write to memory first if ANY match:\n"
            "- User mentioned a person, pet, project, place, or organisation\n"
            "- User expressed a preference, habit, or style rule (like/dislike/always/never)\n"
            "- A technical decision, plan choice, or architecture convention was made\n"
            "- A rule, workflow, or working agreement was established\n"
            "- User shared emotions, status, news, or a significant event\n"
            "- A lesson, insight, or conclusion worth remembering long-term emerged\n"
            "None of the above match → reply directly, no file write needed.\n"
            "</memory-write-reminder>"
        )
