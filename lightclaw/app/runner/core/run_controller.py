"""Run controller for a single agent query.

This module owns the lifecycle of a single ``query_handler`` call:
* Logging and basic request metadata
* Wiring session / tools / events together
* Handling cancellation, errors and finalisation

Execution is handled exclusively via the LangGraph path, including:
* LangGraph-native command handling
* Session persistence (save/load history + compressed summary)
* Graceful cancellation via middleware
* Middleware integration (Bootstrap + MemoryCompaction)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Real
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage

from lightclaw.agent.bootstrap_state import is_bootstrap_pending
from lightclaw.agent.core.adapters.turn_protocol import (
    langchain_message_to_turn_event,
    turn_inputs_to_langchain_messages,
)
from lightclaw.agent.utils.token_counting import count_str_tokens
from lightclaw.agent.utils.token_estimation import count_json_like_tokens
from lightclaw.app.channels.core.schema import DEFAULT_CHANNEL
from lightclaw.app.core.error_utils import classify_exception
from lightclaw.app.runner.core.session_persistence import SessionPersistenceManager
from lightclaw.app.runner.core.turn_finalization import TurnFinalizationOrchestrator
from lightclaw.app.runner.core.turn_state import ExecutionContext, TurnState
from lightclaw.app.runner.query_error_dump import write_query_error_dump
from lightclaw.app.runner.session.session_factory import SessionFactory
from lightclaw.constant import DAILY_TALK_ENABLED, WORKING_DIR
from lightclaw.domain import TurnEventType

logger = logging.getLogger(__name__)
_MEDIA_BLOCK_TYPES = frozenset({"file", "image", "audio", "video"})

# SessionPersistenceManager singleton initialized at startup
_session_persistence: SessionPersistenceManager | None = None

# TurnFinalizationOrchestrator singleton initialized at startup
_turn_finalizer: TurnFinalizationOrchestrator | None = None


@dataclass
class RunContext:
    """Compatibility wrapper for legacy _run_langgraph test call-sites."""

    run_id: str
    session_id: str | None
    user_id: str | None
    channel: str


def _preview_text(text: str | None, *, limit: int = 160) -> str:
    """Return a single-line preview suitable for logs."""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _usage_summary(usage: dict[str, Any] | None) -> str:
    """Format token-usage dicts into a concise log string."""
    if not usage:
        return "-"
    return f"in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)} total={usage.get('total_tokens', 0)}"


def _usage_source(usage: dict[str, Any] | None) -> str:
    """Return whether usage came from the provider or local estimation."""
    if not usage:
        return "missing"
    if bool(usage.get("estimated")) or usage.get("source") == "local_token_counting":
        return "estimated"
    return "provider"


def _format_token_diag_value(value: Any) -> str:
    """Format a token-diagnostics scalar for aligned logs."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{int(value):,}"
    return str(value)


def _format_usage_metric(usage: dict[str, Any] | None, key: str) -> str:
    """Format a usage metric for aligned logs."""
    if not usage:
        return "-"
    value = usage.get(key)
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{int(value):,}"
    return "-"


def _format_input_vs_run_usage(summary: dict[str, Any] | None, usage: dict[str, Any] | None) -> str:
    """Format the gap between estimated turn input and run-usage input."""
    if not summary or not usage:
        return "-"
    estimated_input = summary.get("estimated_input_tokens")
    run_input = usage.get("input_tokens")
    if not isinstance(estimated_input, Real) or isinstance(estimated_input, bool):
        return "-"
    if not isinstance(run_input, Real) or isinstance(run_input, bool):
        return "-"
    return f"{int(estimated_input) - int(run_input):+,}"


def _format_token_diag_block(
    *,
    run_id: str,
    turn_id: str,
    session_id: str,
    summary: dict[str, Any] | None,
    run_usage: dict[str, Any] | None,
) -> str:
    """Render a multi-line turn-level token diagnostics block."""
    if not summary:
        return f"\n[Token][Turn Summary]\nrun={run_id} turn={turn_id} session={session_id}\nsummary=missing"

    lines = [
        "[Token][Turn Summary]",
        f"run={run_id} turn={turn_id} session={session_id}",
        f"{'model_calls':<22}{_format_token_diag_value(summary.get('model_calls', 0)):>12}",
        f"{'first_call_actual':<22}{_format_token_diag_value(summary.get('first_call_actual_tokens', 0)):>12}",
        f"{'estimated_input':<22}{_format_token_diag_value(summary.get('estimated_input_tokens', 0)):>12}",
        f"{'tool_schemas':<22}{_format_token_diag_value(summary.get('tool_schema_tokens', 0)):>12}",
        f"{'max_actual':<22}{_format_token_diag_value(summary.get('max_actual_tokens', 0)):>12}",
        f"{'max_over':<22}{_format_token_diag_value(summary.get('max_over_budget', 0)):>12}",
        f"{'bootstrap_first':<22}{_format_token_diag_value(summary.get('bootstrap_first_call', False)):>12}",
        f"{'later_tool':<22}{_format_token_diag_value(summary.get('later_tool_result_seen', False)):>12}",
        f"{'later_large_tool':<22}{_format_token_diag_value(summary.get('later_large_tool_result_seen', False)):>12}",
        "run_usage:",
        f"{'  source':<22}{_usage_source(run_usage):>12}",
        f"{'  input':<22}{_format_usage_metric(run_usage, 'input_tokens'):>12}",
        f"{'  output':<22}{_format_usage_metric(run_usage, 'output_tokens'):>12}",
        f"{'  total':<22}{_format_usage_metric(run_usage, 'total_tokens'):>12}",
        f"{'input_vs_run_usage':<22}{_format_input_vs_run_usage(summary, run_usage):>12}",
    ]
    return "\n" + "\n".join(lines)


def _estimate_content_tokens(blocks: Any) -> int:
    """Estimate visible token count from TurnEvent content blocks."""
    if not isinstance(blocks, list):
        return 0

    total = 0
    for block in blocks:
        if isinstance(block, str):
            total += count_str_tokens(block)
            continue
        if not isinstance(block, dict):
            continue
        for key in ("text", "thinking", "output", "arguments"):
            value = block.get(key)
            if isinstance(value, str) and value:
                total += count_str_tokens(value)
    return total


def _extract_text_from_turn_content(blocks: Any) -> str:
    """Extract visible assistant text from turn-event content blocks."""
    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _estimate_tool_call_event_tokens(data: Any) -> int:
    """Estimate output tokens for a tool-start event emitted from model tool calls."""
    if not isinstance(data, dict):
        return 0

    payload = {
        "call_id": data.get("call_id"),
        "name": data.get("name") or data.get("tool_name"),
        "arguments": data.get("arguments"),
    }
    return count_json_like_tokens(payload)


def _build_estimated_usage(
    *,
    token_summary: dict[str, Any] | None,
    estimated_output_tokens: int,
) -> dict[str, Any] | None:
    """Build a fallback usage dict from local token diagnostics."""
    if not token_summary:
        return None

    input_tokens = int(token_summary.get("estimated_input_tokens") or 0)
    output_tokens = max(int(estimated_output_tokens or 0), 0)
    total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        return None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated": True,
        "source": "local_token_counting",
        "model_calls": int(token_summary.get("model_calls") or 0),
    }


def _attach_usage_to_last_assistant_message(messages: list[Any], usage: dict[str, Any] | None) -> None:
    """Attach usage metadata to the last assistant message when missing."""
    if not usage:
        return

    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        if getattr(message, "usage_metadata", None):
            return
        messages[idx] = message.model_copy(update={"usage_metadata": dict(usage)})
        return


# _background_compact has been moved to SessionPersistenceManager._background_compact()
# See: lightclaw.app.runner.core.session_persistence


def _message_has_file_or_media_blocks(message: Any) -> bool:
    """Return True when the message contains file/media blocks."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False

    for block in content:
        if not isinstance(block, dict):
            if hasattr(block, "model_dump"):
                block = block.model_dump(exclude_none=True)
            elif hasattr(block, "dict"):
                block = block.dict(exclude_none=True)
            else:
                continue

        if block.get("type") in _MEDIA_BLOCK_TYPES:
            return True

    return False


def _extract_user_turn_texts(
    inputs: Any,
    *,
    is_command_predicate: Any,
) -> list[str]:
    """Extract natural-language user texts from turn inputs."""
    if inputs is None:
        return []

    messages = inputs if isinstance(inputs, list) else [inputs]
    result: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role != "user":
            continue

        getter = getattr(msg, "get_text_content", None)
        if callable(getter):
            text = getter()
        else:
            text = ""

        normalized = (text or "").strip()
        if not normalized:
            continue
        if is_command_predicate(normalized):
            continue
        result.append(normalized)
    return result


class RunController:
    """Control the lifecycle of a single agent run."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
    ) -> None:
        self._session_factory = session_factory
        self.last_run_usage: dict[str, Any] | None = None
        self.last_final_output_usage: dict[str, Any] | None = None

    @staticmethod
    def _initialize_singletons():
        """Initialize module-level singletons on first use."""
        global _session_persistence, _turn_finalizer
        if _session_persistence is None:
            _session_persistence = SessionPersistenceManager()
        if _turn_finalizer is None:
            _turn_finalizer = TurnFinalizationOrchestrator(
                session_factory=None,  # Will be set per-call
                session_persistence=_session_persistence,
            )

    async def run(
        self,
        *,
        inputs=None,
        msgs=None,
        request: Any,
    ) -> AsyncIterator[tuple[Any, bool]]:
        """Execute a single agent run as an async generator.

        Always dispatches to the LangGraph execution path.
        """
        if inputs is None:
            inputs = msgs
        self.last_run_usage = None
        self.last_final_output_usage = None
        run_id = str(uuid4())
        session_id = request.session_id
        user_id = request.user_id
        channel = request.channel or DEFAULT_CHANNEL

        logger.info(
            "Agent run starting: run_id=%s turn_id=%s session_id=%s user_id=%s channel=%s ephemeral=%s inputs=%d",
            run_id,
            request.turn_id or run_id,
            session_id or "-",
            user_id or "-",
            channel,
            bool(getattr(request, "ephemeral", False)),
            len(inputs) if isinstance(inputs, list) else int(inputs is not None),
        )

        async for msg, last in self._run_langgraph(inputs=inputs, request=request, run_id=run_id):
            yield msg, last

    # ------------------------------------------------------------------
    # LangGraph path
    # ------------------------------------------------------------------

    async def _run_langgraph(
        self,
        *,
        inputs=None,
        msgs=None,
        request: Any,
        run_id: str | None = None,
        run_ctx: RunContext | None = None,
    ) -> AsyncIterator[tuple[Any, bool]]:
        """New LangGraph agent execution path.

        Stage 1: Inspect the last user message and short-circuit pure commands
        Stage 2: For normal queries, preprocess files and build full context
        Stage 3: Execute ReAct agent via graph.astream_events

        PR5: Commands are now handled natively; session state is persisted.
        """
        from langchain_core.messages import AIMessage as _AIMessage

        from lightclaw.agent.core.command_handler import is_command
        from lightclaw.agent.utils import process_file_and_media_blocks_in_message

        run_started_at = time.monotonic()
        token_summary: dict[str, Any] | None = None

        normalized_run_id = run_id or (run_ctx.run_id if run_ctx is not None else str(uuid4()))

        # Build TurnState from request metadata
        turn_state = TurnState(
            run_id=normalized_run_id,
            session_id=request.session_id,
            user_id=request.user_id,
            channel=request.channel or DEFAULT_CHANNEL,
            is_ephemeral=request.ephemeral,
            bootstrap_pending=is_bootstrap_pending(WORKING_DIR),
        )

        # ExecutionContext built later after SessionFactory.create_langgraph_context()
        exec_context: ExecutionContext | None = None

        deferred_daily_write = False
        pending_user_texts: list[str] = []
        if inputs is None:
            inputs = msgs

        async def _resolve_command_handler():
            nonlocal exec_context
            if exec_context is None:
                command_context = await self._session_factory.create_command_context(
                    msgs=inputs,
                    request=request,
                )
                exec_context = ExecutionContext(session_context=command_context)
            elif exec_context.session_context.command_handler is None:
                fallback_command_context = await self._session_factory.create_command_context(
                    msgs=inputs,
                    request=request,
                )
                exec_context.session_context.command_handler = fallback_command_context.command_handler
            return exec_context.session_context.command_handler

        async def _execute_command(query_text: str) -> str:
            command_handler = await _resolve_command_handler()
            return await command_handler.handle(query_text)

        def _command_turn_event(response_text: str) -> tuple[Any, bool]:
            return (
                langchain_message_to_turn_event(
                    _AIMessage(content=response_text),
                    turn_id=request.turn_id or turn_state.run_id,
                ),
                True,
            )

        try:
            # Initialize here so `finally` (line ~900) can always read them,
            # even when Stage 1 short-circuits before the stage-3 init below.
            assistant_completed_text: str = ""
            assistant_delta_text_parts: list[str] = []

            daily_talk_store = None
            _actor_meta: dict[str, str] = {}
            if DAILY_TALK_ENABLED:
                from lightclaw.app.runner.session.daily_talk_store import DailyTalkStore

                daily_talk_store = DailyTalkStore(working_dir=WORKING_DIR)
                # Extract ActorContext metadata for channel/session attribution
                _req_ctx = getattr(request, "context", None) or {}
                _actor_dict = _req_ctx.get("actor_context") if isinstance(_req_ctx, dict) else None
                if _actor_dict:
                    _actor_meta = {
                        "channel": str(_actor_dict.get("channel", "")),
                        "chat_type": str(_actor_dict.get("chat_type", "")),
                        "session": str(_actor_dict.get("session_id", "")),
                    }

            pending_user_texts = _extract_user_turn_texts(
                inputs,
                is_command_predicate=is_command,
            )
            if pending_user_texts:
                logger.info(
                    ">>> USER turn_id=%s | %r",
                    request.turn_id or turn_state.run_id,
                    _preview_text(pending_user_texts[0]),
                )
            # Do not write proactive/ephemeral messages into daily talk —
            # they are system-initiated, not genuine user conversation.
            _should_write_daily_talk = (
                daily_talk_store is not None
                and pending_user_texts
                and not turn_state.is_ephemeral
                and not turn_state.bootstrap_pending
            )
            if daily_talk_store is not None and pending_user_texts and turn_state.bootstrap_pending:
                logger.info(
                    "Skipping DAILY_TALK write during bootstrap: run_id=%s turn_id=%s session_id=%s texts=%d",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    len(pending_user_texts),
                )
            if _should_write_daily_talk:
                daily_talk_started_at = time.monotonic()
                try:
                    deferred_daily_write = await daily_talk_store.needs_rollover()
                    if not deferred_daily_write:
                        for text in pending_user_texts:
                            await daily_talk_store.append_user_message(
                                text, actor_meta=_actor_meta,
                            )
                    logger.info(
                        "Daily talk prepared: run_id=%s turn_id=%s session_id=%s deferred=%s texts=%d duration=%.2fs",
                        turn_state.run_id,
                        request.turn_id or turn_state.run_id,
                        turn_state.session_id or "-",
                        deferred_daily_write,
                        len(pending_user_texts),
                        time.monotonic() - daily_talk_started_at,
                    )
                except Exception:
                    logger.exception("Failed to append user message into DAILY_TALK.md")

            # ── Stage 1: Command short-circuit ──
            stage1_started_at = time.monotonic()
            last_msg = inputs[-1] if isinstance(inputs, list) else inputs
            query = last_msg.get_text_content() if hasattr(last_msg, "get_text_content") else None
            is_pure_command = bool(
                last_msg is not None and query and is_command(query) and not _message_has_file_or_media_blocks(last_msg)
            )

            if is_pure_command:
                command_started_at = time.monotonic()
                logger.info(
                    "Pure command short-circuit: run_id=%s turn_id=%s session_id=%s command=%r",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    query.strip(),
                )
                response_text = await _execute_command(query)
                yield _command_turn_event(response_text)
                logger.info(
                    "Pure command finished: run_id=%s turn_id=%s session_id=%s duration=%.2fs",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    time.monotonic() - command_started_at,
                )
                return
            logger.info(
                "Stage 1 command inspection passed: run_id=%s turn_id=%s session_id=%s query_present=%s duration=%.2fs",
                turn_state.run_id,
                request.turn_id or turn_state.run_id,
                turn_state.session_id or "-",
                bool(query),
                time.monotonic() - stage1_started_at,
            )

            # ── Stage 2: Full context for normal queries ──
            stage2_started_at = time.monotonic()
            if inputs is not None:
                logger.info(
                    "Stage 2 context build starting: run_id=%s turn_id=%s session_id=%s inputs=%d",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    len(inputs) if isinstance(inputs, list) else int(inputs is not None),
                )
                preprocess_started_at = time.monotonic()
                await process_file_and_media_blocks_in_message(inputs)
                logger.info(
                    "Stage 2 input preprocessing finished: run_id=%s turn_id=%s session_id=%s duration=%.2fs",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    time.monotonic() - preprocess_started_at,
                )

            session_context = await self._session_factory.create_langgraph_context(
                msgs=inputs,
                request=request,
            )
            exec_context = ExecutionContext(session_context=session_context)
            logger.info(
                "Stage 2 context ready: run_id=%s turn_id=%s session_id=%s history=%d summary_chars=%d chat_id=%s duration=%.2fs",
                turn_state.run_id,
                request.turn_id or turn_state.run_id,
                turn_state.session_id or "-",
                len(exec_context.history_messages),
                len(exec_context.compressed_summary or ""),
                getattr(exec_context.chat, "id", "-") if exec_context.chat is not None else "-",
                time.monotonic() - stage2_started_at,
            )
            # Ephemeral runs (proactivity / cron) still get a chat entry
            # so the user can see proactive messages in their conversation.
            # The ephemeral flag controls history handling (only persist final
            # AI message, filter SKIP, cap history) — NOT sidebar visibility.
            chat = exec_context.chat
            if deferred_daily_write and _should_write_daily_talk:
                try:
                    for text in pending_user_texts:
                        await daily_talk_store.append_user_message(
                            text, actor_meta=_actor_meta,
                        )
                except Exception:
                    logger.exception("Failed to append deferred user message into DAILY_TALK.md")

            # ── Command interception after full init ──
            if query and is_command(query):
                command_started_at = time.monotonic()
                logger.info(
                    "Command handled after full init: run_id=%s turn_id=%s session_id=%s command=%r",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    query,
                )
                response_text = await _execute_command(query)
                yield _command_turn_event(response_text)
                logger.info(
                    "Command after full init finished: run_id=%s turn_id=%s session_id=%s duration=%.2fs",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    time.monotonic() - command_started_at,
                )
                return

            # ── Stage 3: Execute Agent ──
            lc_messages = turn_inputs_to_langchain_messages(inputs if isinstance(inputs, list) else [inputs])
            # Prepend history (loaded from session store)
            all_messages = exec_context.history_messages + lc_messages

            # Pre-seed browser_control conversation context so that
            # browser sessions created during this run are correctly
            # bound to the conversation.  Use the Chat UUID (chat.id)
            # because the frontend queries browser sessions by chatId
            # which is the ChatSpec.id UUID — NOT the session_id field.
            _chat_uuid = (chat.id if chat is not None else "") or ""
            try:
                import lightclaw.agent.tools.browser_control as _bc

                _bc._pending_conversation_id = _chat_uuid
                _bc._pending_channel_source = turn_state.channel or "dashboard"
            except Exception:
                pass

            # Snapshot the original history before running the agent so
            # ephemeral runs can later reconstruct what to persist.
            if turn_state.is_ephemeral:
                turn_state.snapshot_pre_run_history(exec_context.history_messages)

            engine_session = exec_context.engine_session

            from lightclaw.app.observability import is_langfuse_enabled, langfuse_trace

            run_config = {
                "configurable": {
                    "thread_id": exec_context.session_context.session_id or turn_state.run_id,
                    "session_id": turn_state.session_id or "",
                    "user_id": turn_state.user_id or "",
                    "channel": turn_state.channel or "",
                    "chat_id": _chat_uuid,
                },
            }

            callbacks: list = []
            langfuse_enabled = is_langfuse_enabled()
            logger.info(
                "Stage 3 agent execution starting: run_id=%s turn_id=%s session_id=%s chat_id=%s history=%d input_messages=%d total_messages=%d langfuse=%s max_iterations=%s",
                turn_state.run_id,
                request.turn_id or turn_state.run_id,
                turn_state.session_id or "-",
                _chat_uuid or "-",
                len(exec_context.history_messages),
                len(lc_messages),
                len(all_messages),
                langfuse_enabled,
                getattr(engine_session, "_max_iterations", "-"),
            )
            engine_started_at = time.monotonic()
            estimated_output_tokens = 0
            assistant_delta_text_parts: list[str] = []
            assistant_completed_text = ""
            runtime_context = exec_context.session_context.runtime_context
            stream_supports_runtime_context = (
                runtime_context is not None and "runtime_context" in inspect.signature(engine_session.stream).parameters
            )

            def _build_stream_kwargs(callback_list: list[Any]) -> dict[str, Any]:
                kwargs: dict[str, Any] = {
                    "turn_id": request.turn_id or turn_state.run_id,
                    "run_config": run_config,
                    "callbacks": callback_list,
                }
                if stream_supports_runtime_context:
                    kwargs["runtime_context"] = runtime_context
                return kwargs

            def _accumulate_stream_event(event: Any) -> None:
                nonlocal assistant_completed_text, estimated_output_tokens
                if event.type == TurnEventType.ASSISTANT_COMPLETED:
                    completed_text = _extract_text_from_turn_content(getattr(event, "content", None))
                    if completed_text:
                        assistant_completed_text = completed_text
                    estimated_output_tokens += _estimate_content_tokens(getattr(event, "content", None))
                elif event.type == TurnEventType.ASSISTANT_DELTA:
                    delta = getattr(event, "delta", None)
                    metadata = getattr(event, "metadata", None)
                    message_type = str(metadata.get("message_type") or "") if isinstance(metadata, dict) else ""
                    if isinstance(delta, dict) and delta.get("type") == "text" and message_type != "reasoning":
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            assistant_delta_text_parts.append(text)
                elif event.type == TurnEventType.TOOL_STARTED:
                    estimated_output_tokens += _estimate_tool_call_event_tokens(getattr(event, "data", None))

            langfuse_cm = (
                langfuse_trace(
                    trace_name="agent_run",
                    trace_id_seed=request.turn_id or turn_state.run_id,
                    session_id=turn_state.session_id,
                    user_id=turn_state.user_id,
                    metadata={"channel": turn_state.channel, "run_id": turn_state.run_id},
                )
                if langfuse_enabled
                else nullcontext(None)
            )
            with langfuse_cm as langfuse_cb:
                callbacks = [langfuse_cb] if langfuse_cb is not None else []
                async for event in engine_session.stream(
                    all_messages,
                    **_build_stream_kwargs(callbacks),
                ):
                    _accumulate_stream_event(event)
                    yield event, event.type == TurnEventType.ASSISTANT_COMPLETED

            self.last_run_usage = engine_session.run_usage
            self.last_final_output_usage = engine_session.final_output_usage
            if self.last_run_usage:
                logger.info(
                    "Using provider run usage: run_id=%s turn_id=%s session_id=%s usage_source=%s usage=%s",
                    turn_state.run_id,
                    request.turn_id or turn_state.run_id,
                    turn_state.session_id or "-",
                    _usage_source(self.last_run_usage),
                    _usage_summary(self.last_run_usage),
                )
            if (
                not self.last_run_usage
                and exec_context.middleware is not None
                and hasattr(exec_context.middleware, "get_token_log_summary")
            ):
                try:
                    token_summary = exec_context.middleware.get_token_log_summary(
                        runtime_context=exec_context.session_context.runtime_context
                    )
                except Exception:
                    logger.exception("Failed to collect token diagnostics summary for usage fallback")
                else:
                    estimated_usage = _build_estimated_usage(
                        token_summary=token_summary,
                        estimated_output_tokens=estimated_output_tokens,
                    )
                    if estimated_usage:
                        self.last_run_usage = estimated_usage
                        logger.info(
                            "Using estimated run usage fallback: run_id=%s turn_id=%s session_id=%s usage_source=%s usage=%s",
                            turn_state.run_id,
                            request.turn_id or turn_state.run_id,
                            turn_state.session_id or "-",
                            _usage_source(self.last_run_usage),
                            _usage_summary(self.last_run_usage),
                        )

            if not self.last_final_output_usage and self.last_run_usage:
                self.last_final_output_usage = dict(self.last_run_usage)

            if engine_session.final_output_messages and self.last_run_usage:
                _attach_usage_to_last_assistant_message(
                    engine_session.final_output_messages,
                    self.last_run_usage,
                )
            logger.info(
                "Stage 3 agent execution finished: run_id=%s turn_id=%s session_id=%s duration=%.2fs final_messages=%d usage_source=%s usage=%s final_output_usage_source=%s final_output_usage=%s",
                turn_state.run_id,
                request.turn_id or turn_state.run_id,
                turn_state.session_id or "-",
                time.monotonic() - engine_started_at,
                len(engine_session.final_output_messages),
                _usage_source(self.last_run_usage),
                _usage_summary(self.last_run_usage),
                _usage_source(self.last_final_output_usage),
                _usage_summary(self.last_final_output_usage),
            )

            # After streaming completes, update history with the full
            # conversation (input + output) for session persistence.
            if engine_session.final_output_messages:
                if turn_state.is_ephemeral:
                    # Ephemeral run (proactivity / cron): only persist the
                    # final AI message so the proactive prompt and
                    # intermediate tool calls don't pollute the session.
                    # If the message is a [SKIP] or HEARTBEAT_OK marker, discard it entirely.
                    from langchain_core.messages import AIMessage

                    # Keep in sync with agent_task._SKIP_MARKERS.
                    _SKIP_MARKERS = {"[SKIP]", "HEARTBEAT_OK"}

                    final_ai_msgs = [
                        m
                        for m in engine_session.final_output_messages
                        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
                    ]
                    # Filter out skip-marker messages — they should not be persisted.
                    pre_filter_count = len(final_ai_msgs)
                    final_ai_msgs = [
                        m
                        for m in final_ai_msgs
                        if not any(marker in (getattr(m, "content", "") or "") for marker in _SKIP_MARKERS)
                    ]
                    if len(final_ai_msgs) < pre_filter_count:
                        logger.info(
                            "ephemeral run: filtered out %d skip-marker message(s)",
                            pre_filter_count - len(final_ai_msgs),
                        )
                    if final_ai_msgs:
                        new_history = [*turn_state.pre_run_history, final_ai_msgs[-1]]
                    else:
                        # No clean AI message — keep original history unchanged
                        new_history = turn_state.pre_run_history

                    # Cap ephemeral session history to avoid unbounded growth.
                    # Keep only the most recent messages.
                    _EPHEMERAL_MAX_HISTORY = 50
                    if len(new_history) > _EPHEMERAL_MAX_HISTORY:
                        new_history = new_history[-_EPHEMERAL_MAX_HISTORY:]
                        logger.info(
                            "ephemeral run: trimmed history to last %d messages",
                            _EPHEMERAL_MAX_HISTORY,
                        )

                    exec_context.update_history(new_history)
                    logger.info(
                        "ephemeral run: persisting %d history + %d new AI msg(s)",
                        len(turn_state.pre_run_history),
                        len(final_ai_msgs),
                    )
                else:
                    # Save the full uncompacted history immediately so the
                    # session file is written before the SSE connection closes.
                    # Background compaction will overwrite this with the
                    # compacted version while the user is reading the response.
                    exec_context.update_history(engine_session.final_output_messages)
            else:
                if turn_state.is_ephemeral:
                    # For ephemeral turns with no final model output, keep
                    # pre-run history unchanged.
                    exec_context.update_history(turn_state.pre_run_history)
                else:
                    # Some provider / middleware paths may stream deltas but
                    # produce no final_output_messages. Fall back to persisting
                    # the known input history plus streamed assistant text.
                    fallback_history = list(all_messages)
                    fallback_text = (assistant_completed_text or "".join(assistant_delta_text_parts)).strip()
                    if fallback_text:
                        fallback_history.append(AIMessage(content=fallback_text))
                        if self.last_run_usage:
                            _attach_usage_to_last_assistant_message(fallback_history, self.last_run_usage)
                    exec_context.update_history(fallback_history)
                    logger.warning(
                        "No final_output_messages from engine; fallback persistence used: "
                        "run_id=%s turn_id=%s session_id=%s input_messages=%d fallback_assistant_chars=%d "
                        "persisted_messages=%d",
                        turn_state.run_id,
                        request.turn_id or turn_state.run_id,
                        turn_state.session_id or "-",
                        len(all_messages),
                        len(fallback_text),
                        len(fallback_history),
                    )
            turn_state.mark_succeeded()

        except asyncio.CancelledError:
            # PR5: graceful cancel via middleware
            if exec_context is not None:
                exec_context.cancel()
            logger.warning(
                "Agent run cancelled: run_id=%s turn_id=%s session_id=%s",
                turn_state.run_id,
                request.turn_id or turn_state.run_id,
                turn_state.session_id or "-",
            )
            raise
        except Exception as exc:  # pylint: disable=broad-except
            error_info = classify_exception(exc)
            debug_dump_path = write_query_error_dump(
                request=request,
                exc=exc,
                locals_=locals(),
            )
            path_hint = f"\n(Details:  {debug_dump_path})" if debug_dump_path else ""
            logger.exception(
                "Error in LangGraph query handler [%s/%s]: %s%s",
                error_info.code,
                error_info.source,
                error_info.raw_message or exc,
                path_hint,
            )
            exc.lightclaw_error_info = error_info.to_event_error()
            if debug_dump_path:
                exc.debug_dump_path = debug_dump_path
                if hasattr(exc, "add_note"):
                    exc.add_note(f"(Details:  {debug_dump_path})")
                suffix = f"\n(Details:  {debug_dump_path})"
                exc.args = (f"{exc.args[0]}{suffix}" if exc.args else suffix.strip(), *exc.args[1:])
            raise
        finally:
            # Initialize singletons on first use
            RunController._initialize_singletons()

            # Orchestrate all finalization tasks
            if _turn_finalizer is not None:
                _turn_finalizer._session_factory = self._session_factory
                mm = self._session_factory.memory_manager
                _turn_finalizer._fact_extractor = mm.get_fact_extractor() if mm is not None else None
                # Build user/assistant text for fact extraction.
                # pending_user_texts is populated earlier in the generator; join
                # multiple inputs (rare) with a newline so the extractor sees them.
                _user_msg = "\n".join(pending_user_texts).strip() if pending_user_texts else ""
                _asst_msg = (assistant_completed_text or "".join(assistant_delta_text_parts)).strip()
                await _turn_finalizer.finalize(
                    turn_state=turn_state,
                    exec_context=exec_context,
                    request=request,
                    run_started_at=run_started_at,
                    run_usage=self.last_run_usage,
                    token_summary=token_summary,
                    format_token_diag_func=_format_token_diag_block,
                    usage_source_func=_usage_source,
                    usage_summary_func=_usage_summary,
                    user_message=_user_msg,
                    assistant_message=_asst_msg,
                )
