"""Shared cron agent execution: build request → run agent → dispatch to channel.

Both heartbeat and proactivity call ``run_cron_agent()`` so the
request-building / streaming / dispatch logic lives in one place.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from lightclaw.config import load_config
from lightclaw.constant import HEARTBEAT_TARGET_LAST, PROACTIVE_SESSION_PREFIX
from lightclaw.domain import TurnEvent, TurnEventType, TurnInput, TurnRequest
from lightclaw.domain.identity import build_canonical_user_id

from lightclaw.app.cron.turn_accumulator import AssistantTurnAccumulator

logger = logging.getLogger(__name__)


# Markers that indicate the agent decided nothing is worth pushing.
# HEARTBEAT_OK: agent found nothing to do during a heartbeat cycle.
# [SKIP]: explicit skip signal used by proactivity/cron agents.
_SKIP_MARKERS = {"[SKIP]", "HEARTBEAT_OK"}

# Regex to strip <thinking>…</thinking> blocks (including multiline).
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)

# Delta message types that are NOT text and should be skipped entirely.
# "thinking" / "reasoning" are handled separately via accumulator.add_thinking().
_SKIP_DELTA_TYPES = {
    "tool_call",
    "tool_result",
    "function_call",
    "function_call_output",
}


def _strip_thinking(text: str) -> str:
    """Remove <thinking>…</thinking> blocks and leading/trailing whitespace."""
    return _THINKING_RE.sub("", text).strip()


async def run_cron_agent(
    *,
    query_text: str,
    runner: Any,
    channel_manager: Any,
    config: Any | None = None,
    target: str = "main",
    session_id: str = "main",
    user_id: str = "main",
    source: str = "cron",
    timeout: int = 120,
    allowed_tools: list[str] | None = None,
    tool_policy_source: str | None = None,
    runtime_profile: str | None = None,
    max_input_length_override: int | None = None,
    max_iters_override: int | None = None,
) -> bool:
    """Run an agent with *query_text* and optionally dispatch to a channel.

    Returns ``True`` if a message was actually dispatched (or run-only
    completed), ``False`` if the agent output contained ``[SKIP]``.

    Args:
        query_text: The prompt text to send to the agent.
        runner: AgentRunner instance.
        channel_manager: ChannelManager instance.
        target: ``"main"`` (run only) or ``"last"`` (dispatch to last channel).
        session_id: Session ID for the request.
        user_id: User ID for the request.
        source: Origin label (``"heartbeat"``, ``"proactivity"``, ``"cron"``).
        timeout: Maximum seconds for the entire run.
        allowed_tools: If non-None, only these tool names are available to the agent.
        tool_policy_source: Label for tool policy (e.g. "heartbeat").
        runtime_profile: Runtime profile label (e.g. "heartbeat").
        max_input_length_override: Override max input tokens for this run.
        max_iters_override: Override max iterations for this run.
    """
    app_config = config if config is not None else load_config()

    ld = app_config.last_dispatch
    should_dispatch = target == HEARTBEAT_TARGET_LAST

    if should_dispatch:
        # Resolve dispatch target: use last_dispatch if available,
        # otherwise fall back to dashboard channel.
        dispatch_channel = ld.channel if ld and ld.channel else "dashboard"
        dispatch_user_id = (ld.user_id if ld else "") or user_id
        dispatch_session_id = (ld.session_id if ld else "") or session_id
    else:
        dispatch_channel = "dashboard"
        dispatch_user_id = user_id
        dispatch_session_id = session_id

    # Session isolation for proactivity and heartbeat: LLM runs use a separate
    # session to prevent mixing internal history with user chat history.
    # The LLM loads context from "proactive_{dispatch_session_id}",
    # but dispatch goes to "{dispatch_session_id}" (visible to user).
    if source in ("proactivity", "heartbeat"):
        llm_session_id = f"{PROACTIVE_SESSION_PREFIX}{dispatch_session_id}"
        logger.debug(
            "Session isolation: LLM uses %s, dispatch goes to %s",
            llm_session_id,
            dispatch_session_id,
        )
    else:
        llm_session_id = dispatch_session_id

    # Build canonical TurnRequest
    turn_request = TurnRequest(
        inputs=[
            TurnInput(
                role="user",
                content=[{"type": "text", "text": query_text}],
            ),
        ],
        session_id=llm_session_id,  # LLM loads context from this session
        user_id=dispatch_user_id,
        canonical_user_id=build_canonical_user_id(dispatch_channel, dispatch_user_id),
        channel=dispatch_channel,
        # Extra fields for runner internals
        source=source,
        ephemeral=True,
    )
    # Attach ActorContext for session/principal isolation firewall
    from lightclaw.agent.core.actor_context import ActorContext, actor_context_to_dict

    actor_ctx = ActorContext.system(
        job_id=session_id or "heartbeat",
        agent_id="default",
    )
    actor_ctx.channel = dispatch_channel
    actor_ctx.session_id = llm_session_id
    turn_request.context = turn_request.context or {}
    turn_request.context["actor_context"] = actor_context_to_dict(actor_ctx)
    # Attach tool policy + runtime budget overrides (P1b: heartbeat hard limits)
    if allowed_tools is not None:
        turn_request.context["allowed_tools"] = allowed_tools
    if tool_policy_source is not None:
        turn_request.context["tool_policy_source"] = tool_policy_source
    if runtime_profile is not None:
        turn_request.context["runtime_profile"] = runtime_profile
    if max_input_length_override is not None:
        turn_request.context["max_input_length_override"] = max_input_length_override
    if max_iters_override is not None:
        turn_request.context["max_iters_override"] = max_iters_override

    async def _run_and_maybe_dispatch() -> bool:
        """Collect turn output via AssistantTurnAccumulator, check for SKIP, then dispatch.

        The accumulator separates three streams:
        - delta_text: streaming display-only text (NOT authoritative for outbound)
        - thinking_text: reasoning content (NEVER enters outbound)
        - completed_text: authoritative final text from ASSISTANT_COMPLETED

        Outbound always uses ``accumulator.outbound_text`` which prefers
        completed_text and only falls back to delta_text with a WARNING.
        """
        accumulator = AssistantTurnAccumulator(turn_id="cron")
        completed_event: TurnEvent | None = None

        async for event in runner.stream_turns(turn_request):
            if event.type == TurnEventType.ASSISTANT_DELTA:
                meta = getattr(event, "metadata", None) or {}
                msg_type = meta.get("message_type", "")

                # Skip tool/fn call deltas entirely
                if msg_type in _SKIP_DELTA_TYPES:
                    continue

                # Extract text from delta
                delta = event.delta
                if isinstance(delta, str):
                    text = delta
                elif isinstance(delta, dict) and delta.get("type") == "text":
                    text = delta.get("text") or ""
                else:
                    text = ""

                # Route thinking / reasoning to separate accumulator bucket
                if msg_type in ("thinking", "reasoning"):
                    accumulator.add_thinking(text)
                else:
                    accumulator.add_delta(text)

            elif event.type in (TurnEventType.TOOL_STARTED, TurnEventType.TOOL_COMPLETED):
                # Record tool calls for audit only.
                # Do NOT reset delta_text — the accumulator relies on
                # completed_text as the authoritative source, so preamble
                # text before tool calls cannot pollute the final outbound.
                data = getattr(event, "data", None) or {}
                if event.type == TurnEventType.TOOL_STARTED:
                    accumulator.record_tool_call(dict(data) if isinstance(data, dict) else {})
                else:
                    accumulator.record_tool_result(dict(data) if isinstance(data, dict) else {})

            elif event.type == TurnEventType.ASSISTANT_COMPLETED:
                completed_event = event
                # Update accumulator identity from completed event
                if event.turn_id:
                    accumulator.turn_id = event.turn_id
                    accumulator.call_id = event.message_id

                # Extract authoritative completed text from content blocks.
                # Skip thinking/reasoning parts — they belong in thinking_text.
                parts: list[str] = []
                for part in event.content or []:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type", "")
                    if part_type in ("thinking", "reasoning"):
                        continue
                    if part_type == "text":
                        t = (part.get("text") or "").strip()
                        if t:
                            parts.append(t)
                completed = "".join(parts)
                accumulator.set_completed(completed, source="assistant_completed")

        # Assemble the final turn and extract outbound text.
        turn = accumulator.assemble()
        full_text = _strip_thinking(turn.outbound_text)
        logger.info("%s: agent output collected, len=%d", source, len(full_text))
        logger.debug(
            "%s: output content: %s", source,
            full_text[:200] if full_text else "(empty)",
        )
        logger.debug(
            "%s: accumulator summary delta=%d thinking=%d completed=%d source=%s",
            source,
            len(turn.delta_text),
            len(turn.thinking_text),
            len(turn.completed_text),
            turn.completed_source or "none",
        )

        # Check if agent decided nothing is worth pushing.
        if not full_text or any(m in full_text for m in _SKIP_MARKERS):
            logger.info(
                "%s: agent returned skip marker or empty, not dispatching",
                source,
            )
            return False

        # Quality gate: run outbound sanitizer before dispatch.
        from lightclaw.app.cron.outbound_sanitizer import OutboundMessageSanitizer

        sanitizer = OutboundMessageSanitizer()
        result = sanitizer.sanitize(full_text, context={"source": source})

        if result.should_block:
            logger.warning(
                "%s: outbound blocked: %s",
                source,
                result.block_reason,
            )
            return False

        full_text = result.text
        if result.actions:
            logger.info(
                "%s: sanitizer applied actions=%s, orig_len=%d -> cleaned_len=%d",
                source,
                result.actions,
                result.original_len,
                result.cleaned_len,
            )

        if not should_dispatch:
            logger.info("%s: run-only completed (no dispatch)", source)
            return True

        # Build a complete TurnEvent with the full text for dispatch.
        dispatch_event = TurnEvent(
            type=TurnEventType.ASSISTANT_COMPLETED,
            turn_id=completed_event.turn_id if completed_event else "cron",
            message_id=completed_event.message_id if completed_event else None,
            role="assistant",
            content=[{"type": "text", "text": full_text}],
            usage=completed_event.usage if completed_event else None,
        )

        try:
            await channel_manager.send_event(
                channel=dispatch_channel,
                user_id=dispatch_user_id,
                session_id=dispatch_session_id,
                event=dispatch_event,
                meta={},
            )
            logger.info("%s: dispatched to %s", source, dispatch_channel)
        except Exception:
            logger.exception("%s: send_event failed", source)

        # Ensure a chat entry exists so the message appears in the session list.
        try:
            await _ensure_chat_exists(
                runner=runner,
                channel=dispatch_channel,
                user_id=dispatch_user_id,
                session_id=dispatch_session_id,
                source=source,
            )
        except Exception:
            logger.debug("%s: ensure_chat_exists failed", source, exc_info=True)

        # Persist the message into the dispatch target session so it survives
        # page refresh. The LLM ran in an isolated session (proactive_xxx),
        # but the user sees dispatch_session_id — append there.
        try:
            await _persist_to_session(
                runner=runner,
                channel=dispatch_channel,
                user_id=dispatch_user_id,
                session_id=dispatch_session_id,
                text=full_text,
                source=source,
            )
        except Exception:
            logger.debug("%s: persist_to_session failed", source, exc_info=True)

        return True

    try:
        return await asyncio.wait_for(_run_and_maybe_dispatch(), timeout=timeout)
    except TimeoutError:
        logger.warning("%s run timed out", source)
    return False


async def _ensure_chat_exists(
    *,
    runner: Any,
    channel: str,
    user_id: str,
    session_id: str,
    source: str,
) -> None:
    """Create a chat entry if one doesn't already exist for this session.

    This ensures proactive/cron messages appear in the dashboard session list.
    """
    chat_manager = getattr(runner, "_chat_manager", None)
    if chat_manager is None:
        return

    # Build a default chat name based on source
    _SOURCE_NAMES = {
        "proactivity": "Proactive Chat",
        "heartbeat": "Heartbeat",
        "cron": "Scheduled Task",
    }
    name = _SOURCE_NAMES.get(source, "Chat")

    await chat_manager.get_or_create_chat(
        session_id=session_id,
        user_id=user_id,
        channel=channel,
        canonical_user_id=build_canonical_user_id(channel, user_id),
        name=name,
    )
    logger.info("%s: chat entry ensured for session=%s", source, session_id)


async def _persist_to_session(
    *,
    runner: Any,
    channel: str,
    user_id: str,
    session_id: str,
    text: str,
    source: str,
) -> None:
    """Append the proactive message to the dispatch target session file.

    This makes the message survive page refreshes. The LangGraph session
    store is loaded, the AI message is appended, and the file is saved back.
    """
    session = getattr(runner, "session", None)
    save_dir = getattr(session, "save_dir", None) or getattr(session, "_save_dir", None)
    if not save_dir:
        logger.debug("%s: no save_dir found on runner.session, skip persist", source)
        return

    from lightclaw.agent.core.engines.langgraph.session_store import LangGraphSessionStore

    store = LangGraphSessionStore(save_dir=save_dir)
    messages, summary = await store.aload(session_id, user_id, channel=channel)

    from langchain_core.messages import AIMessage

    messages.append(AIMessage(content=text))

    await store.asave(
        session_id,
        user_id,
        messages,
        summary,
        channel=channel,
        canonical_user_id=build_canonical_user_id(channel, user_id),
    )
    logger.info("%s: persisted message to session %s", source, session_id)
