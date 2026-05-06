"""LangGraph EventBridge — streams LangGraph events as turn-native events.

This module consumes the ``astream_events`` (v2) stream from a compiled
LangGraph graph and yields canonical ``TurnEvent`` objects.  Legacy
``Message`` / process-protocol projection happens later at the app
boundary.

Event mapping (LangGraph v2 → TurnEvent):
    on_chat_model_stream   → assistant.delta
    on_tool_start          → tool.started
    on_tool_end            → tool.completed
    on_chain_end(LightClaw)→ assistant.completed

Usage:
    >>> bridge = LangGraphEventBridge()
    >>> async for event in bridge.stream(graph, messages, turn_id="turn-1"):
    ...     print(event.type)
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from copy import deepcopy
from typing import Any
from uuid import uuid4

from langgraph.errors import GraphRecursionError

from lightclaw.agent.core.adapters.turn_protocol import langchain_message_to_turn_event
from lightclaw.agent.core.engines.langgraph.event_bridge._think_processor import ThinkStreamProcessor
from lightclaw.agent.core.engines.langgraph.event_bridge._tool_output import _extract_tool_result_payload
from lightclaw.agent.core.engines.langgraph.event_bridge._usage_aggregator import _coerce_usage, _UsageAggregator
from lightclaw.agent.core.interfaces import BaseEventBridge
from lightclaw.domain import TurnEvent, TurnEventType
from lightclaw.domain.message_types import MessageType

logger = logging.getLogger(__name__)


class LangGraphEventBridge(BaseEventBridge):
    """Bridge LangGraph ``astream_events`` to canonical turn events.

    The execution path stays LangChain-native; any legacy protocol
    projection happens outside this bridge.
    """

    # Type aliases for the two injected security hooks
    _GuardCheckFn = Callable[..., Coroutine[Any, Any, "TurnEvent | None"]]
    _GuardWaitFn = Callable[..., Coroutine[Any, Any, str]]

    def __init__(
        self,
        *,
        emit_tool_events: bool = False,
        stream_version: str = "v2",
        tool_guard_check: _GuardCheckFn | None = None,
        tool_guard_wait: _GuardWaitFn | None = None,
    ) -> None:
        """
        Args:
            emit_tool_events: If ``True``, tool start/end events are
                also yielded as Msg.  Defaults to ``False`` (only LLM
                stream tokens and the final message are emitted).
            stream_version: astream_events version. ``"v2"`` recommended.
            tool_guard_check: Optional async callable injected by the app
                layer to perform security scans before each tool call.
                Signature: (tool_name, tool_input, turn_id, run_id,
                session_id, user_id, channel) -> TurnEvent | None.
            tool_guard_wait: Optional async callable injected by the app
                layer to await a human approval decision.
                Signature: (session_id, tool_name, timeout) -> str.
        """
        self._emit_tool_events = emit_tool_events
        self._stream_version = stream_version
        self._tool_guard_check = tool_guard_check
        self._tool_guard_wait = tool_guard_wait
        # PR5: capture final output messages for session persistence
        self.final_output_messages: list = []
        self.final_output_usage: dict[str, Any] | None = None
        self.run_usage: dict[str, Any] | None = None
        # Track whether tool events have been streamed so we can avoid
        # duplicating them in the final ``on_chain_end`` message.
        self._streamed_tool_events: bool = False
        self._usage_aggregator = _UsageAggregator()
        self._current_stream_message_id: str | None = None
        self._current_stream_message_type: str | None = None
        self._think_processor = ThinkStreamProcessor(self._start_stream_message)
        self._tool_started_at: dict[str, float] = {}
        self._chat_model_started_at: dict[str, float] = {}
        self._chat_model_chunks: dict[str, int] = {}
        self._chat_model_text_chars: dict[str, int] = {}
        self._chat_model_reasoning_chars: dict[str, int] = {}
        self._chat_model_meta: dict[str, dict[str, str]] = {}
        self._chat_model_first_token_logged: set[str] = set()

    def _start_stream_message(self, message_type: str) -> str:
        if self._current_stream_message_type != message_type:
            self._current_stream_message_id = str(uuid4())
            self._current_stream_message_type = message_type
        elif self._current_stream_message_id is None:
            self._current_stream_message_id = str(uuid4())
        return self._current_stream_message_id

    def _reset_stream_message(self) -> None:
        self._current_stream_message_id = None
        self._current_stream_message_type = None

    @staticmethod
    def _truncate_preview(text: str, *, limit: int = 160) -> str:
        """Return a single-line bounded preview for logs."""
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    @classmethod
    def _message_preview(cls, message: Any) -> str:
        """Extract visible text from a message-like object for logs."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return cls._truncate_preview(content)
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
        return cls._truncate_preview("\n".join(parts))

    @staticmethod
    def _concat_text_blocks(blocks: Sequence[Any]) -> str:
        """Concatenate visible text from text blocks in a turn event payload."""
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

    @staticmethod
    def _extract_model_meta(event: dict[str, Any]) -> dict[str, str]:
        """Extract provider/model identity from LangChain callback metadata."""
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        model_name = metadata.get("ls_model_name") or metadata.get("model_name") or metadata.get("model")
        provider = metadata.get("ls_provider") or metadata.get("provider")
        invocation = event.get("name") or metadata.get("ls_model_type") or "chat_model"
        return {
            "model": str(model_name or invocation),
            "provider": str(provider or "-"),
            "invocation": str(invocation),
        }

    def _log_first_token(self, *, turn_id: str, run_id: str, default_started_at: float) -> None:
        """Emit a first-token latency log once per chat-model run."""
        if run_id in self._chat_model_first_token_logged:
            return
        self._chat_model_first_token_logged.add(run_id)
        started_at = self._chat_model_started_at.get(run_id, default_started_at)
        meta = self._chat_model_meta.get(run_id, {"model": "-", "provider": "-", "invocation": "-"})
        logger.info(
            "[ModelStream][first_token] turn=%s model_run_id=%s model=%s provider=%s latency=%.2fs",
            turn_id,
            run_id or "-",
            meta.get("model", "-"),
            meta.get("provider", "-"),
            time.monotonic() - started_at,
        )

    # ------------------------------------------------------------------
    # ToolGuard integration
    # ------------------------------------------------------------------

    async def _check_tool_guard(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        turn_id: str,
        run_id: str,
        session_id: str,
        user_id: str,
        channel: str,
    ) -> TurnEvent | None:
        """Delegate tool guard scan to the injected app-layer hook.

        Returns a TOOL_APPROVAL_REQUIRED event when intervention is
        needed, or None when execution should proceed.  If no hook was
        injected the engine layer has no security knowledge and all
        tool calls are allowed through.
        """
        if self._tool_guard_check is None:
            return None
        return await self._tool_guard_check(
            tool_name=tool_name,
            tool_input=tool_input,
            turn_id=turn_id,
            run_id=run_id,
            session_id=session_id,
            user_id=user_id,
            channel=channel,
        )

    async def _wait_for_approval(
        self,
        *,
        session_id: str,
        tool_name: str,
        timeout: float = 300.0,
    ) -> str:
        """Delegate approval wait to the injected app-layer hook.

        Returns "approved", "denied", or "timeout".  Falls back to
        "denied" when no hook is injected.
        """
        if self._tool_guard_wait is None:
            return "denied"
        return await self._tool_guard_wait(
            session_id=session_id,
            tool_name=tool_name,
            timeout=timeout,
        )

    async def stream(
        self,
        graph: Any,
        messages: Sequence[Any],
        *,
        turn_id: str,
        config: dict | None = None,
        context: Any | None = None,
        max_iterations: int | None = None,
    ) -> AsyncIterator[TurnEvent]:
        """Stream a LangGraph agent run, yielding canonical ``TurnEvent`` objects.

        Args:
            graph: A compiled LangGraph graph.
            messages: Input LangChain messages.
            turn_id: Turn identifier propagated through the event stream.
        """
        self.final_output_messages = []
        self.final_output_usage = None
        self.run_usage = None
        self._streamed_tool_events = False
        self._usage_aggregator.reset()
        self._reset_stream_message()
        self._think_processor.reset()
        self._tool_started_at = {}
        self._chat_model_started_at = {}
        self._chat_model_chunks = {}
        self._chat_model_text_chars = {}
        self._chat_model_reasoning_chars = {}
        self._chat_model_meta = {}
        self._chat_model_first_token_logged = set()

        final_result: dict | None = None
        text_buffer: list[str] = []
        model_runs_seen: set[str] = set()

        # Convert max_iterations to LangGraph recursion_limit and inject into config.
        # Each ReAct cycle: agent node + tools node = 2 steps, plus 1 for the final agent output.
        run_config = dict(config or {})
        if max_iterations is not None:
            run_config["recursion_limit"] = max_iterations * 2 + 1
            logger.debug(
                "Setting recursion_limit=%d (max_iterations=%d)",
                run_config["recursion_limit"],
                max_iterations,
            )

        _t_stream_start = time.monotonic()
        logger.info("LLM stream started (astream_events, turn_id=%s)", turn_id)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                async for event in graph.astream_events(
                    {"messages": list(messages)},
                    config=run_config,
                    context=context,
                    version=self._stream_version,
                ):
                    kind = event.get("event", "")
                    run_id = str(event.get("run_id", "") or "")
                    if kind.startswith("on_chat_model"):
                        model_runs_seen.add(run_id)

                    if kind == "on_chat_model_start":
                        # Each model call starts a new message — clear the
                        # streaming text buffer so drop_completed_text
                        # compares only the current call's text against the
                        # final AIMessage.  Without this, text from prior
                        # tool-calling turns leaks into streamed_text and
                        # prevents dedup, duplicating the final reply.
                        text_buffer.clear()
                        meta = self._extract_model_meta(event)
                        self._chat_model_started_at[run_id] = time.monotonic()
                        self._chat_model_chunks[run_id] = 0
                        self._chat_model_text_chars[run_id] = 0
                        self._chat_model_reasoning_chars[run_id] = 0
                        self._chat_model_meta[run_id] = meta
                        logger.info(
                            "[ModelStream][start] turn_id=%s model_run_id=%s model=%s provider=%s invocation=%s",
                            turn_id,
                            run_id or "-",
                            meta.get("model", "-"),
                            meta.get("provider", "-"),
                            meta.get("invocation", "-"),
                        )

                    # ── LLM streaming token ──
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk is None:
                            continue
                        self._usage_aggregator.record_stream_chunk(
                            run_id,
                            chunk,
                        )
                        self._chat_model_chunks[run_id] = self._chat_model_chunks.get(run_id, 0) + 1

                        content = getattr(chunk, "content", None)
                        if content and isinstance(content, str):
                            self._chat_model_text_chars[run_id] = self._chat_model_text_chars.get(run_id, 0) + len(
                                content
                            )
                            self._log_first_token(
                                turn_id=turn_id,
                                run_id=run_id,
                                default_started_at=_t_stream_start,
                            )
                            # Accumulate into think_buffer to handle
                            # <think> tags that span across chunks.
                            async for te in self._think_processor.process(content, turn_id, text_buffer):
                                yield te

                        # Handle thinking blocks in streaming.
                        # "thinking" = Anthropic native extended thinking.
                        # "reasoning_content" = OpenAI-compatible models (DeepSeek-R1,
                        #   GLM-5, Qwen-QwQ, etc.) that expose reasoning as a separate
                        #   delta field rather than inline <think> tags.
                        extra = getattr(chunk, "additional_kwargs", {}) or {}
                        thinking = extra.get("thinking") or extra.get("reasoning_content")
                        if thinking:
                            if isinstance(thinking, str):
                                self._chat_model_reasoning_chars[run_id] = self._chat_model_reasoning_chars.get(
                                    run_id, 0
                                ) + len(thinking)
                                self._log_first_token(
                                    turn_id=turn_id,
                                    run_id=run_id,
                                    default_started_at=_t_stream_start,
                                )
                            yield TurnEvent(
                                type=TurnEventType.ASSISTANT_DELTA,
                                turn_id=turn_id,
                                message_id=self._start_stream_message(MessageType.REASONING),
                                role="assistant",
                                delta={"type": "text", "text": thinking},
                                metadata={"message_type": MessageType.REASONING},
                            )

                    elif kind == "on_chat_model_end":
                        # Flush any remaining think buffer when the model
                        # finishes so partial <think> blocks are not lost.
                        async for te in self._think_processor.flush(turn_id, text_buffer):
                            yield te

                        self._usage_aggregator.record_final_message(
                            run_id,
                            event.get("data", {}).get("output"),
                        )
                        output = event.get("data", {}).get("output")
                        meta = self._chat_model_meta.pop(run_id, self._extract_model_meta(event))
                        started_at = self._chat_model_started_at.pop(run_id, _t_stream_start)
                        chunks = self._chat_model_chunks.pop(run_id, 0)
                        text_chars = self._chat_model_text_chars.pop(run_id, 0)
                        reasoning_chars = self._chat_model_reasoning_chars.pop(run_id, 0)
                        usage = _coerce_usage(getattr(output, "usage_metadata", None))
                        logger.info(
                            "[ModelStream][end] turn_id=%s model_run_id=%s model=%s provider=%s duration=%.2fs "
                            "chunks=%d text_chars=%d reasoning_chars=%d tool_calls=%d usage=%s preview=%r",
                            turn_id,
                            run_id or "-",
                            meta.get("model", "-"),
                            meta.get("provider", "-"),
                            time.monotonic() - started_at,
                            chunks,
                            text_chars,
                            reasoning_chars,
                            len(getattr(output, "tool_calls", []) or []),
                            usage or {},
                            self._message_preview(output),
                        )

                        # Detect content_filter finish_reason and emit a
                        # dedicated warning so operators can diagnose quickly.
                        _resp_meta = getattr(output, "response_metadata", None)
                        if isinstance(_resp_meta, dict):
                            _finish = _resp_meta.get("finish_reason", "")
                            if _finish in {"content_filter", "content_filtered", "sensitive"}:
                                logger.warning(
                                    "[ModelStream][content_filter] turn_id=%s model=%s "
                                    "finish_reason=%s — model safety policy triggered, "
                                    "reply may be a generic refusal. preview=%r",
                                    turn_id,
                                    meta.get("model", "-"),
                                    _finish,
                                    self._message_preview(output),
                                )

                    # ── Tool execution events ──
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        tool_run_id = str(event.get("run_id", "") or "")
                        tool_input = event.get("data", {}).get("input", {})
                        tool_input_dict = tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)}
                        self._tool_started_at[tool_run_id] = time.monotonic()
                        logger.info(
                            "Tool started: turn_id=%s tool=%s run_id=%s",
                            turn_id,
                            tool_name,
                            tool_run_id or "-",
                        )

                        # ToolGuard intercept: check parameter safety before tool execution
                        guard_event = await self._check_tool_guard(
                            tool_name=tool_name,
                            tool_input=tool_input_dict,
                            turn_id=turn_id,
                            run_id=tool_run_id,
                            session_id=config.get("configurable", {}).get("session_id", "") if config else "",
                            user_id=config.get("configurable", {}).get("user_id", "") if config else "",
                            channel=config.get("configurable", {}).get("channel", "") if config else "",
                        )
                        if guard_event is not None:
                            # Tool intercepted: yield TOOL_APPROVAL_REQUIRED then pause
                            yield guard_event
                            # Wait for approval (blocks until user /approve or denial)
                            decision = await self._wait_for_approval(
                                session_id=config.get("configurable", {}).get("session_id", "") if config else "",
                                tool_name=tool_name,
                            )
                            if decision != "approved":
                                # Approval denied or timed out — abort current turn
                                return
                            # Approval granted — continue normally (skip TOOL_STARTED)

                        if self._emit_tool_events:
                            self._streamed_tool_events = True
                            self._reset_stream_message()
                            yield TurnEvent(
                                type=TurnEventType.TOOL_STARTED,
                                turn_id=turn_id,
                                data={
                                    "call_id": event.get("run_id", ""),
                                    "name": tool_name,
                                    "arguments": tool_input_dict,
                                },
                            )

                    elif kind == "on_tool_end" and self._emit_tool_events:
                        self._reset_stream_message()
                        tool_run_id = str(event.get("run_id", "") or "")
                        started_at = self._tool_started_at.pop(tool_run_id, None)
                        tool_output, tool_metadata = _extract_tool_result_payload(
                            event.get("data", {}).get("output", "")
                        )
                        tool_payload: dict[str, Any] = {
                            "call_id": tool_run_id,
                            "name": event.get("name", ""),
                            "output": tool_output,
                        }
                        if tool_metadata:
                            tool_payload.update(tool_metadata)
                        logger.info(
                            "Tool completed: turn_id=%s tool=%s run_id=%s duration=%s",
                            turn_id,
                            event.get("name", "") or "unknown",
                            tool_run_id or "-",
                            f"{time.monotonic() - started_at:.2f}s" if started_at is not None else "-",
                        )
                        yield TurnEvent(
                            type=TurnEventType.TOOL_COMPLETED,
                            turn_id=turn_id,
                            data=tool_payload,
                        )

                    # ── Graph execution complete ──
                    elif kind == "on_chain_end" and event.get("name") == "LightClaw":
                        output = event.get("data", {}).get("output", {})
                        final_messages = output.get("messages", [])
                        if final_messages:
                            final_result = final_messages[-1]
                            # PR5: capture all output messages for persistence
                            self.final_output_messages = list(final_messages)

        except GraphRecursionError:
            self.run_usage = self._usage_aggregator.run_usage
            self.final_output_usage = self._usage_aggregator.last_usage
            # Reached max iteration limit; return a user-friendly guidance message
            logger.warning(
                "GraphRecursionError: reached recursion_limit (max_iterations=%s)",
                max_iterations,
            )
            yield TurnEvent(
                type=TurnEventType.ASSISTANT_COMPLETED,
                turn_id=turn_id,
                message_id=self._current_stream_message_id or str(uuid4()),
                role="assistant",
                content=[
                    {
                        "type": "text",
                        "text": (
                            "⚠️ 已达到最大迭代次数限制，任务未能在限定步骤内完成。\n\n"
                            "**建议：**\n"
                            "- 尝试将任务拆分为更小的子任务后重新提问\n"
                            "- 在「高级设置 → 运行配置」中适当增大「最大迭代次数」\n"
                            "- 或者换一种更简洁的方式描述你的需求"
                        ),
                    }
                ],
                usage=self.final_output_usage,
            )
            return

        self.run_usage = self._usage_aggregator.run_usage
        logger.info(
            "LLM stream finished: turn_id=%s duration=%.2fs model_runs=%d usage=%s final_output=%s",
            turn_id,
            time.monotonic() - _t_stream_start,
            len([run_id for run_id in model_runs_seen if run_id]),
            self.run_usage or {},
            final_result is not None,
        )

        # ── Yield final assistant event ──
        if final_result is not None:
            final_event = langchain_message_to_turn_event(
                final_result,
                turn_id=turn_id,
                message_id=self._current_stream_message_id,
            )
            final_usage = _coerce_usage(final_event.usage)
            if final_usage is None and final_event.role == "assistant":
                final_usage = self._usage_aggregator.last_usage
                if final_usage is not None:
                    final_event = final_event.model_copy(update={"usage": deepcopy(final_usage)})
            self.final_output_usage = deepcopy(final_usage) if final_usage else None

            streamed_text = "".join(text_buffer) if text_buffer else ""
            completed_text = self._concat_text_blocks(final_event.content)
            drop_completed_text = bool(streamed_text) and completed_text == streamed_text
            if streamed_text and not drop_completed_text:
                logger.debug(
                    "Preserving completed text because streamed text differs: turn_id=%s streamed_chars=%d completed_chars=%d",
                    turn_id,
                    len(streamed_text),
                    len(completed_text),
                )

            if streamed_text or self._streamed_tool_events:
                filtered_blocks: list[dict[str, Any]] = []
                for block in final_event.content:
                    if not isinstance(block, dict):
                        filtered_blocks.append(block)
                        continue
                    btype = block.get("type", "")
                    if btype == "text" and drop_completed_text:
                        continue
                    if btype in ("tool_use", "tool_result") and self._streamed_tool_events:
                        continue
                    filtered_blocks.append(block)

                final_event = final_event.model_copy(update={"content": filtered_blocks})

            yield final_event
        else:
            full_text = "".join(text_buffer) if text_buffer else ""
            self.final_output_usage = self._usage_aggregator.last_usage
            yield TurnEvent(
                type=TurnEventType.ASSISTANT_COMPLETED,
                turn_id=turn_id,
                message_id=self._current_stream_message_id or str(uuid4()),
                role="assistant",
                content=[{"type": "text", "text": full_text}],
                usage=self.final_output_usage,
            )
        self._reset_stream_message()

    async def stream_agent_run(
        self,
        *,
        graph: Any,
        messages: Sequence[Any],
        turn_id: str,
        config: dict | None = None,
        max_iterations: int | None = None,
    ) -> AsyncIterator[TurnEvent]:
        """Compatibility alias for the turn-native stream method."""
        async for event in self.stream(
            graph,
            messages,
            turn_id=turn_id,
            config=config,
            max_iterations=max_iterations,
        ):
            yield event
