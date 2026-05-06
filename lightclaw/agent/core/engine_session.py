"""LangGraph implementation of the EngineSession interface.

``LangGraphEngineSession`` wraps a compiled ``CompiledStateGraph`` together
with its ``LangGraphEventBridge`` so that ``run_controller`` can execute an
agent run through the engine-agnostic ``EngineSession`` interface without
ever touching LangGraph-specific types directly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from lightclaw.agent.core.interfaces import EngineSession
from lightclaw.domain import TurnEventType

logger = logging.getLogger(__name__)


class LangGraphEngineSession(EngineSession):
    """Execute a LangGraph ReAct agent run through the ``EngineSession`` interface.

    Args:
        graph:              Compiled ``CompiledStateGraph`` returned by
                            ``LangGraphAgentBuilder.build()``.
        max_iterations:     Maximum ReAct loop iterations before aborting.
        tool_guard_check:   Callable injected from ``app.security.hooks``
                            (kept opaque to avoid upward imports here).
        tool_guard_wait:    Callable injected from ``app.security.hooks``.
    """

    def __init__(
        self,
        graph: Any,
        max_iterations: int,
        *,
        tool_guard_check: Any = None,
        tool_guard_wait: Any = None,
        memory_manager: Any = None,
        turn_model: Any = None,
    ) -> None:
        self._graph = graph
        self._max_iterations = max_iterations
        self._tool_guard_check = tool_guard_check
        self._tool_guard_wait = tool_guard_wait
        self._memory_manager = memory_manager
        self._turn_model = turn_model
        self._bridge: Any | None = None  # LangGraphEventBridge, set after stream()

    async def stream(
        self,
        messages: list,
        *,
        turn_id: str,
        run_config: dict | None = None,
        runtime_context: Any | None = None,
        callbacks: list | None = None,
    ) -> AsyncIterator[Any]:
        """Stream the agent run, yielding ``TurnEvent`` objects.

        Constructs a fresh ``LangGraphEventBridge`` per run (bridges are
        stateful and must not be reused across requests), then delegates to
        ``bridge.stream()``.  After the generator is exhausted, usage stats
        and final messages are available via the read-only properties.
        """
        from lightclaw.agent.core.engines.langgraph.event_bridge import LangGraphEventBridge

        bridge = LangGraphEventBridge(
            emit_tool_events=True,
            tool_guard_check=self._tool_guard_check,
            tool_guard_wait=self._tool_guard_wait,
        )

        config: dict = dict(run_config or {})
        if callbacks:
            config["callbacks"] = callbacks
        configurable = config.get("configurable", {}) if isinstance(config.get("configurable"), dict) else {}

        started_at = time.monotonic()
        model_token = None
        if (
            self._memory_manager is not None
            and self._turn_model is not None
            and hasattr(self._memory_manager, "bind_turn_chat_model")
        ):
            model_token = self._memory_manager.bind_turn_chat_model(self._turn_model)
        first_token_logged = False
        logger.info(
            "LangGraph engine stream starting: turn_id=%s session_id=%s thread_id=%s chat_id=%s messages=%d callbacks=%d max_iterations=%d",
            turn_id,
            configurable.get("session_id", "-") or "-",
            configurable.get("thread_id", "-") or "-",
            configurable.get("chat_id", "-") or "-",
            len(messages),
            len(callbacks or []),
            self._max_iterations,
        )

        # ── P2b: per-run max_iters override (heartbeat uses HEARTBEAT_MAX_AGENT_ITERS=5) ─
        max_iters = self._max_iterations
        if runtime_context is not None:
            override = getattr(runtime_context, "max_iters_override", None)
            if override is not None and isinstance(override, int) and override > 0:
                max_iters = override
                logger.info(
                    "runtime_budget: profile=%s max_iters=%d (global_default=%d) max_input=%s",
                    getattr(runtime_context, "runtime_profile", "unknown"),
                    max_iters,
                    self._max_iterations,
                    getattr(runtime_context, "max_input_length_override", "-"),
                )

        try:
            async for event in bridge.stream(
                self._graph,
                messages,
                turn_id=turn_id,
                config=config,
                context=runtime_context,
                max_iterations=max_iters,
            ):
                if not first_token_logged and getattr(event, "type", "") == TurnEventType.ASSISTANT_DELTA:
                    first_token_logged = True
                    logger.info(
                        "[Metrics] first_token_ms=%.2f turn_id=%s session_id=%s",
                        (time.monotonic() - started_at) * 1000,
                        turn_id,
                        configurable.get("session_id", "-") or "-",
                    )
                yield event
        finally:
            if (
                self._memory_manager is not None
                and model_token is not None
                and hasattr(self._memory_manager, "reset_turn_chat_model")
            ):
                self._memory_manager.reset_turn_chat_model(model_token)
            if not first_token_logged:
                logger.info(
                    "[Metrics] first_token_ms=missing turn_id=%s session_id=%s",
                    turn_id,
                    configurable.get("session_id", "-") or "-",
                )
            # Retain bridge reference so post-stream properties are readable.
            self._bridge = bridge
            logger.info(
                "LangGraph engine stream finished: turn_id=%s session_id=%s thread_id=%s chat_id=%s duration=%.2fs final_messages=%d usage=%s",
                turn_id,
                configurable.get("session_id", "-") or "-",
                configurable.get("thread_id", "-") or "-",
                configurable.get("chat_id", "-") or "-",
                time.monotonic() - started_at,
                len(self.final_output_messages),
                self.run_usage or {},
            )

    @property
    def final_output_messages(self) -> list:
        return self._bridge.final_output_messages if self._bridge else []

    @property
    def run_usage(self) -> dict[str, Any] | None:
        return self._bridge.run_usage if self._bridge else None

    @property
    def final_output_usage(self) -> dict[str, Any] | None:
        return self._bridge.final_output_usage if self._bridge else None
