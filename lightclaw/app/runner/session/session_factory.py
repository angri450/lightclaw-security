"""Session factory for constructing per-run session context.

Single responsibility: orchestrate the assembly of a SessionContext
by delegating to focused helpers:

* model_routing         — resolves command model for slash commands
* LangGraphSessionStore — loads / saves session history
* LightClawMiddleware   — shared middleware pipeline (runtime context driven)
* LightClawRuntimeContext — per-turn mutable runtime state
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lightclaw.agent.bootstrap_state import is_bootstrap_pending
from lightclaw.app.channels.core.schema import DEFAULT_CHANNEL
from lightclaw.app.runner.session.model_router import LangGraphModelRouter
from lightclaw.app.runner.session.session import SafeSession
from lightclaw.constant import (
    MEMORY_COMPACT_KEEP_RECENT,
    MEMORY_COMPACT_RATIO,
    MEMORY_DIR,
    PROACTIVE_SESSION_PREFIX,
    WORKING_DIR,
    is_memory_manager_enabled,
)
from lightclaw.domain.identity import build_canonical_user_id

if TYPE_CHECKING:
    from lightclaw.agent.core.command_handler import BaseCommandHandler
    from lightclaw.agent.core.engines.langgraph.runtime_context import LightClawRuntimeContext
    from lightclaw.agent.core.interfaces import BaseMiddleware, EngineSession, SessionRepository

logger = logging.getLogger(__name__)


def _describe_model(model: Any) -> str:
    """Return a stable human-readable model identifier for logs."""
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return type(model).__name__


@dataclass
class SessionContext:
    """Container for a single agent run.

    ``engine_session``  — engine-agnostic handle for normal agent runs.
    ``command_handler`` — engine-agnostic handler for slash commands.
    Both are built by ``SessionFactory`` so ``RunController`` never imports
    concrete engine classes.
    """

    session_id: str | None
    user_id: str | None
    peer_user_id: str | None
    canonical_user_id: str | None
    channel: str
    env_context: str
    chat: Any | None = None
    history_messages: list = field(default_factory=list)
    middleware: BaseMiddleware | None = None
    session_store: SessionRepository | None = None
    compressed_summary: str = ""
    runtime_context: LightClawRuntimeContext | None = None
    fact_extractor: Any | None = None
    fact_store: Any | None = None
    engine_session: EngineSession | None = None
    command_handler: BaseCommandHandler | None = None


class SessionFactory:
    session_store: SessionRepository | None = None
    compressed_summary: str = ""
    fact_extractor: Any | None = None
    fact_store: Any | None = None
    engine_session: EngineSession | None = None
    command_handler: BaseCommandHandler | None = None

    """Orchestrates per-run session context assembly.

    All heavy lifting is delegated to focused helpers.  This class only
    wires them together and fills ``SessionContext``.
    """

    def __init__(
        self,
        *,
        session: SafeSession,
        memory_manager: Any | None,
        chat_manager: Any | None,
        config: Any | None = None,
        middleware: BaseMiddleware | None = None,
        engine_session: EngineSession | None = None,
    ) -> None:
        self._session = session
        self._memory_manager = memory_manager
        self._chat_manager = chat_manager
        # Config injected from the request entry point so the entire request
        # sees one consistent snapshot.  Falls back to load_config() only when
        # not provided (e.g. in tests or legacy call sites).
        self._config = config
        self._middleware = middleware
        self._engine_session = engine_session

    @property
    def memory_manager(self) -> Any | None:
        """Expose memory manager for callers that need to pass it downstream."""
        return self._memory_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_command_context(
        self,
        *,
        msgs,
        request: Any,
    ) -> SessionContext:
        """Build the minimal context required for pure slash commands."""
        from lightclaw.agent.core.engines.langgraph.command_handler import LangGraphCommandHandler
        from lightclaw.agent.core.engines.langgraph.middleware import LightClawMiddleware
        from lightclaw.config import load_config

        session_id = request.session_id
        user_id = request.user_id
        peer_user_id = getattr(request, "peer_user_id", "") or ""
        channel = request.channel or DEFAULT_CHANNEL
        canonical_user_id = getattr(request, "canonical_user_id", "") or build_canonical_user_id(channel, user_id or "")
        started_at = time.monotonic()
        app_config = self._config if self._config is not None else load_config()
        language = "en" if app_config.agents.language == "en" else "zh"
        model = None
        if self._memory_manager is not None:
            # Commands such as /compact rely on MemoryManager LLM calls.
            # Resolve the per-turn model so command execution can bind it
            # without mutating shared global model pointers.
            from lightclaw.app.services.providers_service import load_providers_json as _load_providers

            try:
                providers_data = _load_providers()
                model = LangGraphModelRouter().select(msgs, providers_data)
            except Exception:
                logger.exception("Failed to resolve command model; memory commands may degrade gracefully.")

        session_store = self._make_session_store()
        history_messages, compressed_summary = await session_store.aload(
            session_id=session_id,
            user_id=user_id,
            channel=channel,
        )

        middleware = LightClawMiddleware(
            working_dir=WORKING_DIR,
            language=language,
            memory_manager=self._memory_manager,
            fact_injector=(self._memory_manager.get_fact_injector() if self._memory_manager else None),
        )
        middleware.compressed_summary = compressed_summary

        command_handler = LangGraphCommandHandler(
            history_messages=history_messages,
            middleware=middleware,
            session_store=session_store,
            session_id=session_id or "",
            user_id=user_id,
            channel=channel,
            language=language,
            memory_manager=self._memory_manager,
            turn_model=model,
            chat_manager=self._chat_manager,
        )

        chat = await self._get_or_create_chat(
            msgs=msgs,
            session_id=session_id,
            user_id=user_id,
            peer_user_id=peer_user_id,
            canonical_user_id=canonical_user_id,
            channel=channel,
        )

        logger.info(
            "Command context ready: turn_id=%s session_id=%s history=%d summary_chars=%d chat_id=%s duration=%.2fs",
            getattr(request, "turn_id", "") or "-",
            session_id or "-",
            len(history_messages),
            len(compressed_summary or ""),
            getattr(chat, "id", "-") if chat is not None else "-",
            time.monotonic() - started_at,
        )

        return SessionContext(
            session_id=session_id,
            user_id=user_id,
            peer_user_id=peer_user_id,
            canonical_user_id=canonical_user_id,
            channel=channel,
            env_context="",
            chat=chat,
            history_messages=history_messages,
            middleware=middleware,
            session_store=session_store,
            compressed_summary=compressed_summary,
            command_handler=command_handler,
        )

    async def create_langgraph_context(
        self,
        *,
        msgs,
        request: Any,
    ) -> SessionContext:
        """Build runtime context for a normal agent run.

        The compiled graph and shared middleware are initialized by
        ``AgentRunner`` at startup.  This method only loads session history
        and prepares per-turn runtime context.
        """
        from lightclaw.agent.core.engines.langgraph.middleware.steps.compactor import MemoryCompactor
        from lightclaw.agent.core.engines.langgraph.runtime_context import LightClawRuntimeContext
        from lightclaw.agent.memory.compaction_config import load_compaction_config
        from lightclaw.config import load_config

        session_id = request.session_id
        user_id = request.user_id
        peer_user_id = getattr(request, "peer_user_id", "") or ""
        channel = request.channel or DEFAULT_CHANNEL
        canonical_user_id = getattr(request, "canonical_user_id", "") or build_canonical_user_id(channel, user_id or "")
        started_at = time.monotonic()

        # Use injected config snapshot; fall back to load_config() only when
        # SessionFactory was constructed without one (legacy / test path).
        app_config = self._config if self._config is not None else load_config()
        max_input_length = app_config.agents.running.max_input_length

        # Step 1: load session history and compressed summary.
        step_started_at = time.monotonic()
        session_store = self._make_session_store()
        history_messages, compressed_summary = await session_store.aload(
            session_id=session_id,
            user_id=user_id,
            channel=channel,
        )
        logger.info(
            "SessionFactory step 1/3 history loaded: turn_id=%s session_id=%s history=%d summary_chars=%d duration=%.2fs",
            getattr(request, "turn_id", "") or "-",
            session_id or "-",
            len(history_messages),
            len(compressed_summary or ""),
            time.monotonic() - step_started_at,
        )

        # Step 2: build runtime context injected into LangGraph runtime.context.
        step_started_at = time.monotonic()
        enable_mm = is_memory_manager_enabled()
        turn_compactor = MemoryCompactor(
            memory_manager=(self._memory_manager if enable_mm else None),
            memory=None,
            compact_threshold=int(max_input_length * MEMORY_COMPACT_RATIO),
            keep_recent=MEMORY_COMPACT_KEEP_RECENT,
            channel=channel,
            session_id=session_id or "",
            compaction_config=load_compaction_config(app_config),
            compressed_summary=compressed_summary,
            language=app_config.agents.language,
        )

        # Route model per-turn so modality-specific models (e.g. vision) are
        # selected when the request contains images or other non-text content.
        # Uses the process-level model instance cache in create_chat_model so
        # the same (provider, model, base_url, api_key) tuple is never
        # re-instantiated.
        #
        # MiniMax vision sticky path: when the active provider is MiniMax and
        # the turn carries images, keep the active model pinned, rewrite image
        # blocks into path references, and hint the model to call mmx-cli.
        turn_model: Any | None = None
        try:
            from lightclaw.app.services.providers_service import load_providers_json as _load_providers

            providers_data = _load_providers()
            turn_model = self._resolve_vision_sticky_model(
                msgs=msgs,
                providers_data=providers_data,
                language=app_config.agents.language,
            )
            if turn_model is None:
                turn_model = LangGraphModelRouter().select(msgs, providers_data)
        except Exception:
            logger.exception("Per-turn model routing failed; shared graph model will be used")

        runtime_context = LightClawRuntimeContext(
            session_id=session_id or "",
            turn_id=getattr(request, "turn_id", "") or "",
            channel=channel,
            compressed_summary=compressed_summary,
            compactor=turn_compactor,
            turn_model=turn_model,
        )
        # Extract ActorContext from TurnRequest.context for coordinator recall/authz
        request_ctx = getattr(request, "context", None) or {}
        actor_ctx_dict = request_ctx.get("actor_context") if isinstance(request_ctx, dict) else None
        if actor_ctx_dict:
            from lightclaw.agent.core.actor_context import actor_context_from_dict
            runtime_context.actor_context = actor_context_from_dict(actor_ctx_dict)
        # Extract tool policy + runtime budget overrides (P1b heartbeat hard limits)
        if isinstance(request_ctx, dict):
            if "allowed_tools" in request_ctx:
                runtime_context.allowed_tools = request_ctx["allowed_tools"]
            if "tool_policy_source" in request_ctx:
                runtime_context.tool_policy_source = request_ctx["tool_policy_source"]
            if "runtime_profile" in request_ctx:
                runtime_context.runtime_profile = request_ctx["runtime_profile"]
            if "max_input_length_override" in request_ctx:
                runtime_context.max_input_length_override = request_ctx["max_input_length_override"]
            if "max_iters_override" in request_ctx:
                runtime_context.max_iters_override = request_ctx["max_iters_override"]
        logger.info(
            "SessionFactory step 2/3 runtime context ready: turn_id=%s session_id=%s duration=%.2fs",
            runtime_context.turn_id or "-",
            session_id or "-",
            time.monotonic() - step_started_at,
        )

        # Step 3: attach chat metadata for UI (if visible session).
        chat = None
        if session_id and not session_id.startswith(PROACTIVE_SESSION_PREFIX):
            chat = await self._get_or_create_chat(
                msgs=msgs,
                session_id=session_id,
                user_id=user_id,
                peer_user_id=peer_user_id,
                canonical_user_id=canonical_user_id,
                channel=channel,
            )
        elif session_id and session_id.startswith(PROACTIVE_SESSION_PREFIX):
            logger.debug("Skipping chat creation for proactive session: %s", session_id)

        middleware = self._middleware
        engine_session = self._engine_session
        if middleware is None or engine_session is None:
            from lightclaw.agent.core import AgentConfig
            from lightclaw.agent.core.engines.langgraph import (
                LangGraphAgentBuilder,
                LangGraphEngineSession,
                LightClawMiddleware,
                LightClawRuntimeContext,
            )
            from lightclaw.agent.core.engines.langgraph.tool_assembler import ToolAssembler
            from lightclaw.app.runner.session.prompt_assembler import assemble_system_prompt
            from lightclaw.app.security.hooks import tool_guard_check, tool_guard_wait
            from lightclaw.app.services.providers_service import load_providers_json as _load_providers

            providers_data = _load_providers()
            model = LangGraphModelRouter().select(msgs, providers_data)
            tools = await ToolAssembler(
                memory_manager=self._memory_manager,
                memory_dir=MEMORY_DIR,
            ).assemble(model=model)
            middleware = LightClawMiddleware(
                working_dir=WORKING_DIR,
                language=app_config.agents.language,
                memory_manager=(self._memory_manager if enable_mm else None),
                fact_injector=(
                    self._memory_manager.get_fact_injector() if enable_mm and self._memory_manager else None
                ),
                memory_compact_threshold=int(max_input_length * MEMORY_COMPACT_RATIO),
                max_input_length=max_input_length,
                keep_recent=MEMORY_COMPACT_KEEP_RECENT,
                compaction_config=load_compaction_config(app_config),
            )
            graph_config = AgentConfig(
                model=model,
                tools=tools,
                system_prompt=assemble_system_prompt(
                    env_context="",
                    language=app_config.agents.language,
                    channel=channel,
                    request=request,
                ),
                max_iterations=app_config.agents.running.max_iters,
                middleware=middleware,
                context_schema=LightClawRuntimeContext,
            )
            graph = LangGraphAgentBuilder().build(graph_config)
            engine_session = LangGraphEngineSession(
                graph=graph,
                max_iterations=app_config.agents.running.max_iters,
                tool_guard_check=tool_guard_check,
                tool_guard_wait=tool_guard_wait,
                memory_manager=self._memory_manager if enable_mm else None,
                turn_model=model,
            )
            logger.warning(
                "Shared LangGraph runtime missing; built per-request fallback graph for session_id=%s",
                session_id or "-",
            )

        logger.info(
            "SessionFactory step 3/3 context ready: turn_id=%s session_id=%s history=%d chat_id=%s duration=%.2fs",
            runtime_context.turn_id or "-",
            session_id or "-",
            len(history_messages),
            getattr(chat, "id", "-") if chat is not None else "-",
            time.monotonic() - started_at,
        )

        return SessionContext(
            session_id=session_id,
            user_id=user_id,
            peer_user_id=peer_user_id,
            canonical_user_id=canonical_user_id,
            channel=channel,
            env_context="",
            chat=chat,
            history_messages=history_messages,
            middleware=middleware,
            session_store=session_store,
            compressed_summary=compressed_summary,
            runtime_context=runtime_context,
            engine_session=engine_session,
            command_handler=None,
        )

    async def save_langgraph_session_state(
        self,
        context: SessionContext,
        final_messages: list | None = None,
    ) -> None:
        """Persist LangGraph session state after a run completes."""
        if context.session_store is None or context.session_id is None:
            return

        if self._chat_manager is not None and context.chat is not None and getattr(context.chat, "id", None):
            existing_chat = await self._chat_manager.get_chat(context.chat.id)
            if existing_chat is None:
                logger.debug(
                    "Skipping LangGraph session save for deleted chat %s",
                    context.chat.id,
                )
                return

        messages_to_save = final_messages or context.history_messages
        runtime_summary = getattr(getattr(context, "runtime_context", None), "compressed_summary", "")
        compressed_summary = runtime_summary or (
            context.middleware.compressed_summary if context.middleware else context.compressed_summary
        )

        if not messages_to_save and not compressed_summary:
            logger.debug(
                "Skipping LangGraph session save for %s: no messages or summary",
                context.session_id,
            )
            return

        channel = getattr(context, "channel", None)
        canonical_user_id = getattr(context, "canonical_user_id", "") or ""
        peer_user_id = getattr(context, "peer_user_id", "") or ""
        await context.session_store.asave(
            session_id=context.session_id,
            user_id=context.user_id,
            messages=messages_to_save,
            compressed_summary=compressed_summary,
            channel=channel,
            canonical_user_id=canonical_user_id,
            peer_user_id=peer_user_id,
        )
        logger.debug(
            "LangGraph session persisted: session_id=%s messages=%d summary_chars=%d",
            context.session_id,
            len(messages_to_save),
            len(compressed_summary or ""),
        )

    async def update_chat(self, chat: Any | None) -> None:
        """Persist chat metadata if a chat object exists."""
        if self._chat_manager is None or chat is None:
            return
        chat_id = getattr(chat, "id", None)
        if chat_id:
            existing_chat = await self._chat_manager.get_chat(chat_id)
            if existing_chat is None:
                logger.debug("Skipping chat metadata update for deleted chat %s", chat_id)
                return
        await self._chat_manager.update_chat(chat)
        logger.debug("Chat persisted: chat_id=%s", getattr(chat, "id", "-"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_session_store(self):
        from lightclaw.agent.core.engines.langgraph.session_store import LangGraphSessionStore

        return LangGraphSessionStore(save_dir=self._session.save_dir)

    async def _rollover_daily_talk(self, model: Any, language: str) -> None:
        """Trigger DAILY_TALK.md date rollover when the day has changed.

        Prompt injection is handled by ``AutoRecallInjector`` in the middleware
        pipeline (recall.py) to avoid double-injecting the same content.
        Gated by ``DAILY_TALK_ENABLED``; never called when the switch is off.
        """
        if is_bootstrap_pending(WORKING_DIR):
            logger.info("Skipping DAILY_TALK rollover while bootstrap is pending")
            return

        from lightclaw.app.runner.session.daily_talk_store import DailyTalkStore

        store = DailyTalkStore(working_dir=WORKING_DIR)
        try:
            await store.maybe_rollover(model=model, language=language)
        except Exception:
            logger.exception("DAILY_TALK rollover failed")

    def _get_chat_name(self, msgs) -> str:
        name = "New Chat"
        if msgs:
            content = msgs[0].get_text_content()
            name = content[:10] if content else "Media Message"
        return name

    def _resolve_vision_sticky_model(
        self,
        *,
        msgs: Any,
        providers_data: Any,
        language: str,
    ) -> Any | None:
        """Return the active chat model when MiniMax sticky path applies, else None.

        When active provider == minimax AND latest user message carries images:
        inject a mmx-cli hint, rewrite image blocks into path references, and
        resolve the user's active model (bypassing modality-based routing).
        Any failure returns None so the caller falls back to normal routing.
        """
        from lightclaw.agent.core.adapters.vision_preprocessor import (
            inject_image_tool_hint,
            rewrite_image_blocks_for_tool_first,
            should_sticky_for_active_provider,
        )

        if not should_sticky_for_active_provider(msgs=msgs, providers_data=providers_data):
            return None

        from lightclaw.agent.core.engines.langgraph.model_factory import create_chat_model
        from lightclaw.app.services.providers_service import list_resolved_llm_configs

        active_provider_id, active_model_id = providers_data.parse_active_llm()
        inject_image_tool_hint(msgs=msgs, language=language)
        rewrite_result = rewrite_image_blocks_for_tool_first(msgs=msgs, language=language)
        if not rewrite_result.fully_text:
            logger.warning(
                "vision-sticky: image rewrite incomplete (rewritten=%d remaining=%d); fallback to routing",
                rewrite_result.rewritten_image_blocks,
                rewrite_result.remaining_image_blocks,
            )
            return None
        resolved = next(
            (
                candidate
                for candidate in list_resolved_llm_configs(providers_data)
                if candidate.provider_id == active_provider_id and candidate.model == active_model_id
            ),
            None,
        )
        if resolved is None:
            logger.warning(
                "vision-sticky: active model %s/%s not resolvable; fallback to routing",
                active_provider_id or "-",
                active_model_id or "-",
            )
            return None
        logger.info(
            "vision-sticky: pinning MiniMax active model %s/%s for image turn",
            active_provider_id, active_model_id,
        )
        return create_chat_model(resolved)

    async def _get_or_create_chat(
        self,
        *,
        msgs,
        session_id: str | None,
        user_id: str | None,
        peer_user_id: str | None,
        canonical_user_id: str | None,
        channel: str,
    ) -> Any | None:
        if self._chat_manager is None:
            return None
        return await self._chat_manager.get_or_create_chat(
            session_id,
            user_id,
            channel,
            peer_user_id=peer_user_id,
            canonical_user_id=canonical_user_id,
            name=self._get_chat_name(msgs),
        )
