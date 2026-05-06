"""Factory for creating LangChain-compatible chat models.

Bridges LightClaw's existing provider configuration (providers.json) to
LangChain's ``BaseChatModel`` hierarchy so the rest of the LangGraph
pipeline can stay provider-agnostic.

Supports multiple model providers (OpenAI, Anthropic, Google, Ollama,
Azure, and any OpenAI-compatible endpoint) via the ``chat_model`` field
in each provider definition.

Example:
    >>> from lightclaw.agent.core.model_factory import create_chat_model
    >>> model = create_chat_model()          # uses active provider
    >>> model = create_chat_model(llm_cfg)   # explicit config
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Type alias for the model instance cache key.
ModelCacheKey = tuple[str, str, str, str, str, str, tuple[tuple[str, str], ...]]

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from lightclaw.infra.providers.models import ResolvedModelConfig

logger = logging.getLogger(__name__)

# Bounded LRU caches — both use OrderedDict for O(1) move_to_end eviction.
# Sizes are intentionally small: in normal usage only 1-2 distinct models and
# a handful of provider/base_url pairs are ever active in a single process.
# Override via env vars if you run many providers simultaneously.
_MODEL_CACHE_MAX_SIZE = int(os.environ.get("LIGHTCLAW_MODEL_CACHE_MAX_SIZE", "8"))
_STREAM_USAGE_CACHE_MAX_SIZE = int(os.environ.get("LIGHTCLAW_STREAM_USAGE_CACHE_MAX_SIZE", "32"))

# Model runtime guard: request-level timeout and retry settings.
# Applied to all OpenAI-compatible (ChatOpenAI) model instances.
# Overridable via env vars.
def _parse_model_timeout() -> float | None:
    v = os.environ.get("LIGHTCLAW_MODEL_REQUEST_TIMEOUT", "120").strip()
    try:
        val = float(v)
        return val if val > 0 else 120.0
    except (ValueError, TypeError):
        logger.warning("LIGHTCLAW_MODEL_REQUEST_TIMEOUT=%r invalid, using 120", v)
        return 120.0


def _parse_model_max_retries() -> int:
    v = os.environ.get("LIGHTCLAW_MODEL_MAX_RETRIES", "2").strip()
    try:
        val = int(v)
        return max(0, min(val, 5))
    except (ValueError, TypeError):
        logger.warning("LIGHTCLAW_MODEL_MAX_RETRIES=%r invalid, using 2", v)
        return 2


_MODEL_REQUEST_TIMEOUT = _parse_model_timeout()
_MODEL_MAX_RETRIES = _parse_model_max_retries()

_STREAM_USAGE_CAPABILITY_CACHE: OrderedDict[tuple[str, str], bool] = OrderedDict()

# Process-level cache: (provider_id, model, base_url, api_key, api, chat_model, headers) → BaseChatModel instance.
# BaseChatModel instances are stateless across calls (stream_usage mutations are idempotent),
# so sharing one instance across concurrent turns is safe.
_MODEL_INSTANCE_CACHE: OrderedDict[ModelCacheKey, Any] = OrderedDict()

_OPENAI_COMPAT_APIS = {
    "openai-completions",
    "openai-responses",
    "openai-codex-responses",
}


def _stream_usage_cache_key(provider_id: str, base_url: str) -> tuple[str, str]:
    """Return a normalized cache key for stream-usage capability checks."""
    return ((provider_id or "").strip().lower(), (base_url or "").strip().lower())


def _supports_stream_usage(provider_id: str, base_url: str) -> bool:
    """Return whether the provider should receive OpenAI ``stream_options``."""
    key = _stream_usage_cache_key(provider_id, base_url)
    val = _STREAM_USAGE_CAPABILITY_CACHE.get(key)
    if val is not None:
        _STREAM_USAGE_CAPABILITY_CACHE.move_to_end(key)
    return val if val is not None else True


def _mark_stream_usage_unsupported(provider_id: str, base_url: str) -> None:
    """Remember that a provider/base-url pair rejects streaming usage metadata."""
    key = _stream_usage_cache_key(provider_id, base_url)
    _STREAM_USAGE_CAPABILITY_CACHE[key] = False
    _STREAM_USAGE_CAPABILITY_CACHE.move_to_end(key)
    if len(_STREAM_USAGE_CAPABILITY_CACHE) > _STREAM_USAGE_CACHE_MAX_SIZE:
        _STREAM_USAGE_CAPABILITY_CACHE.popitem(last=False)


def _is_stream_usage_unsupported_error(exc: Exception) -> bool:
    """Return True when an API error indicates ``stream_options`` is unsupported."""
    message = str(exc).lower()
    mentions_stream_usage = "stream_options" in message or "include_usage" in message
    unsupported_hints = (
        "unsupported",
        "not supported",
        "invalid",
        "unexpected",
        "unknown parameter",
        "unrecognized",
        "additional properties",
        "extra inputs",
        "unknown field",
    )
    return mentions_stream_usage and any(hint in message for hint in unsupported_hints)


def _resolve_deepseek_thinking_extra_body(
    provider_id: str,
    model_name: str,
) -> dict[str, object] | None:
    """Build ``extra_body`` for DeepSeek thinking mode when applicable.

    Resolution order:
    1. Provider-level ``thinking`` override from ``ProviderEntry``
       (set by the user via Dashboard toggle).
    2. Per-model ``thinking`` field from ``model_templates.json``.
    3. No injection when neither source provides a value.

    Only applies to the ``deepseek`` provider.  Other providers are
    unaffected even if they happen to host DeepSeek-compatible models.
    """
    if provider_id != "deepseek":
        return None

    # Step 1: Check provider-level override (user toggle in Dashboard).
    try:
        from lightclaw.infra.providers.models import ProvidersData
        from lightclaw.infra.providers.store import load_providers_json
    except ImportError:
        logger.debug("Provider store or models not available; skipping provider-level thinking override")
    else:
        try:
            data = load_providers_json()
            # Guard against LegacyProvidersData whose entries lack 'thinking'.
            if isinstance(data, ProvidersData):
                entry = data.providers.get(provider_id)
                if entry is not None and entry.thinking is not None:
                    thinking_type = "enabled" if entry.thinking else "disabled"
                    return {"thinking": {"type": thinking_type}}
        except Exception:
            logger.debug(
                "Failed to read provider-level thinking override for %s; falling back to template",
                provider_id,
                exc_info=True,
            )

    # Step 2: Fall back to per-model template.
    try:
        from lightclaw.infra.providers.registry import get_model_template
    except ImportError:
        return None

    tmpl = get_model_template(provider_id, model_name)
    if not tmpl or "thinking" not in tmpl:
        return None

    thinking_enabled = bool(tmpl["thinking"])
    thinking_type = "enabled" if thinking_enabled else "disabled"
    return {"thinking": {"type": thinking_type}}


def _make_openai_with_reasoning(
    base_class: type,
    *,
    provider_id: str,
    provider_base_url: str,
    **kwargs,
) -> object:
    """Return a ChatOpenAI instance that forwards ``reasoning_content`` deltas.

    LangChain's ``ChatOpenAI._convert_chunk_to_generation_chunk`` only reads
    ``delta.content`` and ignores the non-standard ``delta.reasoning_content``
    field used by DeepSeek-R1, GLM-5, Qwen-QwQ and other reasoning models.
    This wrapper extracts it and injects it into ``additional_kwargs`` so the
    rest of the pipeline (``event_bridge/bridge.py``) can pick it up.
    """
    from langchain_core.outputs import ChatGenerationChunk

    class _ChatOpenAIWithReasoning(base_class):  # type: ignore[valid-type]
        def _get_request_payload(
            self,
            input_: Any,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict:
            """Inject ``reasoning_content`` into outbound assistant message dicts.

            LangChain's ``BaseChatOpenAI._get_request_payload`` converts
            LangChain messages to OpenAI-format dicts via a **module-level**
            ``_convert_message_to_dict`` function.  That function ignores
            non-standard fields stored in ``additional_kwargs``.

            DeepSeek requires that ``reasoning_content`` be echoed back in
            subsequent requests when thinking mode is enabled, otherwise
            the API returns HTTP 400.  This override post-processes the
            payload to:

            1. Inject ``reasoning_content`` from ``additional_kwargs`` for
               assistant messages that carry it.
            2. When thinking mode is enabled (detected via ``extra_body``),
               inject an empty ``reasoning_content`` for assistant messages
               that lack it.  This prevents 400 errors when historical
               messages were produced with thinking disabled and the user
               later re-enables thinking mode.
            """
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            messages = payload.get("messages")
            if not isinstance(messages, list):
                return payload

            # Detect whether thinking mode is currently enabled *for DeepSeek*
            # so we can fill in missing reasoning_content for historical
            # messages that were produced without thinking.  The guard on
            # ``provider_id`` (captured via closure) ensures this never
            # affects non-DeepSeek providers even if they happen to share
            # the same ``_ChatOpenAIWithReasoning`` wrapper class.
            thinking_enabled = False
            if provider_id == "deepseek":
                extra_body = payload.get("extra_body") or getattr(self, "extra_body", None)
                if isinstance(extra_body, dict):
                    thinking_cfg = extra_body.get("thinking")
                    if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled":
                        thinking_enabled = True

            # Resolve original LangChain messages to match positionally
            # with the already-converted dicts.  ``_convert_input`` is
            # idempotent and lightweight (it merely wraps the input into
            # a PromptValue if necessary).
            try:
                original_messages = self._convert_input(input_).to_messages()
            except Exception:
                return payload

            if len(original_messages) != len(messages):
                return payload

            for lc_msg, msg_dict in zip(original_messages, messages, strict=False):
                if not isinstance(msg_dict, dict) or msg_dict.get("role") != "assistant":
                    continue
                additional_kwargs = getattr(lc_msg, "additional_kwargs", None)
                if not isinstance(additional_kwargs, dict):
                    additional_kwargs = {}
                # Support both "reasoning_content" and "thinking" keys
                reasoning = additional_kwargs.get("reasoning_content") or additional_kwargs.get("thinking")
                if isinstance(reasoning, str) and reasoning.strip():
                    msg_dict["reasoning_content"] = reasoning
                elif thinking_enabled:
                    # When thinking mode is enabled, DeepSeek requires
                    # reasoning_content on all assistant messages (especially
                    # those with tool_calls).  Historical messages produced
                    # while thinking was disabled won't have it, so inject
                    # an empty string to satisfy the API constraint.
                    msg_dict["reasoning_content"] = ""

            return payload

        def _convert_chunk_to_generation_chunk(
            self,
            chunk: dict,
            default_chunk_class: type,
            base_generation_info: dict | None,
        ) -> ChatGenerationChunk | None:
            result = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
            if result is None:
                return None
            choices = chunk.get("choices") or []
            if not choices:
                return result
            reasoning = (choices[0].get("delta") or {}).get("reasoning_content")
            if reasoning:
                result.message.additional_kwargs["reasoning_content"] = reasoning
            return result

        def _disable_stream_usage_and_should_retry(self, exc: Exception) -> bool:
            if not _is_stream_usage_unsupported_error(exc):
                return False
            _mark_stream_usage_unsupported(provider_id, provider_base_url)
            self.stream_usage = False
            logger.warning(
                "Provider rejected stream_options.include_usage; retrying without stream_usage: provider=%s base_url=%s error=%s",
                provider_id or "(empty)",
                _format_base_url_for_log(provider_base_url),
                exc,
            )
            return True

        def _stream(
            self,
            messages,
            stop=None,
            run_manager=None,
            *,
            stream_usage=None,
            **kwargs,
        ):
            yielded_any = False
            try:
                for chunk in super()._stream(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    stream_usage=stream_usage,
                    **kwargs,
                ):
                    yielded_any = True
                    yield chunk
            except Exception as exc:
                if yielded_any or not self._disable_stream_usage_and_should_retry(exc):
                    raise
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("stream_options", None)
                for chunk in super()._stream(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    stream_usage=False,
                    **retry_kwargs,
                ):
                    yield chunk

        async def _astream(
            self,
            messages,
            stop=None,
            run_manager=None,
            *,
            stream_usage=None,
            **kwargs,
        ):
            yielded_any = False
            try:
                async for chunk in super()._astream(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    stream_usage=stream_usage,
                    **kwargs,
                ):
                    yielded_any = True
                    yield chunk
            except Exception as exc:
                if yielded_any or not self._disable_stream_usage_and_should_retry(exc):
                    raise
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("stream_options", None)
                async for chunk in super()._astream(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    stream_usage=False,
                    **retry_kwargs,
                ):
                    yield chunk

    return _ChatOpenAIWithReasoning(**kwargs)


def invalidate_model_cache() -> None:
    """Clear the process-level model instance cache.

    Called by ``store.update_provider_settings`` when a provider config
    change (e.g. thinking toggle) requires cached model instances to be
    rebuilt on the next request.
    """
    _MODEL_INSTANCE_CACHE.clear()


def _resolve_chat_model_name(provider_id: str) -> str:
    """Look up the ``chat_model`` class name for *provider_id*.

    Returns ``"ChatOpenAI"`` when the provider is unknown or has no
    explicit ``chat_model`` configured.
    """
    if not provider_id:
        return "ChatOpenAI"

    try:
        from lightclaw.infra.providers.registry import get_provider_chat_model

        name = get_provider_chat_model(provider_id)
        # Map legacy provider names to LangChain equivalents
        if name == "OpenAIChatModel":
            return "ChatOpenAI"
        return name
    except Exception:
        return "ChatOpenAI"


def _format_base_url_for_log(base_url: str) -> str:
    """Return a stable base-url label for logs."""
    return base_url or "(empty)"


def _log_model_init(
    *,
    chat_model_name: str,
    model_name: str,
    provider_id: str,
    base_url: str,
    streaming: bool,
    stream_usage: bool | None,
    api_key: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a single normalized model-initialization log line."""
    payload = {
        "chat_model": chat_model_name,
        "model": model_name,
        "provider": provider_id or "(empty)",
        "base_url": _format_base_url_for_log(base_url),
        "streaming": streaming,
        "stream_usage": stream_usage if stream_usage is not None else "n/a",
        "api_key_configured": bool(api_key),
    }
    if extra:
        payload.update(extra)
    fields = " ".join(f"{key}={value}" for key, value in payload.items())
    logger.info("LLM init: %s", fields)


def _safe_instantiate(
    factory: type | Callable[..., Any],
    kwargs: dict[str, Any],
) -> Any:
    """Instantiate a model with *kwargs*; retry without ``default_headers`` if unsupported."""
    try:
        return factory(**kwargs)
    except TypeError as exc:
        if "default_headers" not in kwargs or "default_headers" not in str(exc):
            raise
        kwargs.pop("default_headers", None)
        return factory(**kwargs)


def _normalized_header_items(headers: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not headers:
        return ()
    normalized = [((k or "").strip().lower(), (v or "").strip()) for k, v in headers.items() if (k or "").strip()]
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _model_cache_key(
    llm_cfg: ResolvedModelConfig,
) -> ModelCacheKey:
    return (
        llm_cfg.provider_id or "",
        llm_cfg.model or "",
        llm_cfg.base_url or "",
        llm_cfg.api_key or "",
        llm_cfg.api or "",
        llm_cfg.chat_model or "",
        _normalized_header_items(llm_cfg.default_headers),
    )


def create_chat_model(
    llm_cfg: ResolvedModelConfig | None = None,
) -> BaseChatModel:
    """Create (or return a cached) LangChain ``BaseChatModel`` from LightClaw provider config.

    Resolution order:
    1. If *llm_cfg* is provided, use it directly.
    2. Otherwise call ``get_active_llm_config()`` to read providers.json.

    Instances are cached by (provider_id, model, base_url, api_key) so that
    switching model per turn costs a dict lookup rather than a full object
    construction.  BaseChatModel instances are call-stateless; the only
    mutable field is ``stream_usage`` which can only flip from True → False
    and is idempotent, so sharing across concurrent turns is safe.

    Args:
        llm_cfg: Resolved model configuration.  When ``None``, the
            active configuration is loaded automatically.

    Returns:
        A ready-to-use LangChain chat model.

    Raises:
        ValueError: If no LLM is configured.
    """
    if llm_cfg is None:
        from lightclaw.infra.providers import get_active_llm_config

        llm_cfg = get_active_llm_config()

    if llm_cfg is None:
        raise ValueError("No LLM configured. Please add a provider and select a model first.")

    key = _model_cache_key(llm_cfg)
    cached = _MODEL_INSTANCE_CACHE.get(key)
    if cached is not None:
        _MODEL_INSTANCE_CACHE.move_to_end(key)
        return cached

    model = _create_remote_model(llm_cfg)
    _MODEL_INSTANCE_CACHE[key] = model
    if len(_MODEL_INSTANCE_CACHE) > _MODEL_CACHE_MAX_SIZE:
        _MODEL_INSTANCE_CACHE.popitem(last=False)
    return model


def _create_remote_model(
    llm_cfg: ResolvedModelConfig | None,
) -> BaseChatModel:
    """Create a remote chat model, routing to the correct LangChain class."""
    if not llm_cfg or not (llm_cfg.model or llm_cfg.base_url or llm_cfg.provider_id):
        raise ValueError("No LLM configured. Please add a provider and select a model first.")

    model_name = llm_cfg.model
    if not model_name:
        raise ValueError("Active LLM has no model name set. Please select a model in provider settings.")
    api_key = llm_cfg.api_key
    base_url = llm_cfg.base_url
    provider_id = llm_cfg.provider_id
    api = (llm_cfg.api or "").strip().lower()
    default_headers = llm_cfg.default_headers

    # Resolve transport class by API protocol first, then fall back to provider chat_model.
    if api and api in _OPENAI_COMPAT_APIS:
        chat_model_name = "ChatOpenAI"
    elif api == "anthropic-messages":
        chat_model_name = "ChatAnthropic"
    elif llm_cfg.chat_model:
        chat_model_name = llm_cfg.chat_model
    else:
        chat_model_name = _resolve_chat_model_name(provider_id)

    from lightclaw.infra.providers.registry import get_langchain_model_class

    model_class = get_langchain_model_class(chat_model_name)

    return _instantiate_model(
        model_class=model_class,
        chat_model_name=chat_model_name,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        provider_id=provider_id,
        default_headers=default_headers,
    )


def _instantiate_model(
    *,
    model_class: type,
    chat_model_name: str,
    model_name: str,
    api_key: str,
    base_url: str,
    provider_id: str = "",
    default_headers: dict[str, str] | None = None,
) -> BaseChatModel:
    """Instantiate a LangChain ChatModel with provider-specific parameters.

    Different providers have slightly different constructor signatures;
    this function handles the variance.
    """
    if chat_model_name == "ChatAnthropic":
        kwargs: dict = {
            "model": model_name,
            "api_key": api_key,
            "streaming": True,
            "stream_usage": True,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers
        model = _safe_instantiate(model_class, kwargs)
        _log_model_init(
            chat_model_name=chat_model_name,
            model_name=model_name,
            provider_id=provider_id,
            base_url=base_url,
            streaming=True,
            stream_usage=True,
            api_key=api_key,
            extra={"default_headers": len(default_headers or {})},
        )
        return model

    if chat_model_name == "ChatGoogleGenerativeAI":
        # Current langchain-google-genai does not accept `stream_usage`.
        model = model_class(
            model=model_name,
            google_api_key=api_key,
            streaming=True,
        )
        _log_model_init(
            chat_model_name=chat_model_name,
            model_name=model_name,
            provider_id=provider_id,
            base_url=base_url,
            streaming=True,
            stream_usage=None,
            api_key=api_key,
        )
        return model

    if chat_model_name == "ChatOllama":
        kwargs: dict = {
            "model": model_name,
        }
        if base_url:
            kwargs["base_url"] = base_url
        model = model_class(**kwargs)
        _log_model_init(
            chat_model_name=chat_model_name,
            model_name=model_name,
            provider_id=provider_id,
            base_url=base_url,
            streaming=False,
            stream_usage=None,
            api_key=api_key,
        )
        return model

    if chat_model_name == "AzureChatOpenAI":
        kwargs: dict = {
            "model": model_name,
            "api_key": api_key or "lightclaw-local",
            "azure_endpoint": base_url,
            "streaming": True,
            "stream_usage": True,
        }
        if default_headers:
            kwargs["default_headers"] = default_headers
        model = _safe_instantiate(model_class, kwargs)
        _log_model_init(
            chat_model_name=chat_model_name,
            model_name=model_name,
            provider_id=provider_id,
            base_url=base_url,
            streaming=True,
            stream_usage=True,
            api_key=api_key,
            extra={"default_headers": len(default_headers or {})},
        )
        return model

    # Default: ChatOpenAI and any OpenAI-compatible API.
    # Enable stream_usage by default, then dynamically disable it for a
    # provider/base-url pair if the first request proves it unsupported.
    use_stream_usage = _supports_stream_usage(provider_id, base_url)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key or "lightclaw-local",
        "base_url": base_url,
        "streaming": True,
        "stream_usage": use_stream_usage,
        "request_timeout": _MODEL_REQUEST_TIMEOUT,
        "max_retries": _MODEL_MAX_RETRIES,
    }
    if default_headers:
        kwargs["default_headers"] = default_headers

    # Inject DeepSeek thinking toggle via extra_body when configured.
    thinking_extra_body = _resolve_deepseek_thinking_extra_body(provider_id, model_name)
    if thinking_extra_body:
        kwargs["extra_body"] = thinking_extra_body

    model = _safe_instantiate(
        lambda **kw: _make_openai_with_reasoning(
            model_class,
            provider_id=provider_id,
            provider_base_url=base_url,
            **kw,
        ),
        kwargs,
    )
    _log_model_init(
        chat_model_name=chat_model_name,
        model_name=model_name,
        provider_id=provider_id,
        base_url=base_url,
        streaming=True,
        stream_usage=use_stream_usage,
        api_key=api_key,
        extra={
            "reasoning_chunk_passthrough": True,
            "default_headers": len(default_headers or {}),
            "thinking_extra_body": bool(thinking_extra_body),
            "request_timeout": _MODEL_REQUEST_TIMEOUT,
            "max_retries": _MODEL_MAX_RETRIES,
        },
    )
    return model
