"""LLM-powered authentication page detector.

Uses a lightweight LLM call against the page's ARIA Snapshot (accessibility
tree text) to semantically determine whether the current page is a login /
authentication page, replacing the keyword-based heuristics that produced
frequent false positives.

Falls back to the legacy keyword detector when the LLM is unavailable or
times out.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from lightclaw.agent.prompt_catalog import get_auth_detector_system_prompt
from lightclaw.agent.tools.browser.screenshot_stream import AuthDetectionResult, AuthPageDetector
from lightclaw.agent.tools.browser.session import BrowserSession
from lightclaw.agent.utils.token_counting import strip_model_special_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_DETECT_TIMEOUT = 5.0  # seconds
CACHE_TTL = 30.0  # seconds
_MAX_SNAPSHOT_CHARS = 6000  # truncate ARIA snapshot to keep prompt small

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    result: AuthDetectionResult
    expires_at: float


_cache: dict[str, _CacheEntry] = {}


def _cache_key(url: str, snapshot_text: str) -> str:
    h = hashlib.md5((url + snapshot_text[:500]).encode(), usedforsecurity=False).hexdigest()
    return f"{url}:{h}"


def _get_cached(key: str) -> AuthDetectionResult | None:
    entry = _cache.get(key)
    if entry and entry.expires_at > time.monotonic():
        return entry.result
    _cache.pop(key, None)
    return None


def _put_cached(key: str, result: AuthDetectionResult) -> None:
    _cache[key] = _CacheEntry(result=result, expires_at=time.monotonic() + CACHE_TTL)
    # Evict expired entries periodically
    if len(_cache) > 50:
        now = time.monotonic()
        expired = [k for k, v in _cache.items() if v.expires_at <= now]
        for k in expired:
            _cache.pop(k, None)


# ---------------------------------------------------------------------------
# LLM auth detector
# ---------------------------------------------------------------------------


class LLMAuthDetector:
    """Detect auth pages by asking an LLM to analyse the ARIA snapshot.

    Usage::

        detector = LLMAuthDetector(session)
        result = await detector.detect()
    """

    def __init__(self, session: BrowserSession) -> None:
        self._session = session
        self._fallback = AuthPageDetector(session)

    async def detect(self) -> AuthDetectionResult:
        """Analyse the current page for authentication indicators.

        Tries the LLM first; falls back to keyword heuristics on failure
        or timeout.
        """
        session = self._session
        if not session.cdp or not session.cdp.context:
            return AuthDetectionResult()

        try:
            pages = session.cdp.context.pages
            if not pages:
                return AuthDetectionResult()
            page = pages[0]
            url = page.url
            session.current_url = url
        except Exception:
            return AuthDetectionResult()

        # Obtain ARIA snapshot
        try:
            snapshot_text: str = await asyncio.wait_for(
                page.locator(":root").aria_snapshot(),
                timeout=3.0,
            )
        except Exception as exc:
            logger.debug("ARIA snapshot failed, falling back to keywords: %s", exc)
            return await self._fallback.detect()

        if not snapshot_text:
            return await self._fallback.detect()

        # Truncate
        if len(snapshot_text) > _MAX_SNAPSHOT_CHARS:
            snapshot_text = snapshot_text[:_MAX_SNAPSHOT_CHARS] + "\n…(truncated)"

        # Check cache
        key = _cache_key(url, snapshot_text)
        cached = _get_cached(key)
        if cached is not None:
            return cached

        # Call LLM
        try:
            result = await asyncio.wait_for(
                self._call_llm(url, snapshot_text),
                timeout=LLM_DETECT_TIMEOUT,
            )
            _put_cached(key, result)
            return result
        except TimeoutError:
            logger.warning("LLM auth detection timed out — falling back to keywords")
        except Exception as exc:
            logger.warning("LLM auth detection failed — falling back to keywords: %s", exc)

        return await self._fallback.detect()

    async def _call_llm(self, url: str, snapshot_text: str) -> AuthDetectionResult:
        """Send the ARIA snapshot to the LLM and parse its response."""
        model = _get_model()
        if model is None:
            raise RuntimeError("No LLM model available")

        from langchain_core.messages import HumanMessage, SystemMessage

        user_content = f"Page URL: {url}\n\nARIA Snapshot:\n{snapshot_text}"
        response = await model.ainvoke(
            [
                SystemMessage(content=get_auth_detector_system_prompt()),
                HumanMessage(content=strip_model_special_tokens(user_content)),
            ],
        )

        return _parse_llm_response(response.content, url)


# ---------------------------------------------------------------------------
# Model lazy-init
# ---------------------------------------------------------------------------

_model_instance: Any = None

_STREAM_USAGE_UNSUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {
        "minimax",
        "zhipu",
        "moonshot",
        "hunyuan",
        "mimo",
    }
)


def _supports_stream_usage(provider_id: str, base_url: str) -> bool:
    """Return whether OpenAI-style stream usage is supported."""
    provider_id_lower = (provider_id or "").lower()
    base_url_lower = (base_url or "").lower()

    if provider_id_lower in _STREAM_USAGE_UNSUPPORTED_PROVIDERS:
        return False

    unsupported_domains = (
        "minimaxi.com",
        "bigmodel.cn",
        "moonshot.cn",
        "hunyuan.cloud.tencent.com",
        "xiaomimimo.com",
        "ark.cn-beijing.volces.com",
    )
    if any(domain in base_url_lower for domain in unsupported_domains):
        return False

    return not (not provider_id_lower and not base_url_lower)


def _resolve_chat_model_name(provider_id: str) -> str:
    """Resolve provider chat model class name with a safe fallback."""
    if not provider_id:
        return "ChatOpenAI"

    try:
        from lightclaw.infra.providers import get_provider_chat_model

        name = get_provider_chat_model(provider_id)
        if name == "OpenAIChatModel":
            return "ChatOpenAI"
        return name
    except Exception:
        return "ChatOpenAI"


def _create_chat_model() -> Any:
    """Create a chat model from provider config without langgraph dependency."""
    from lightclaw.infra.providers import get_active_llm_config, get_langchain_model_class

    llm_cfg = get_active_llm_config()
    if llm_cfg and (llm_cfg.model or llm_cfg.base_url or llm_cfg.provider_id):
        model_name = llm_cfg.model or "DeepSeek-V3-0324"
        api_key = llm_cfg.api_key
        base_url = llm_cfg.base_url
        provider_id = llm_cfg.provider_id
    else:
        model_name = llm_cfg.model if llm_cfg else "DeepSeek-V3-0324"
        api_key = llm_cfg.api_key if llm_cfg else ""
        base_url = llm_cfg.base_url if llm_cfg else ""
        provider_id = llm_cfg.provider_id if llm_cfg else ""

    chat_model_name = _resolve_chat_model_name(provider_id)
    model_class = get_langchain_model_class(chat_model_name)

    if chat_model_name == "ChatAnthropic":
        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "streaming": True,
            "stream_usage": True,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return model_class(**kwargs)

    if chat_model_name == "ChatGoogleGenerativeAI":
        return model_class(
            model=model_name,
            google_api_key=api_key,
            streaming=True,
        )

    if chat_model_name == "ChatOllama":
        kwargs = {"model": model_name}
        if base_url:
            kwargs["base_url"] = base_url
        return model_class(**kwargs)

    if chat_model_name == "AzureChatOpenAI":
        return model_class(
            model=model_name,
            api_key=api_key or "lightclaw-local",
            azure_endpoint=base_url,
            streaming=True,
            stream_usage=True,
        )

    return model_class(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        streaming=True,
        stream_usage=_supports_stream_usage(provider_id, base_url),
    )


def _get_model() -> Any:
    """Lazily create a lightweight chat model for auth detection."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    try:
        _model_instance = _create_chat_model()
        logger.info("LLMAuthDetector: created chat model for auth page detection")
        return _model_instance
    except Exception as exc:
        logger.warning("LLMAuthDetector: failed to create model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_llm_response(content: str, url: str) -> AuthDetectionResult:
    """Parse the LLM's JSON response into an AuthDetectionResult."""
    result = AuthDetectionResult(url=url)

    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("LLM returned non-JSON: %s", text[:200])
        return result

    result.is_auth_page = bool(data.get("is_auth_page", False))
    result.auth_type = str(data.get("auth_type", "not_auth"))
    result.confidence = float(data.get("confidence", 0.0))
    result.detail = str(data.get("reasoning", ""))

    return result
