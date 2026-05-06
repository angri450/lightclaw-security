"""Qdrant memory system initialization factory.

Aligns with openclaw MemoryIndexManager.get() factory method.

Called during agent startup to initialize:
1. Create embedding_fn (from EMBEDDING_* env vars)
2. Create and start QdrantMemoryIndexer (full sync + file watching)
3. Create QdrantMemorySearcher
4. Inject into MemoryManager (set_memory_backend)
5. Create ProactiveMemoryEngine (optional, starts when PROACTIVE_MEMORY_ENABLED=true)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightclaw.agent.memory.memory_manager import MemoryManager
    from lightclaw.agent.memory.proactive_engine import ProactiveMemoryEngine
    from lightclaw.agent.memory.qdrant_indexer import QdrantMemoryIndexer
    from lightclaw.agent.memory.qdrant_searcher import QdrantMemorySearcher


def _load_embedding_config() -> dict:
    """Load embedding config from file; return empty dict on failure."""
    try:
        from lightclaw.config.utils import load_config

        cfg = load_config()
        emb = cfg.embedding
        if emb is not None:
            if emb.provider == "custom":
                return {
                    "api_key": emb.api_key or "",
                    "base_url": emb.base_url or "",
                    "model_name": emb.model_name or "",
                    "dimensions": emb.dimensions or 768,
                }
            if emb.provider == "local":
                return {
                    "api_key": "local",
                    "base_url": "",  # no HTTP — handled by create_embedding_fn
                    "model_name": emb.local_model or "",
                    "dimensions": emb.dimensions or 768,
                }
    except Exception:
        pass
    return {}


logger = logging.getLogger(__name__)


def is_embedding_configured() -> bool:
    """Return True if the embedding provider is configured (base_url + model_name set).

    Used by the backend bootstrapper to decide whether to attempt Qdrant
    initialisation at all.  When False, starting a HybridMemoryBackend would
    succeed but create an empty AsyncQdrantClient that consumes 50-80 MB of
    mmap memory for no benefit, so we skip straight to FTS-only.
    """
    _emb_cfg = _load_embedding_config()
    base_url = os.environ.get("EMBEDDING_BASE_URL") or _emb_cfg.get("base_url", "")
    model_name = os.environ.get("EMBEDDING_MODEL_NAME") or _emb_cfg.get("model_name", "")
    # Local in-process provider only needs api_key="local"; no base_url required.
    api_key = os.environ.get("EMBEDDING_API_KEY") or _emb_cfg.get("api_key", "")
    if api_key == "local":
        return True
    return bool(base_url and model_name)


def create_embedding_fn(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    dimensions: int | None = None,
) -> Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]] | None:
    """Create async embedding function from environment variables.

    Returns None if no embedding provider is configured (degrades to FTS-only mode).

    When provider="local", bypasses HTTP entirely and calls
    LocalEmbeddingServer.embed() in-process for zero-latency inference.
    """
    _api_key = (api_key or os.environ.get("EMBEDDING_API_KEY", "")).strip()

    # Local provider: in-process inference, no HTTP needed
    if _api_key == "local":
        local_server = _get_local_server_if_available(_api_key)
        if local_server is not None:
            _model_name = model_name or os.environ.get("EMBEDDING_MODEL_NAME", "")
            _dims = dimensions or int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))
            return _create_local_embedding_fn(local_server, _model_name, _dims)
        logger.info(
            "Local embedding server not ready yet; memory scoring will use keyword mode "
            "until the server is available (heartbeat will retry)."
        )
        return None

    # HTTP-based provider
    _base_url = base_url or os.environ.get("EMBEDDING_BASE_URL", "")
    _model_name = model_name or os.environ.get("EMBEDDING_MODEL_NAME", "")

    if not _base_url or not _model_name:
        logger.info(
            "Qdrant embedding_fn: disabled (EMBEDDING_BASE_URL or EMBEDDING_MODEL_NAME not set). "
            "Memory search will use FTS-only mode. LLM will handle topic search via memory_search tool."
        )
        return None


def _get_local_server_if_available(api_key: str | None) -> object | None:
    """Return the active LocalEmbeddingServer if provider is 'local' and ready."""
    if api_key != "local":
        return None
    try:
        from lightclaw.infra.embedding.manager import get_active_local_server

        server = get_active_local_server()
        if server is not None and server.is_ready:
            return server
    except ImportError:
        pass
    return None


async def _wait_for_local_server(timeout_seconds: float = 15.0) -> object | None:
    """Poll get_active_local_server() until the embedding server is ready or timeout.

    Called when create_embedding_fn() discovers the local server isn't ready yet.
    This avoids permanent FTS-only degradation when the model loads ~3 s after
    init_memory_system() checks for it.
    """
    try:
        from lightclaw.infra.embedding.manager import get_active_local_server
    except ImportError:
        return None

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        server = get_active_local_server()
        if server is not None and getattr(server, "is_ready", False):
            logger.info(
                "Local embedding server became ready after %.1fs wait",
                timeout_seconds - (deadline - asyncio.get_event_loop().time()),
            )
            return server
        await asyncio.sleep(0.5)

    logger.info(
        "Local embedding server still not ready after %.0fs timeout; "
        "falling back to FTS-only mode (background task will retrofit once ready).",
        timeout_seconds,
    )
    return None


def _create_local_embedding_fn(
    local_server: Any,
    model_name: str,
    dims: int,
) -> Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]]:
    """Create embedding_fn that calls LocalEmbeddingServer.embed() in-process."""
    _max_batch = int(os.environ.get("EMBEDDING_MAX_BATCH_SIZE", "10"))
    _cache_enabled = os.environ.get("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    _cache_max_size = int(os.environ.get("EMBEDDING_MAX_CACHE_SIZE", "1000"))
    _cache: dict[str, list[float]] = OrderedDict()
    _first_call_logged = False

    async def _raw_embed_batch(batch: list[str]) -> list[list[float]]:
        """In-process batch embedding via fastembed (no HTTP)."""
        nonlocal _first_call_logged
        try:
            result = await local_server.embed(batch)
            if not _first_call_logged:
                sample_dim = len(result[0]) if result and result[0] else 0
                logger.info(
                    "✓ Local embedding first call OK: %d texts → dim=%d",
                    len(batch),
                    sample_dim,
                )
                _first_call_logged = True
            return result
        except Exception as exc:
            logger.warning("Local embedding batch failed: %s", exc)
            return [[] for _ in batch]

    async def _embedding_fn(texts: list[str]) -> list[list[float]]:
        if not _cache_enabled:
            results: list[list[float]] = []
            for i in range(0, len(texts), _max_batch):
                results.extend(await _raw_embed_batch(texts[i : i + _max_batch]))
            return results

        results = [None] * len(texts)  # type: ignore[list-item]
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, text in enumerate(texts):
            if text in _cache:
                _cache.move_to_end(text)
                results[idx] = _cache[text]
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            miss_embeddings: list[list[float]] = []
            for i in range(0, len(miss_texts), _max_batch):
                miss_embeddings.extend(await _raw_embed_batch(miss_texts[i : i + _max_batch]))

            for text, emb in zip(miss_texts, miss_embeddings, strict=False):
                if emb:
                    _cache[text] = emb
                    if len(_cache) > _cache_max_size:
                        _cache.popitem(last=False)

            for idx, emb in zip(miss_indices, miss_embeddings, strict=False):
                results[idx] = emb

        return results  # type: ignore[return-value]

    hit_info = f"cache={'on' if _cache_enabled else 'off'}, max_size={_cache_max_size}"
    logger.info(
        "Qdrant embedding_fn: created IN-PROCESS (model=%s, dims=%d, %s)",
        model_name,
        dims,
        hit_info,
    )
    return _embedding_fn


def _create_http_embedding_fn(
    api_key: str | None,
    base_url: str,
    model_name: str,
    dims: int,
) -> Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]] | None:
    """Create embedding_fn that uses AsyncOpenAI HTTP client."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key or "dummy", base_url=base_url)
        _model = model_name
        _max_batch = int(os.environ.get("EMBEDDING_MAX_BATCH_SIZE", "10"))

        _cache_enabled = os.environ.get("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
        _cache_max_size = int(os.environ.get("EMBEDDING_MAX_CACHE_SIZE", "1000"))
        _cache: dict[str, list[float]] = OrderedDict()
        _first_call_logged = False

        async def _raw_embed_batch(batch: list[str]) -> list[list[float]]:
            """HTTP batch embedding API call."""
            nonlocal _first_call_logged
            try:
                resp = await client.embeddings.create(model=_model, input=batch)
                vectors = [item.embedding for item in resp.data]
                if not _first_call_logged:
                    sample_dim = len(vectors[0]) if vectors and vectors[0] else 0
                    logger.info(
                        "✓ HTTP embedding first call OK: %d texts → dim=%d (url=%s)",
                        len(batch),
                        sample_dim,
                        base_url,
                    )
                    _first_call_logged = True
                return vectors
            except Exception as exc:
                if not _first_call_logged:
                    logger.warning(
                        "✗ HTTP embedding first call FAILED (url=%s): %s",
                        base_url,
                        exc,
                    )
                    _first_call_logged = True
                else:
                    logger.warning("Embedding batch failed: %s", exc)
                return [[] for _ in batch]

        async def _embedding_fn(texts: list[str]) -> list[list[float]]:
            if not _cache_enabled:
                results: list[list[float]] = []
                for i in range(0, len(texts), _max_batch):
                    results.extend(await _raw_embed_batch(texts[i : i + _max_batch]))
                return results

            results = [None] * len(texts)  # type: ignore[list-item]
            miss_indices: list[int] = []
            miss_texts: list[str] = []

            for idx, text in enumerate(texts):
                if text in _cache:
                    _cache.move_to_end(text)
                    results[idx] = _cache[text]
                else:
                    miss_indices.append(idx)
                    miss_texts.append(text)

            if miss_texts:
                miss_embeddings: list[list[float]] = []
                for i in range(0, len(miss_texts), _max_batch):
                    miss_embeddings.extend(await _raw_embed_batch(miss_texts[i : i + _max_batch]))

                for text, emb in zip(miss_texts, miss_embeddings, strict=False):
                    if emb:
                        _cache[text] = emb
                        if len(_cache) > _cache_max_size:
                            _cache.popitem(last=False)

                for idx, emb in zip(miss_indices, miss_embeddings, strict=False):
                    results[idx] = emb

            return results  # type: ignore[return-value]

        hit_info = f"cache={'on' if _cache_enabled else 'off'}, max_size={_cache_max_size}"
        logger.info(
            "Qdrant embedding_fn: created HTTP (model=%s, dims=%d, base_url=%s, %s)",
            _model,
            dims,
            base_url,
            hit_info,
        )
        return _embedding_fn

    except ImportError:
        logger.warning("openai package not installed; Qdrant embedding_fn disabled. Install with: pip install openai")
        return None


async def init_memory_system(
    working_dir: str,
    memory_manager: MemoryManager,
    on_proactive_trigger: Callable[[str], Coroutine[Any, Any, None]] | None = None,
) -> tuple[QdrantMemoryIndexer, QdrantMemorySearcher, ProactiveMemoryEngine | None]:
    """Initialize Qdrant memory system and inject into MemoryManager.

    Aligns with openclaw MemoryIndexManager.get() factory method.

    Args:
        working_dir           — Agent working directory (where MEMORY.md is located)
        memory_manager        — MemoryManager instance (injection target)
        on_proactive_trigger  — Optional proactive trigger callback

    Returns:
        (indexer, searcher, proactive_engine)
        proactive_engine is None if proactive mode not enabled.
    """
    from lightclaw.agent.memory.proactive_engine import create_proactive_engine
    from lightclaw.agent.memory.qdrant_indexer import create_qdrant_indexer
    from lightclaw.agent.memory.qdrant_searcher import QdrantMemorySearcher

    # Step 1: Create embedding_fn (prioritize config file, env vars override)
    _emb_cfg = _load_embedding_config()
    embedding_fn = create_embedding_fn(
        api_key=os.environ.get("EMBEDDING_API_KEY") or _emb_cfg.get("api_key"),
        base_url=os.environ.get("EMBEDDING_BASE_URL") or _emb_cfg.get("base_url"),
        model_name=os.environ.get("EMBEDDING_MODEL_NAME") or _emb_cfg.get("model_name"),
        dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS") or _emb_cfg.get("dimensions") or 768),
    )

    # If the local embedding server is still loading (background task from
    # EmbeddingManager.startup), wait up to WAIT_FOR_LOCAL_EMBEDDING_TIMEOUT
    # seconds before falling back to FTS-only.  This closure avoids permanent
    # degradation when the model finishes loading 2-3 s after this init call.
    _api_key = os.environ.get("EMBEDDING_API_KEY") or _emb_cfg.get("api_key", "")
    _wait_timeout = float(os.environ.get("WAIT_FOR_LOCAL_EMBEDDING_TIMEOUT", "15"))
    if embedding_fn is None and _api_key == "local" and _wait_timeout > 0:
        local_server = await _wait_for_local_server(_wait_timeout)
        if local_server is not None:
            _model_name = os.environ.get("EMBEDDING_MODEL_NAME") or _emb_cfg.get("model_name", "")
            _dims = int(os.environ.get("EMBEDDING_DIMENSIONS") or _emb_cfg.get("dimensions") or 768)
            embedding_fn = _create_local_embedding_fn(local_server, _model_name, _dims)
            logger.info("Embedding function created after waiting for local server")

    # Step 2: Read Qdrant configuration
    qdrant_backend = os.environ.get("QDRANT_BACKEND", "local")
    qdrant_path = os.environ.get("QDRANT_PATH", os.path.expanduser("~/.lightclaw/qdrant_data"))
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection_name = os.environ.get("QDRANT_COLLECTION", "lightclaw_memory")
    model_name = os.environ.get("EMBEDDING_MODEL_NAME") or _emb_cfg.get("model_name", "")

    # 自动推导向量维度：优先环境变量，再用配置的推导方法
    if os.environ.get("EMBEDDING_DIMENSIONS"):
        vector_dims = int(os.environ.get("EMBEDDING_DIMENSIONS"))
    else:
        # 从配置推导维度（考虑模型类型）
        try:
            from lightclaw.config.utils import load_config

            cfg = load_config()
            vector_dims = cfg.embedding.get_vector_dimensions()
        except Exception:
            vector_dims = 768  # 最后的兜底值

    data_dir = os.path.expanduser("~/.lightclaw")

    # Auto-detect actual embedding dimension by probing the model.
    # This prevents mismatch when configured dimension differs from model output.
    if embedding_fn is not None:
        try:
            probe_result = await embedding_fn(["dimension probe"])
            if probe_result and probe_result[0]:
                actual_dims = len(probe_result[0])
                if actual_dims != vector_dims:
                    logger.warning(
                        "Embedding dimension auto-detected: actual=%d, configured=%d. Using actual dimension.",
                        actual_dims,
                        vector_dims,
                    )
                    vector_dims = actual_dims
                else:
                    logger.debug("Embedding dimension confirmed: %d", vector_dims)
        except Exception as exc:
            logger.warning("Embedding dimension probe failed, using configured=%d: %s", vector_dims, exc)

    logger.info(
        "Qdrant memory system: backend=%s, collection=%s, vector_dims=%d, working_dir=%s",
        qdrant_backend,
        collection_name,
        vector_dims,
        working_dir,
    )

    # Step 3: Create QdrantMemoryIndexer
    indexer = await create_qdrant_indexer(
        working_dir=working_dir,
        embedding_fn=embedding_fn,
        qdrant_backend=qdrant_backend,
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        model_name=model_name,
        vector_dims=vector_dims,
        data_dir=data_dir,
    )

    # Step 4: Full sync + start file watching
    await indexer.start()

    # Step 5: Create QdrantMemorySearcher
    searcher = QdrantMemorySearcher(
        indexer=indexer,
        embedding_fn=embedding_fn,
    )

    # Step 6: Inject into MemoryManager
    memory_manager.set_memory_backend(indexer, searcher)
    logger.info("Qdrant memory system: injected into MemoryManager")

    # 7. 创建 ProactiveMemoryEngine（Qdrant 可用时始终创建，由 setup_proactive_engine 统一启动）
    # 不再受 PROACTIVE_MEMORY_ENABLED 环境变量门控——该变量仅用于 standalone 模式。
    # 集成模式下，engine 的 _run_loop 由 cron manager 调用 setup_proactive_engine() 时启动。
    proactive_engine = create_proactive_engine(
        indexer=indexer,
        embedding_fn=embedding_fn,
    )
    logger.info("ProactiveMemoryEngine: created (will be started by cron manager)")

    # If the embedding function is still None (local server hasn't finished
    # loading within the timeout), schedule a background task that retrofits
    # the embedding_fn into the indexer, searcher, and proactive engine as
    # soon as the server becomes ready.  This only happens on slow first-time
    # model downloads; the common cached case resolves during the wait above.
    _api_key = os.environ.get("EMBEDDING_API_KEY") or _emb_cfg.get("api_key", "")
    if embedding_fn is None and _api_key == "local":
        _model_name_retry = os.environ.get("EMBEDDING_MODEL_NAME") or _emb_cfg.get("model_name", "")
        _dims_retry = int(os.environ.get("EMBEDDING_DIMENSIONS") or _emb_cfg.get("dimensions") or 768)

        async def _retrofit_embedding_fn() -> None:
            """Background task: wait for local server, then patch indexer+searcher."""
            server = await _wait_for_local_server(timeout_seconds=300)  # 5 min for download
            if server is None:
                logger.warning(
                    "Embedding retrofit gave up after 300 s; vector search permanently disabled."
                )
                return
            fn = _create_local_embedding_fn(server, _model_name_retry, _dims_retry)
            indexer.embedding_fn = fn
            searcher.embedding_fn = fn
            if proactive_engine is not None:
                proactive_engine.embedding_fn = fn
            logger.info("Embedding function retrofitted into indexer, searcher, and proactive engine")

        asyncio.ensure_future(_retrofit_embedding_fn())
        logger.info(
            "Scheduled background retrofit of embedding_fn (will patch indexer+searcher "
            "when local server is ready)"
        )

    return indexer, searcher, proactive_engine


async def init_fts_only_system(
    working_dir: str,
    memory_manager: MemoryManager,
) -> tuple[QdrantMemoryIndexer, QdrantMemorySearcher, ProactiveMemoryEngine]:
    """FTS-only degraded initialization: skip Qdrant, use only SQLite FTS5.

    Called as fallback when init_memory_system() fails.

    Args:
        working_dir    — Agent working directory (where MEMORY.md is located)
        memory_manager — MemoryManager instance (injection target)

    Returns:
        (indexer, searcher, proactive_engine)
        indexer.client = None; all Qdrant write operations are safely skipped.
    """
    from lightclaw.agent.memory.qdrant_indexer import QdrantMemoryIndexer
    from lightclaw.agent.memory.qdrant_schema import open_fts_db
    from lightclaw.agent.memory.qdrant_searcher import QdrantMemorySearcher

    data_dir = os.path.expanduser("~/.lightclaw")
    fts_db = open_fts_db(data_dir)

    indexer = QdrantMemoryIndexer(
        working_dir=working_dir,
        embedding_fn=None,  # FTS-only: no vectors
        qdrant_client=None,  # Skip all Qdrant operations
        fts_db=fts_db,
        collection_name="lightclaw_memory",
        model_name="",
        vector_dims=768,
    )
    await indexer.start()

    searcher = QdrantMemorySearcher(
        indexer=indexer,
        embedding_fn=None,  # FTS-only mode
    )

    memory_manager.set_memory_backend(indexer, searcher)
    logger.info("FTS-only memory system: injected into MemoryManager (Qdrant unavailable)")

    # Create proactive engine even in FTS-only mode so proactivity flow stays
    # available. With embedding_fn=None it will naturally run caring-mode.
    from lightclaw.agent.memory.proactive_engine import create_proactive_engine

    proactive_engine = create_proactive_engine(
        indexer=indexer,
        embedding_fn=None,
    )
    logger.info("ProactiveMemoryEngine: created for FTS-only mode")

    return indexer, searcher, proactive_engine


__all__ = [
    "create_embedding_fn",
    "init_fts_only_system",
    "init_memory_system",
    "is_embedding_configured",
    "shutdown_memory_system",
]


def shutdown_memory_system(
    indexer: QdrantMemoryIndexer | None,
    proactive_engine: ProactiveMemoryEngine | None,
) -> None:
    """Shutdown Qdrant memory system (called when agent stops)."""
    if proactive_engine is not None:
        try:
            proactive_engine.stop()
        except Exception as exc:
            logger.warning("ProactiveMemoryEngine stop failed: %s", exc)

    if indexer is not None:
        try:
            indexer.close()
        except Exception as exc:
            logger.warning("QdrantMemoryIndexer close failed: %s", exc)
