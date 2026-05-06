# pylint: disable=redefined-outer-name,unused-argument
from __future__ import annotations

import asyncio
import os

# Suppress transformers advisory warning ("PyTorch was not found …") before
# any transitive import pulls in the transformers package.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lightclaw import __version__
from lightclaw.agent.skills_manager import ensure_skills_initialized
from lightclaw.agent.utils import copy_prompts
from lightclaw.app.channels import ChannelManager  # pylint: disable=no-name-in-module
from lightclaw.app.channels.utils import make_process_from_runner
from lightclaw.app.cron.manager import CronManager
from lightclaw.app.cron.repo.json_repo import JsonJobRepository
from lightclaw.app.envs import EnvWatcher, load_envs_into_environ
from lightclaw.app.routers import router as api_router
from lightclaw.app.runner import AgentRunner
from lightclaw.app.runner.manager import ChatManager
from lightclaw.app.runner.repo.json_repo import JsonChatRepository
from lightclaw.app.runner_context import set_runner_context
from lightclaw.app.shutdown_reason import clear_shutdown_reason, get_shutdown_reason, install_signal_reason_observer
from lightclaw.app.star_office_manager import ensure_star_office, stop_star_office
from lightclaw.app.tasks import TaskRegistry
from lightclaw.config import ConfigWatcher, load_config, update_last_dispatch
from lightclaw.config.utils import get_chats_path, get_jobs_path
from lightclaw.constant import DOCS_ENABLED, LOG_LEVEL_ENV, migrate_to_workspace
from lightclaw.infra.embedding.manager import EmbeddingManager
from lightclaw.utils.logging import setup_logger

# Apply log level on load so reload child process gets same level as CLI.
logger = setup_logger(os.environ.get(LOG_LEVEL_ENV, "info"))

# One-time migration: move *.md, skills/, memory/ etc. from BASE_DIR root
# into BASE_DIR/workspace/ (new layout introduced after initial release).
migrate_to_workspace()

# Load persisted env vars into os.environ at module import time
# so they are available before the lifespan starts.
load_envs_into_environ()

runner = AgentRunner()
cron_manager: CronManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=too-many-statements
    clear_shutdown_reason()
    install_signal_reason_observer(logger)

    app.state.skills_initialized = False
    try:
        ensure_skills_initialized()
        app.state.skills_initialized = True
        logger.debug("Built-in skills initialized during app startup")
    except Exception:
        logger.exception("Failed to initialize built-in skills on startup")

    # --- ensure workspace prompt files exist (BOOTSTRAP.md etc.) ---
    # copy_prompts uses skip_existing=True so user-edited files are never
    # overwritten; this only fills in files that are missing entirely.
    # Without this, a fresh `lightclaw run` (without prior `lightclaw init`)
    # would silently skip the bootstrap flow because BOOTSTRAP.md is absent.
    config = load_config()
    try:
        lang = config.agents.language
        copied = copy_prompts(lang, skip_existing=True)
        if copied:
            logger.info("Copied missing workspace prompt(s) [%s]: %s", lang, ", ".join(copied))
    except Exception:
        logger.exception("Failed to copy workspace prompts on startup (non-fatal)")

    # --- embedding manager init (local model / custom / none) ---
    # MUST run BEFORE runner.start() so that EMBEDDING_* env vars are
    # available when init_memory_system() creates the embedding_fn.
    embedding_manager = EmbeddingManager()
    try:
        await embedding_manager.startup(config.embedding)
    except Exception:
        logger.exception("Failed to start embedding manager (non-fatal, continuing)")

    # Local embedding model is loaded in a background task (non-blocking).
    # runner.start() → init_memory_system() waits up to
    # WAIT_FOR_LOCAL_EMBEDDING_TIMEOUT seconds (default 15) for the model,
    # creating the embedding_fn for vector+FTS hybrid mode the moment it's
    # ready.  If the model still hasn't loaded by the timeout (first-time
    # download), a background retrofit task patches the embedding_fn into
    # the indexer + searcher as soon as the server announces readiness.
    await runner.start()

    # --- task registry (must be created before channels / cron) ---
    task_registry = TaskRegistry()

    # --- channel connector init/start (from lightclaw.json) ---
    channel_manager = ChannelManager.from_config(
        process=make_process_from_runner(runner, task_registry=task_registry),
        config=config,
        on_last_dispatch=update_last_dispatch,
    )
    await channel_manager.start_all()

    # --- chat manager init and connect to runner.session ---
    chat_repo = JsonChatRepository(get_chats_path())
    chat_manager = ChatManager(
        repo=chat_repo,
    )

    runner.set_chat_manager(chat_manager)

    # --- cron init/start ---
    global cron_manager
    repo = JsonJobRepository(get_jobs_path())
    cron_manager = CronManager(
        repo=repo,
        runner=runner,
        channel_manager=channel_manager,
        timezone="UTC",
        task_registry=task_registry,
    )
    await cron_manager.start()

    # --- config file watcher (auto-reload channels on lightclaw.json change) ---
    async def _reload_channel(name: str, new_cfg: object, show_tool_details: bool) -> None:
        old_channel = await channel_manager.get_channel(name)
        if old_channel is None:
            raise ValueError(f"channel '{name}' not found in manager")
        new_channel = old_channel.clone(new_cfg, show_tool_details=show_tool_details)
        await channel_manager.replace_channel(new_channel)

    async def _reload_runtime(reason: str) -> None:
        await runner.refresh_langgraph_runtime(reason=reason)

    config_watcher = ConfigWatcher(
        on_channel_reload=_reload_channel,
        on_runtime_reload=_reload_runtime,
    )
    await config_watcher.start()

    # --- workspace .env watcher (hot-reload environment variables) ---
    env_watcher = EnvWatcher()
    await env_watcher.start()

    # expose to endpoints
    app.state.runner = runner
    app.state.channel_manager = channel_manager
    # Store runner/channel_manager in ContextVar for tool access (P3.1)
    set_runner_context(runner, channel_manager)
    app.state.cron_manager = cron_manager
    app.state.chat_manager = chat_manager
    app.state.config_watcher = config_watcher
    app.state.env_watcher = env_watcher
    app.state.embedding_manager = embedding_manager
    app.state.task_registry = task_registry

    # --- STT (faster-whisper) model is NOT pre-loaded at startup -----
    # The Whisper model will be lazily downloaded & loaded when the user
    # starts their first voice-chat session. This keeps startup fast and
    # avoids blocking on a ~145 MB download that only voice-chat needs.
    logger.info("STT model will be loaded on-demand (first voice session).")

    # --- Star Office auto-start (fire-and-forget background task) ---
    # Star Office is bundled with LightClaw.  When enabled, launch the
    # embedded Flask server in the background so the dashboard sees "ok"
    # on first health check.

    async def _try_start_star_office() -> None:
        try:
            so_cfg = config.star_office
            if not so_cfg.enabled or not so_cfg.url:
                return
            logger.info("Auto-starting Star Office UI (%s) …", so_cfg.url)
            await asyncio.to_thread(ensure_star_office, so_cfg.url.rstrip("/"))
            logger.info("Star Office UI auto-started successfully.")
        except Exception:
            logger.warning("Star Office UI auto-start failed (non-fatal).", exc_info=True)

    asyncio.create_task(_try_start_star_office())

    try:
        yield
    finally:
        shutdown_reason = get_shutdown_reason() or "unknown"
        logger.warning("LightClaw shutdown started: reason=%s", shutdown_reason)
        # stop order: star-office -> watchers -> cron -> channels -> runner
        with suppress(Exception):
            stop_star_office()
        with suppress(Exception):
            await config_watcher.stop()
        with suppress(Exception):
            await env_watcher.stop()
        try:
            await cron_manager.stop()
        finally:
            await channel_manager.stop_all()
            await runner.stop()
            with suppress(Exception):
                await embedding_manager.shutdown()
            # 确保所有 Langfuse trace 数据在退出前发送
            with suppress(Exception):
                from lightclaw.app.observability import flush_langfuse

                flush_langfuse()


app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)


# Dashboard static dir: env, or lightclaw package data (dashboard), or cwd.
_CONSOLE_STATIC_ENV = "LIGHTCLAW_CONSOLE_STATIC_DIR"


def _resolve_console_static_dir() -> str:
    if os.environ.get(_CONSOLE_STATIC_ENV):
        return os.environ[_CONSOLE_STATIC_ENV]
    # Shipped dist lives in lightclaw package as static data (not a Python pkg).
    pkg_dir = Path(__file__).resolve().parent.parent
    candidate = pkg_dir / "dashboard"
    if candidate.is_dir() and (candidate / "index.html").exists():
        return str(candidate)
    cwd = Path(os.getcwd())
    for subdir in ("dashboard/dist", "dashboard_dist"):
        candidate = cwd / subdir
        if candidate.is_dir() and (candidate / "index.html").exists():
            return str(candidate)
    return str(cwd / "dashboard" / "dist")


_CONSOLE_STATIC_DIR = _resolve_console_static_dir()
_CONSOLE_INDEX = Path(_CONSOLE_STATIC_DIR) / "index.html" if _CONSOLE_STATIC_DIR else None
logger.info(f"STATIC_DIR: {_CONSOLE_STATIC_DIR}")


@app.get("/")
def read_root():
    if _CONSOLE_INDEX and _CONSOLE_INDEX.exists():
        return FileResponse(_CONSOLE_INDEX)
    return {"message": "Hello World"}


@app.get("/api/version")
def get_version():
    """Return the current LightClaw version."""
    return {"version": __version__}


app.include_router(api_router, prefix="/api")

# Mount dashboard: root static files (logo.png etc.) then assets, then SPA
# fallback.
if os.path.isdir(_CONSOLE_STATIC_DIR):
    _console_path = Path(_CONSOLE_STATIC_DIR)

    @app.get("/logo.png")
    def _console_logo():
        f = _console_path / "logo.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")

        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/logo.svg")
    def _console_logo_svg():
        f = _console_path / "logo.svg"
        if f.is_file():
            return FileResponse(f, media_type="image/svg+xml")

        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/lightclaw-symbol.svg")
    def _console_icon():
        f = _console_path / "lightclaw-symbol.svg"
        if f.is_file():
            return FileResponse(f, media_type="image/svg+xml")

        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/lightclaw-logo.png")
    def _console_lightclaw_logo():
        f = _console_path / "lightclaw-logo.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")

        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/lightclaw-symbol.png")
    def _console_lightclaw_symbol():
        f = _console_path / "lightclaw-symbol.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")

        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/manifest.json")
    def _console_manifest():
        """Serve Web App Manifest with correct MIME type required by PWA spec."""
        f = _console_path / "manifest.json"
        if f.is_file():
            return FileResponse(
                f,
                media_type="application/manifest+json",
                headers={"Cache-Control": "no-cache"},
            )
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/sw.js")
    def _console_sw():
        """Service Worker must be served with no-cache so updates propagate immediately."""
        f = _console_path / "sw.js"
        if f.is_file():
            return FileResponse(
                f,
                media_type="application/javascript",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/workbox-{suffix}")
    def _console_workbox(suffix: str):
        """Workbox runtime chunks — long-lived cache is fine (content-hashed names)."""
        f = _console_path / f"workbox-{suffix}"
        if f.is_file():
            return FileResponse(
                f,
                media_type="application/javascript",
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )
        raise HTTPException(status_code=404, detail="Not Found")

    class _CachedStaticFiles(StaticFiles):
        """StaticFiles with Cache-Control: no-cache — always revalidate via ETag."""

        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    _assets_dir = _console_path / "assets"
    if _assets_dir.is_dir():
        app.mount(
            "/assets",
            _CachedStaticFiles(directory=str(_assets_dir)),
            name="assets",
        )

    _models_dir = _console_path / "models"
    if _models_dir.is_dir():
        app.mount(
            "/models",
            _CachedStaticFiles(directory=str(_models_dir)),
            name="models",
        )

    _videos_dir = _console_path / "videos"
    if _videos_dir.is_dir():
        app.mount(
            "/videos",
            StaticFiles(directory=str(_videos_dir)),
            name="videos",
        )

    _gifs_dir = _console_path / "gifs"
    if _gifs_dir.is_dir():
        app.mount(
            "/gifs",
            _CachedStaticFiles(directory=str(_gifs_dir)),
            name="gifs",
        )

    @app.get("/{full_path:path}")
    def _console_spa(full_path: str):
        # Serve root-level static files (e.g. pcm-processor.js) before SPA fallback
        candidate = _console_path / full_path
        if candidate.is_file() and ".." not in full_path and not full_path.startswith("/"):
            import mimetypes

            mt, _ = mimetypes.guess_type(str(candidate))
            return FileResponse(candidate, media_type=mt or "application/octet-stream")

        if _CONSOLE_INDEX and _CONSOLE_INDEX.exists():
            return FileResponse(_CONSOLE_INDEX)

        raise HTTPException(status_code=404, detail="Not Found")
