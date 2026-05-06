"""ContextVar-based runner/channel_manager access for tools.

P3.1: Tools like ``delegate_readonly_tasks`` need the runner instance
to spawn child workers.  This module provides an import-safe way to
share runner and channel_manager across all asyncio contexts without
circular imports.

Uses a module-level global as the primary store because uvicorn request
handler Tasks are NOT children of the lifespan Task and do not inherit
its ContextVars.  ContextVar is kept as a fast-path cache for same-Task
access (e.g. startup code within the lifespan).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_runner: Any = None
_channel_manager: Any = None

_runner_var: ContextVar[Any | None] = ContextVar("runner", default=None)
_channel_manager_var: ContextVar[Any | None] = ContextVar("channel_manager", default=None)


def set_runner_context(runner: Any, channel_manager: Any) -> None:
    """Store runner and channel_manager globally and in the current ContextVar scope.

    Called once during app startup (``_app.lifespan``).
    """
    global _runner, _channel_manager
    _runner = runner
    _channel_manager = channel_manager
    _runner_var.set(runner)
    _channel_manager_var.set(channel_manager)


def get_runner() -> Any:
    """Return the current runner instance (AgentRunner)."""
    r = _runner_var.get()
    if r is not None:
        return r
    return _runner


def get_channel_manager() -> Any:
    """Return the current channel_manager instance."""
    cm = _channel_manager_var.get()
    if cm is not None:
        return cm
    return _channel_manager
