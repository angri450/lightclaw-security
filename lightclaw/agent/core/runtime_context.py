"""Per-turn runtime context schema for shared LangGraph graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LightClawRuntimeContext:
    """Mutable per-turn context passed via LangGraph ``context``.

    This object carries request-scoped state that must not live on a shared
    middleware instance when the compiled graph is reused across turns.
    """

    session_id: str = ""
    turn_id: str = ""
    channel: str = ""
    compressed_summary: str = ""
    model_call_index: int = 0
    token_log_records: list[Any] = field(default_factory=list)
    cancelled: bool = False
    compactor: Any | None = None
    # Per-turn model selected by the router (may differ from the shared graph's
    # default when the request contains images or other modality signals).
    turn_model: Any | None = None
    # ActorContext for session/principal isolation firewall (set by SessionFactory
    # from TurnRequest.context, consumed by coordinator recall/authz pipeline).
    actor_context: Any | None = None
    # ── Tool policy (P1b heartbeat tool whitelist) ──────────────────────────
    # When non-None, only tools whose name is in this list are exposed to the
    # model.  None means "no filtering" (default for all normal user turns).
    allowed_tools: list[str] | None = None
    # Source label for tool policy (e.g. "heartbeat") — logged for diagnostics.
    tool_policy_source: str | None = None
    # ── Runtime profile / budget overrides (P1b) ────────────────────────────
    # Label identifying the runtime profile (e.g. "heartbeat").
    runtime_profile: str | None = None
    # Override for max_input_length (tokens).  When set, the final input guard
    # uses this value instead of the shared graph's default.
    max_input_length_override: int | None = None
    # Override for max_iters (turns).  When set, the runner/controller uses
    # this value instead of the shared graph's default.
    max_iters_override: int | None = None
