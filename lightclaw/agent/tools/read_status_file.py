"""Heartbeat-safe status file reader — logical-target-based, no arbitrary paths.

Replaces generic ``read_file`` for heartbeat agent turns. Only accepts
predefined logical target names; rejects raw paths, ``..`` traversal,
and absolute paths.  Returns at most *max_chars* characters.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from lightclaw.agent.tools.result import LightClawToolResult

# -- resolved at call time so tests can monkeypatch --
_HEARTBEAT_DIR = Path("/root/.lightclaw/workspace/memory/.heartbeat")
_DREAMS_DIR = Path("/root/.lightclaw/workspace/memory/.dreams")
_HEARTBEAT_MD = Path("/root/.lightclaw/workspace/HEARTBEAT.md")

# Logical target → (absolute-path resolver, description)
_TARGETS: dict[str, tuple] = {}


def _build_targets() -> dict[str, tuple]:
    """Lazily build the target map (one-off)."""
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "heartbeat_state": (
            lambda: _HEARTBEAT_DIR / "_state.json",
            "Heartbeat state (JSON)",
        ),
        "heartbeat_today_log": (
            lambda: _HEARTBEAT_DIR / f"{today}.log",
            "Heartbeat daily log",
        ),
        "heartbeat_error_fingerprints": (
            lambda: _HEARTBEAT_DIR / "_error_fingerprints.json",
            "Error fingerprint state (JSON)",
        ),
        "dreaming_status": (
            lambda: _DREAMS_DIR / "status.json",
            "Dreaming pipeline status (JSON)",
        ),
        "heartbeat_prompt": (
            lambda: _HEARTBEAT_MD,
            "Heartbeat task prompt (Markdown)",
        ),
    }


_SAFE_PREFIXES = (
    str(_HEARTBEAT_DIR.resolve()) + "/",
    str(_DREAMS_DIR.resolve()) + "/",
    str(_HEARTBEAT_MD.resolve()),
)


async def read_status_file(target: str, max_chars: int = 4000) -> LightClawToolResult:
    """Read a fixed-status file by logical name — no arbitrary paths.

    Allowed *target* values:

    - ``"heartbeat_state"`` — ``memory/.heartbeat/_state.json``
    - ``"heartbeat_today_log"`` — ``memory/.heartbeat/YYYY-MM-DD.log``
    - ``"heartbeat_error_fingerprints"`` — ``memory/.heartbeat/_error_fingerprints.json``
    - ``"dreaming_status"`` — ``memory/.dreams/status.json``
    - ``"heartbeat_prompt"`` — ``HEARTBEAT.md``

    Args:
        target:     Logical target name (see above).  Raw paths are **rejected**.
        max_chars:  Maximum characters to return (default 4000, hard-cap 8000).

    Returns:
        ``LightClawToolResult`` with the file content truncated to *max_chars*,
        or an error message string.
    """
    # ── Validate target ──────────────────────────────────────────────
    targets = _build_targets()
    entry = targets.get(target)
    if entry is None:
        valid = ", ".join(sorted(targets.keys()))
        return LightClawToolResult(
            text=f"Error: unknown target '{target}'. Valid targets: {valid}"
        )

    resolver, _desc = entry
    try:
        file_path = resolver()
    except Exception as exc:
        return LightClawToolResult(text=f"Error: failed to resolve target '{target}': {exc}")

    # ── Path-safety check (defence in depth) ─────────────────────────
    try:
        resolved = file_path.resolve()
    except Exception:
        return LightClawToolResult(text=f"Error: cannot resolve path for target '{target}'")
    if not any(str(resolved) == p or str(resolved).startswith(p) for p in _SAFE_PREFIXES):
        return LightClawToolResult(
            text=f"Error: resolved path '{resolved}' is outside allowed directories for target '{target}'"
        )

    # ── Read ─────────────────────────────────────────────────────────
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LightClawToolResult(
            text=f"(file not found) target={target} path={file_path}"
        )
    except Exception as exc:
        return LightClawToolResult(
            text=f"Error: read failed for target '{target}': {exc}"
        )

    # ── Format JSON files lightly ────────────────────────────────────
    if file_path.suffix == ".json":
        try:
            data = json.loads(text)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass  # return raw text

    # ── Truncate ─────────────────────────────────────────────────────
    cap = max(1, min(max_chars, 8000))
    if len(text) > cap:
        suffix = f"\n... (truncated, total {len(text)} chars)"
        cut = max(cap - len(suffix), 1)
        text = text[:cut] + suffix

    return LightClawToolResult(text=text)
