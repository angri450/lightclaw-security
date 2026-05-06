"""Heartbeat error summary tool — reads fingerprint state + daily log only.

Never reads the full ``lightclaw.log``.  Pulls from:
1. Error fingerprint state (``_error_fingerprints.json``) — active/unresolved errors
2. Heartbeat daily log (``YYYY-MM-DD.log``) — recent cycle outcomes
3. As a last resort, tail of ``lightclaw.log`` — only ERROR/CRITICAL lines,
   at most 300 lines / 64 KB, redacted for secrets.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from lightclaw.agent.tools.result import LightClawToolResult

_HEARTBEAT_DIR = Path("/root/.lightclaw/workspace/memory/.heartbeat")
_MAIN_LOG = Path("/root/.lightclaw/logs/lightclaw.log")

# Simple secret patterns — if a line matches any of these, redact the value.
_SECRET_RE = re.compile(
    r"(api[_-]?key|secret|token|password|cookie|auth|credential)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _redact_secrets(line: str) -> str:
    return _SECRET_RE.sub(r"\1=***REDACTED***", line)


async def read_recent_heartbeat_errors(
    max_items: int = 10,
    max_chars: int = 4000,
) -> LightClawToolResult:
    """Return a concise summary of recent heartbeat errors.

    Does **not** read the full ``lightclaw.log``.  Priority order:

    1. Active/unresolved fingerprint entries from ``_error_fingerprints.json``.
    2. Last few lines of today's heartbeat log (``YYYY-MM-DD.log``).
    3. Tail of ``lightclaw.log`` (last 300 lines / 64 KB), only ERROR/CRITICAL
       lines, redacted for secrets.

    Args:
        max_items:  Max error entries to return (default 10, hard-cap 20).
        max_chars:  Max total chars to return (default 4000, hard-cap 8000).

    Returns:
        ``LightClawToolResult`` with a human-readable error summary.
    """
    cap_items = max(1, min(max_items, 20))
    cap_chars = max(1, min(max_chars, 8000))

    parts: list[str] = []

    # ── 1. Fingerprint state ─────────────────────────────────────────
    fp_file = _HEARTBEAT_DIR / "_error_fingerprints.json"
    active_fingerprints: list[dict] = []
    try:
        if fp_file.exists():
            data = json.loads(fp_file.read_text(encoding="utf-8"))
            fps = (data.get("fingerprints") if isinstance(data, dict) else {}) or {}
            for key, entry in fps.items():
                if isinstance(entry, dict) and not entry.get("resolved", False):
                    entry["_fp_key"] = key
                    active_fingerprints.append(entry)
    except Exception:
        pass

    if active_fingerprints:
        active_fingerprints.sort(key=lambda e: e.get("last_seen", ""), reverse=True)
        parts.append(f"=== Active error fingerprints ({len(active_fingerprints)} total) ===")
        for fp in active_fingerprints[:cap_items]:
            sample = (fp.get("sample") or "")[:300]
            parts.append(
                f"- [{fp.get('last_seen','?')}] "
                f"seen={fp.get('seen_count',0)} "
                f"missed={fp.get('missed_cycles',0)} | "
                f"{_redact_secrets(sample)}"
            )

    # ── 2. Today's heartbeat log (last 10 lines) ─────────────────────
    today = datetime.now().strftime("%Y-%m-%d")
    hb_log = _HEARTBEAT_DIR / f"{today}.log"
    try:
        if hb_log.exists():
            lines = hb_log.read_text(encoding="utf-8").strip().splitlines()
            recent = [l for l in lines[-10:] if "ERROR" in l or "AGENT_TURN_ERROR" in l]
            if recent:
                parts.append("")
                parts.append("=== Recent heartbeat log entries (today) ===")
                for line in recent[-5:]:
                    parts.append(_redact_secrets(line[:300]))
    except Exception:
        pass

    # ── 3. Tail of lightclaw.log (only ERROR/CRITICAL lines) ─────────
    if not parts:
        try:
            if _MAIN_LOG.exists():
                _tail = _read_tail_errors(_MAIN_LOG, max_lines=300, max_bytes=65536)
                if _tail:
                    parts.append("=== Tail of lightclaw.log (ERROR/CRITICAL only) ===")
                    for line in _tail[:cap_items]:
                        parts.append(_redact_secrets(line[:300]))
        except Exception:
            parts.append("(unable to read lightclaw.log tail)")

    if not parts:
        return LightClawToolResult(text="No recent heartbeat errors found.")

    result = "\n".join(parts)
    if len(result) > cap_chars:
        suffix = f"\n... (truncated, total {len(result)} chars)"
        cut = max(cap_chars - len(suffix), 1)
        result = result[:cut] + suffix

    return LightClawToolResult(text=result)


def _read_tail_errors(path: Path, max_lines: int, max_bytes: int) -> list[str]:
    """Read only the tail of *path*, returning ERROR/CRITICAL lines."""
    file_size = path.stat().st_size
    read_size = min(file_size, max_bytes)
    with open(path, "rb") as f:
        if file_size > read_size:
            f.seek(-read_size, os.SEEK_END)
        raw = f.read(read_size)
    # Skip partial first line
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if file_size > read_size and lines:
        lines = lines[1:]  # drop partial first line
    lines = lines[-max_lines:]
    return [l for l in lines if l.startswith("ERROR ") or l.startswith("CRITICAL ")]
