"""Proactivity: agent-initiated messages with probability gate.

v2: ProactiveMemoryEngine drives the scheduling loop. This module registers
an ``on_evaluate`` callback that implements:
  - active hours check
  - active runs check
  - probability gate (activity_level * 0.5^unanswered)
  - dual mode: topic (with Qdrant candidates) / caring (no candidates, ×0.3 gate)
  - LLM agent invocation via run_cron_agent
  - unanswered_count tracking

Fallback: if engine is unavailable, ``run_proactivity_once`` delegates to
``_run_proactivity_once_legacy`` which uses the original agent-only flow.

A separate daily analysis job uses an LLM to adjust activity_level.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightclaw.config import get_proactivity_config, load_config
from lightclaw.constant import WORKING_DIR
from lightclaw.agent.utils.token_counting import strip_model_special_tokens

if TYPE_CHECKING:
    from lightclaw.agent.memory.proactive_engine import CandidateResult

# Delay (seconds) before incrementing unanswered_count after a proactive
# message, giving the user time to reply first.
_UNANSWERED_DELAY_SECONDS = 5 * 60

# Caring mode secondary probability gate multiplier.
# When no Qdrant candidates are found, the effective probability is scaled
# by this factor so care/greeting messages are sent less frequently.
_CARING_MODE_FACTOR = 0.3

try:
    from lightclaw.agent.core.engines.langgraph.model_factory import create_chat_model
except Exception:  # pragma: no cover – may not be available at import time
    create_chat_model = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_IDENTITY_FILE = "IDENTITY.md"

DEFAULT_ACTIVITY_LEVEL = 0.5


# ── IDENTITY.md read/write utilities ─────────────────────────────────────────


def _identity_path() -> Path:
    return WORKING_DIR / _IDENTITY_FILE


def read_activity_level() -> float:
    """Read ``activity_level`` from IDENTITY.md. Returns 0.5 on any error."""
    path = _identity_path()
    if not path.is_file():
        return DEFAULT_ACTIVITY_LEVEL
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("proactivity: cannot read %s", path)
        return DEFAULT_ACTIVITY_LEVEL

    m = re.search(r"activity_level:\s*(\S+)", text)
    if not m:
        return DEFAULT_ACTIVITY_LEVEL
    try:
        val = float(m.group(1))
    except (ValueError, TypeError):
        logger.warning("proactivity: invalid activity_level=%r, using default", m.group(1))
        return DEFAULT_ACTIVITY_LEVEL
    return max(0.0, min(1.0, val))


def read_identity_field(field: str, default: str = "0") -> str:
    """Read a named field from IDENTITY.md. Returns *default* on any error."""
    path = _identity_path()
    if not path.is_file():
        return default
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default
    m = re.search(rf"{re.escape(field)}:\s*(\S+)", text)
    return m.group(1) if m else default


def write_activity_field(field: str, value: str) -> None:
    """Update a single key-value field in IDENTITY.md (in-place regex replace).

    If the field does not exist yet (e.g. workspace upgraded from an older
    version), it is appended under the ``## Proactivity`` section.  If even
    the section header is missing, the whole section is created at the end of
    the file.
    """
    path = _identity_path()
    if not path.is_file():
        logger.warning("proactivity: cannot write field %s — %s not found", field, path)
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("proactivity: cannot read %s for write", path)
        return

    pattern = re.compile(rf"^({re.escape(field)}:\s*)([^\n]*)", re.MULTILINE)
    new_text, count = pattern.subn(rf"\g<1>{value}", text)

    if count == 0:
        # Field missing — append it under the Proactivity section.
        new_line = f"{field}: {value}"
        section_pattern = re.compile(r"^## Proactivity\b[^\n]*$", re.MULTILINE)
        m_section = section_pattern.search(text)
        if m_section:
            # Insert right after the section header line.
            insert_pos = m_section.end()
            new_text = text[:insert_pos] + "\n" + new_line + text[insert_pos:]
        else:
            # No section at all — create one at EOF.
            separator = (
                "\n\n" if text and not text.endswith("\n\n") else ("\n" if text and not text.endswith("\n") else "")
            )
            new_text = text + separator + "## Proactivity\n\n" + new_line + "\n"
        logger.info("proactivity: appended missing field %s to %s", field, path)

    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError:
        logger.warning("proactivity: failed to write %s", path)


# ── Prompt loading ────────────────────────────────────────────────────────────


def _load_proactive_prompt() -> str:
    """Load PROACTIVE.md — built-in first, then append user custom.

    If the user-level PROACTIVE.md does not exist yet, copy the built-in
    template into WORKING_DIR so the user can customise it later.
    """
    import lightclaw.agent as _agent_pkg

    base_text = ""
    for lang in ("zh", "en"):
        candidate = Path(_agent_pkg.__file__).parent / "prompts" / lang / "PROACTIVE.md"
        if candidate.is_file():
            base_text = candidate.read_text(encoding="utf-8").strip()
            if base_text:
                break

    user_path = WORKING_DIR / "PROACTIVE.md"
    if not user_path.is_file():
        # First run: write built-in template to user working directory
        if base_text:
            try:
                user_path.parent.mkdir(parents=True, exist_ok=True)
                user_path.write_text(base_text + "\n", encoding="utf-8")
                logger.info("proactivity: wrote default PROACTIVE.md to %s", user_path)
            except OSError:
                logger.warning("proactivity: failed to write default PROACTIVE.md")
    else:
        try:
            user_text = user_path.read_text(encoding="utf-8").strip()
            if user_text:
                base_text = base_text + "\n\n" + user_text if base_text else user_text
        except OSError:
            pass

    return base_text or ""


# ── Candidate topic formatting ────────────────────────────────────────────────


def _format_candidates_section(
    candidates: list[CandidateResult],
    language: str = "zh",
) -> str:
    """Format Qdrant candidates as a structured prompt section for LLM agent.

    Output uses categorized sections to help the model distinguish between
    user facts, obligations, recent activity, and safe topics.  Each section
    provides the model with *background reference only* — it must never
    regurgitate or quote the raw data directly to the user.
    """
    if not candidates:
        return ""

    if language == "zh":
        # ── Categorize candidates by path heuristics ──
        user_info: list[str] = []
        obligations: list[str] = []
        recent: list[str] = []
        safe: list[str] = []

        for c in candidates:
            text = c.snippet[:200].strip()
            if not text:
                continue
            p = (c.path or "").lower()

            if any(kw in p for kw in ("user", "profile", "persona", "identity", "owner", "about")):
                user_info.append(f"- {text}")
            elif any(kw in p for kw in ("obligation", "task", "todo", "duty", "project", "plan")):
                obligations.append(f"- {text}")
            elif any(kw in p for kw in ("recent", "timeline", "session", "conversation", "chat")):
                recent.append(f"- {text}")
            else:
                safe.append(f"- {text}")

        lines = [
            "## 推荐话题",
            "",
            "以下是系统为你筛选的可能话题，只能作为**背景参考**：",
            "",
        ]

        if user_info:
            lines.append("### 用户信息")
            lines.extend(user_info)
            lines.append("")

        if obligations:
            lines.append("### 你可以关心的事")
            lines.extend(obligations)
            lines.append("")

        if recent:
            lines.append("### 最近动态")
            lines.extend(recent)
            lines.append("")

        if safe:
            lines.append("### 安全话题（可以聊的）")
            lines.extend(safe)
            lines.append("")

        if not (user_info or obligations or recent or safe):
            # Fallback: all candidates without any text after trimming
            return ""

        lines.append("### ⚠️ 避免")
        lines.append("- 不要复述上述信息给用户")
        lines.append('- 不要以"根据记忆"、"根据记录"开头')
        lines.append("- 如果推荐话题都不合适，可以调用 memory_search 搜索其他话题")
        lines.append("- 如果确实无话可说，回复 [SKIP]")
    else:
        # ── Categorize candidates by path heuristics (English) ──
        user_info: list[str] = []
        obligations: list[str] = []
        recent: list[str] = []
        safe: list[str] = []

        for c in candidates:
            text = c.snippet[:200].strip()
            if not text:
                continue
            p = (c.path or "").lower()

            if any(kw in p for kw in ("user", "profile", "persona", "identity", "owner", "about")):
                user_info.append(f"- {text}")
            elif any(kw in p for kw in ("obligation", "task", "todo", "duty", "project", "plan")):
                obligations.append(f"- {text}")
            elif any(kw in p for kw in ("recent", "timeline", "session", "conversation", "chat")):
                recent.append(f"- {text}")
            else:
                safe.append(f"- {text}")

        lines = [
            "## Recommended Topics",
            "",
            "The following topics were selected from memory. Use them only as **background reference**:",
            "",
        ]

        if user_info:
            lines.append("### User Information")
            lines.extend(user_info)
            lines.append("")

        if obligations:
            lines.append("### Things You Can Follow Up On")
            lines.extend(obligations)
            lines.append("")

        if recent:
            lines.append("### Recent Activity")
            lines.extend(recent)
            lines.append("")

        if safe:
            lines.append("### Safe Topics (you can bring up)")
            lines.extend(safe)
            lines.append("")

        if not (user_info or obligations or recent or safe):
            return ""

        lines.append("### Avoid")
        lines.append("- Do not repeat the above information verbatim to the user")
        lines.append('- Do not start messages with "according to memory" or similar phrases')
        lines.append("- If none of the recommended topics are suitable, call memory_search for others")
        lines.append("- If truly nothing to say, reply [SKIP]")

    return "\n".join(lines)


# ── Utility functions ─────────────────────────────────────────────────────────


def _in_active_hours(active_hours: Any) -> bool:
    """Delegate to heartbeat._in_active_hours (lazy import to avoid cycles)."""
    from lightclaw.app.cron.heartbeat import _in_active_hours as _impl

    return _impl(active_hours)


def _in_quiet_hours(quiet_hours: Any) -> bool:
    """Return True if current local time falls within the configured quiet window.

    Reuses the same range logic as _in_active_hours — a quiet window of
    "22:00"–"08:00" correctly handles the overnight wrap-around.
    """
    if not quiet_hours or not hasattr(quiet_hours, "start") or not hasattr(quiet_hours, "end"):
        return False
    # Delegate to heartbeat._in_active_hours: if we are "in" the quiet window, suppress.
    from lightclaw.app.cron.heartbeat import _in_active_hours as _impl

    return _impl(quiet_hours)


async def _run_cron_agent(**kwargs: Any) -> bool:
    """Lazy proxy for agent_task.run_cron_agent (avoids circular imports)."""
    from lightclaw.app.cron.agent_task import run_cron_agent as _impl

    return await _impl(**kwargs)


def _delayed_increment_unanswered(expected_count: int) -> None:
    """Increment unanswered_count only if no user reply occurred."""
    try:
        current = int(read_identity_field("unanswered_count", "0"))
        if current != expected_count:
            logger.info(
                "proactivity: unanswered_count changed %d -> %d during delay, user likely replied, skipping increment",
                expected_count,
                current,
            )
            return
        new_count = current + 1
        write_activity_field("unanswered_count", str(new_count))
        logger.info("proactivity: unanswered_count %d -> %d (delayed)", current, new_count)
    except Exception:
        logger.debug("proactivity: delayed unanswered_count increment failed", exc_info=True)


# ── Engine startup entry point ────────────────────────────────────────────────


def setup_proactive_engine(
    runner: Any,
    channel_manager: Any,
) -> None:
    """
    Configure and start the ProactiveMemoryEngine.

    The engine's _run_loop is the sole scheduling source. Each tick:
    1. engine.fetch_candidates() — Qdrant dual-layer cooldown + vector search
    2. Calls on_evaluate(candidates, mode) callback (implemented in this module)
    3. on_evaluate performs probability gate, active hours check, LLM agent call, dispatch
    4. engine.update_cooldowns() — writes back to Qdrant payload

    Called during the app lifecycle's cron startup phase.
    """
    memory_manager = getattr(runner, "memory_manager", None)
    if memory_manager is None:
        logger.info("proactivity: no memory_manager, engine not available")
        return

    engine = getattr(memory_manager, "proactive_engine", None)
    if engine is None:
        logger.info("proactivity: no proactive_engine, Qdrant not configured")
        return

    # Read interval from lightclaw.json
    cfg = get_proactivity_config()
    if not cfg.enabled:
        logger.info("proactivity: disabled in config")
        return

    # Parse interval (supports "30m", "1h", "600s", "600" formats)
    every_str = getattr(cfg, "every", "600") or "600"
    interval_seconds = _parse_interval(every_str)

    # Configure engine (v2 on_evaluate callback)
    engine.config.enabled = True
    engine.config.check_interval_seconds = interval_seconds
    if hasattr(cfg, "chunk_cooldown_hours"):
        engine.config.chunk_cooldown_hours = cfg.chunk_cooldown_hours
    if hasattr(cfg, "file_cooldown_hours"):
        engine.config.file_cooldown_hours = cfg.file_cooldown_hours
    elif hasattr(cfg, "cooldown_hours"):
        engine.config.file_cooldown_hours = cfg.cooldown_hours
    engine.config.on_evaluate = create_on_evaluate_callback(runner, channel_manager)

    engine.start()
    logger.info(
        "proactivity: engine started (interval=%ds, chunk_cd=%.1fh, file_cd=%.1fh)",
        interval_seconds,
        engine.config.chunk_cooldown_hours,
        engine.config.file_cooldown_hours,
    )


def create_on_evaluate_callback(
    runner: Any,
    channel_manager: Any,
) -> Any:
    """
    Create the on_evaluate callback.

    Signature: async (candidates: list[CandidateResult], mode: str) -> bool
    Returns True if a message was dispatched (engine will then update_cooldowns).
    """

    async def _on_evaluate(candidates: list[Any], mode: str) -> bool:
        cfg = get_proactivity_config()
        if not cfg.enabled:
            logger.info("proactivity on_evaluate: disabled")
            return False

        # 1. Active hours check
        if not _in_active_hours(cfg.active_hours):
            logger.info("proactivity on_evaluate: outside active hours")
            return False

        # 2. Quiet hours check (overrides active hours if both are configured)
        if _in_quiet_hours(cfg.quiet_hours):
            logger.info("proactivity on_evaluate: suppressed (quiet hours)")
            return False

        # 3. Active runs check
        active = getattr(runner, "_active_runs", 0)
        if active > 0:
            logger.info("proactivity on_evaluate: %d active run(s), skip", active)
            return False

        # 4. Probability gate
        activity_level = read_activity_level()
        if activity_level <= 0.0:
            logger.info("proactivity on_evaluate: activity_level=0 (silent mode)")
            return False

        unanswered = int(read_identity_field("unanswered_count", "0"))
        effective_level = activity_level * (0.5**unanswered)

        # [A3] MBTI activity multiplier: extroverts more proactive, introverts less
        try:
            from lightclaw.agent.memory.mbti_proactive_adapter import (
                build_proactive_context_block,
                get_activity_multiplier,
                get_mbti_profile_for_proactive,
            )

            mbti_type, mbti_dims = get_mbti_profile_for_proactive()
            mbti_multiplier = get_activity_multiplier(mbti_dims)
        except Exception:
            logger.warning("on_evaluate: MBTI adapter unavailable, using multiplier=1.0")
            mbti_type, _, mbti_multiplier = "", {}, 1.0

        effective_level *= mbti_multiplier
        # Final formula: activity_level × (0.5^unanswered) × mbti_multiplier [× 0.3 if caring]

        # Caring mode uses a secondary gate (×0.3) — less frequent
        is_caring = mode == "caring"
        if is_caring:
            effective_level *= _CARING_MODE_FACTOR

        if random.random() >= effective_level:
            logger.info(
                "proactivity on_evaluate: dice roll failed "
                "(mode=%s, level=%.2f, mbti_mult=%.2f, effective=%.4f, unanswered=%d)",
                mode,
                activity_level,
                mbti_multiplier,
                effective_level,
                unanswered,
            )
            return False

        # 5. Build prompt = PROACTIVE.md + candidates section (if topic mode)
        query_text = _load_proactive_prompt()
        if not query_text:
            logger.debug("proactivity on_evaluate: no PROACTIVE.md content")
            return False

        # Resolve language before any mode-specific branching
        _engine_cfg = getattr(
            getattr(getattr(runner, "memory_manager", None), "proactive_engine", None),
            "config",
            None,
        )
        lang = getattr(_engine_cfg, "language", "zh") if _engine_cfg else "zh"

        if candidates:
            candidates_section = _format_candidates_section(candidates, lang)
            if candidates_section:
                query_text += "\n\n---\n\n" + candidates_section

        # [B3] MBTI style context block appended to query_text
        if mbti_type:
            context_block = build_proactive_context_block(mbti_type, mode, lang)
            if context_block:
                query_text += "\n\n" + context_block

        # 6. Run LLM agent
        target = (cfg.target or "last").strip().lower()
        logger.info(
            "proactivity on_evaluate: running agent (mode=%s, level=%.2f, target=%s, candidates=%d)",
            mode,
            activity_level,
            target,
            len(candidates),
        )

        # Ensure messages are saved with correct user_id and session_id
        cfg_for_dispatch = load_config()
        ld = cfg_for_dispatch.last_dispatch
        dispatch_user_id = (ld.user_id if ld else "") or "main"
        dispatch_session_id = (ld.session_id if ld else "") or "main"

        try:
            dispatched = await _run_cron_agent(
                query_text=query_text,
                runner=runner,
                channel_manager=channel_manager,
                config=cfg_for_dispatch,
                target=target,
                source="proactivity",
                user_id=dispatch_user_id,
                session_id=dispatch_session_id,
            )
            if not dispatched:
                logger.info("proactivity on_evaluate: agent returned [SKIP]")
                return False

            # 7. Schedule unanswered_count increment (delayed 5min)
            current_count = int(read_identity_field("unanswered_count", "0"))
            asyncio.get_running_loop().call_later(
                _UNANSWERED_DELAY_SECONDS,
                _delayed_increment_unanswered,
                current_count,
            )
            logger.info(
                "proactivity on_evaluate: dispatched (mode=%s), unanswered increment scheduled in %ds",
                mode,
                _UNANSWERED_DELAY_SECONDS,
            )
            return True
        except Exception:
            logger.exception("proactivity on_evaluate: run failed")
            return False

    return _on_evaluate


# ── Compatibility entry point ─────────────────────────────────────────────────


async def run_proactivity_once(
    runner: Any,
    channel_manager: Any,
) -> None:
    """Compat entry: delegates to engine if available, otherwise legacy flow.

    Called by CronManager when APScheduler triggers (fallback path).
    """
    memory_manager = getattr(runner, "memory_manager", None)
    engine = getattr(memory_manager, "proactive_engine", None) if memory_manager else None

    if engine is not None and engine.config.enabled:
        # Engine available — delegate to it
        await engine.evaluate_and_trigger()
    else:
        # No engine — use legacy agent-only flow
        await _run_proactivity_once_legacy(runner, channel_manager)


async def _run_proactivity_once_legacy(
    runner: Any,
    channel_manager: Any,
) -> None:
    """Original agent-only flow (no Qdrant pre-filtering).

    Kept as fallback when ProactiveMemoryEngine is unavailable.
    """
    cfg = get_proactivity_config()
    if not cfg.enabled:
        return

    if not _in_active_hours(cfg.active_hours):
        logger.info("proactivity legacy: outside active hours")
        return

    if _in_quiet_hours(cfg.quiet_hours):
        logger.info("proactivity legacy: suppressed (quiet hours)")
        return

    active = getattr(runner, "_active_runs", 0)
    if active > 0:
        logger.info("proactivity legacy: %d active run(s), skip", active)
        return

    activity_level = read_activity_level()
    if activity_level <= 0.0:
        return

    unanswered = int(read_identity_field("unanswered_count", "0"))
    effective_level = activity_level * (0.5**unanswered)

    if random.random() >= effective_level:
        logger.info(
            "proactivity legacy: dice roll failed (level=%.2f, effective=%.4f)",
            activity_level,
            effective_level,
        )
        return

    query_text = _load_proactive_prompt()
    if not query_text:
        return

    target = (cfg.target or "last").strip().lower()
    logger.info("proactivity legacy: running agent (level=%.2f, target=%s)", activity_level, target)

    # Ensure messages are saved with correct user_id and session_id
    from lightclaw.config import load_config as _load_cfg_for_legacy_dispatch

    cfg_for_dispatch = _load_cfg_for_legacy_dispatch()
    ld = cfg_for_dispatch.last_dispatch
    dispatch_user_id = (ld.user_id if ld else "") or "main"
    dispatch_session_id = (ld.session_id if ld else "") or "main"

    try:
        dispatched = await _run_cron_agent(
            query_text=query_text,
            runner=runner,
            channel_manager=channel_manager,
            config=cfg_for_dispatch,
            target=target,
            source="proactivity",
            user_id=dispatch_user_id,
            session_id=dispatch_session_id,
        )
        if dispatched:
            current_count = int(read_identity_field("unanswered_count", "0"))
            asyncio.get_running_loop().call_later(
                _UNANSWERED_DELAY_SECONDS,
                _delayed_increment_unanswered,
                current_count,
            )
    except Exception:
        logger.exception("proactivity legacy: run failed")


def _parse_interval(every: str) -> float:
    """Parse interval string to seconds. Supports '30m', '1h', '600s', '600'."""
    every = every.strip().lower()
    if every.endswith("m"):
        return float(every[:-1]) * 60
    if every.endswith("h"):
        return float(every[:-1]) * 3600
    if every.endswith("s"):
        return float(every[:-1])
    return float(every)


# ── Daily analysis (unchanged) ────────────────────────────────────────────────


def _load_analysis_prompt_template() -> str:
    """Return built-in analysis prompt template."""
    return (
        "你是 LightClaw 活跃度分析器。\n"
        "当前 activity_level: {current_level}\n"
        "当前 unanswered_count: {unanswered_count}\n"
        "最近的记忆摘要:\n{memory_content}\n\n"
        "请分析用户最近的情绪和互动模式，判断是否需要调整主动联系的频率。\n"
        '输出 JSON: {{"new_level": 0.0~1.0, "reason": "简短原因"}}\n'
        "如果无需调整，new_level 设为当前值。"
    )


async def run_daily_analysis() -> None:
    """Daily LLM analysis of MEMORY.md to adjust activity_level."""
    cfg = get_proactivity_config()
    if not cfg.enabled:
        logger.debug("daily proactivity analysis skipped: disabled")
        return

    current_level = read_activity_level()
    unanswered = read_identity_field("unanswered_count", "0")

    memory_path = WORKING_DIR / "MEMORY.md"
    if not memory_path.is_file():
        logger.debug("daily analysis skipped: no MEMORY.md")
        return
    try:
        memory_content = memory_path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("daily analysis: cannot read MEMORY.md")
        return
    if not memory_content:
        logger.debug("daily analysis skipped: empty MEMORY.md")
        return

    template = _load_analysis_prompt_template()
    prompt = (
        template.replace("{current_level}", str(current_level))
        .replace("{unanswered_count}", str(unanswered))
        .replace("{memory_content}", memory_content[:4000])
    )

    if create_chat_model is None:
        logger.warning("daily analysis: create_chat_model not available")
        return

    try:
        model = create_chat_model()
    except ValueError as exc:
        # Expected when the user hasn't configured any LLM provider yet —
        # log a concise info message instead of a scary traceback.
        logger.info("daily analysis skipped: %s", exc)
        return
    except Exception:
        logger.exception("daily analysis: failed to create chat model")
        return

    try:
        cleaned_prompt = strip_model_special_tokens(prompt)
        response = await asyncio.wait_for(
            model.ainvoke([{"role": "user", "content": cleaned_prompt}]),
            timeout=60,
        )
        raw = response.content if hasattr(response, "content") else str(response)
    except TimeoutError:
        logger.warning("daily analysis: LLM call timed out")
        return
    except Exception:
        logger.exception("daily analysis: LLM call failed")
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[^}]+\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.warning("daily analysis: cannot parse LLM response: %s", raw[:200])
                return
        else:
            logger.warning("daily analysis: no JSON in LLM response: %s", raw[:200])
            return

    new_level = data.get("new_level")
    reason = data.get("reason", "daily analysis")
    if new_level is None or not isinstance(new_level, int | float):
        logger.warning("daily analysis: invalid new_level in response: %s", data)
        return

    new_level = max(0.0, min(1.0, float(new_level)))

    write_activity_field("activity_level", f"{new_level:.2f}")
    write_activity_field("last_adjusted", date.today().isoformat())
    write_activity_field("adjust_reason", reason.replace("\n", " ")[:100])
    logger.info(
        "daily analysis: activity_level %.2f -> %.2f, reason=%s",
        current_level,
        new_level,
        reason[:50],
    )
