import os
import re
from pathlib import Path


# ── Safe environment variable parsers (never raise at import time) ──────────

def _env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    """Parse an integer env var, falling back to *default* on any error."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    if min_value is not None and val < min_value:
        return default
    if max_value is not None and val > max_value:
        return default
    return val


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var, falling back to *default* on any error."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("true", "1", "yes", "y", "on"):
        return True
    if raw in ("false", "0", "no", "n", "off"):
        return False
    return default


# ── Paths and directories ──────────────────────────────────────────────────

# BASE_DIR: root lightclaw directory — holds config, database, auth, env vars.
# This is NOT the agent working area; use WORKING_DIR for md/skills files.
BASE_DIR = Path(os.environ.get("LIGHTCLAW_BASE_DIR", "~/.lightclaw")).expanduser().resolve()

# WORKING_DIR: agent workspace — holds *.md files, skills/, memory/, etc.
# Keep it fixed at BASE_DIR/workspace so all runtime components (session store,
# workspace APIs, dashboard file tree) always point to the same directory.
WORKING_DIR = (BASE_DIR / "workspace").resolve()

JOBS_FILE = os.environ.get("LIGHTCLAW_JOBS_FILE", "jobs.json")

CHATS_FILE = os.environ.get("LIGHTCLAW_CHATS_FILE", "chats.json")

CONFIG_FILE = os.environ.get("LIGHTCLAW_CONFIG_FILE", "lightclaw.json")

PROVIDERS_FILE = os.environ.get("LIGHTCLAW_PROVIDERS_FILE", "providers.json")

HEARTBEAT_FILE = os.environ.get("LIGHTCLAW_HEARTBEAT_FILE", "HEARTBEAT.md")
HEARTBEAT_DEFAULT_EVERY = "30m"
HEARTBEAT_DEFAULT_TARGET = "main"
HEARTBEAT_TARGET_LAST = "last"

# Proactivity defaults
PROACTIVE_FILE = os.environ.get("LIGHTCLAW_PROACTIVE_FILE", "PROACTIVE.md")
PROACTIVE_DEFAULT_EVERY = "30m"
PROACTIVE_DEFAULT_TARGET = "last"
PROACTIVE_DEFAULT_DAILY_ANALYSIS_TIME = "23:00"

# Env key for app log level (used by CLI and app load for reload child).
LOG_LEVEL_ENV = "LIGHTCLAW_LOG_LEVEL"

# When True, expose /docs, /redoc, /openapi.json
# (dev only; keep False in prod).
DOCS_ENABLED = os.environ.get("LIGHTCLAW_OPENAPI_DOCS", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Skills directories
# Unified skills directory (all skills: builtin synced + user-created)
SKILLS_DIR = WORKING_DIR / "skills"
# Legacy directories (kept for backward-compat migration only)
ACTIVE_SKILLS_DIR = WORKING_DIR / "active_skills"
CUSTOMIZED_SKILLS_DIR = WORKING_DIR / "customized_skills"

# Logs directory (one file per day, e.g. ~/.lightclaw/logs/lightclaw.log.YYYY-MM-DD)
LOGS_DIR = BASE_DIR / "logs"

# Memory directory
MEMORY_DIR = WORKING_DIR / "memory"

# Custom channel modules (installed via `lightclaw channels install`); manager
# loads BaseChannel subclasses from here.
CUSTOM_CHANNELS_DIR = WORKING_DIR / "custom_channels"

# Uploads directory (user-uploaded images / files)
UPLOADS_DIR = WORKING_DIR / "uploads"

# Memory compaction configuration
MEMORY_COMPACT_KEEP_RECENT = int(
    os.environ.get("LIGHTCLAW_MEMORY_COMPACT_KEEP_RECENT", "3"),
)

MEMORY_COMPACT_RATIO = float(
    os.environ.get("LIGHTCLAW_MEMORY_COMPACT_RATIO", "0.7"),
)

# TTL for MCP list_tools cache on each connected client.
MCP_TOOLS_CACHE_TTL_SEC = max(0.0, float(os.environ.get("LIGHTCLAW_MCP_TOOLS_CACHE_TTL_SEC", "30")))

# Memory file governance — limits to prevent uncontrolled file proliferation
MEMORY_MAX_FILES = int(os.environ.get("LIGHTCLAW_MEMORY_MAX_FILES", "100"))
MEMORY_MAX_FILE_SIZE = int(os.environ.get("LIGHTCLAW_MEMORY_MAX_FILE_SIZE", str(50 * 1024)))  # 50 KB

# ---------------------------------------------------------------------------
# Daily Talk — global user-message logging for prompt personalization
# ---------------------------------------------------------------------------

# Master switch: set to "false" to disable the daily-talk memory pipeline
# (DAILY_TALK.md).  Enabled by default for conversation continuity tracking.
DAILY_TALK_ENABLED = os.environ.get("LIGHTCLAW_DAILY_TALK_ENABLED", "true").lower() != "false"

# Maximum characters for DAILY_TALK.md before triggering LLM condensation.
DAILY_TALK_MAX_CHARS = int(os.environ.get("LIGHTCLAW_DAILY_TALK_MAX_CHARS", "10000"))
# Target length after LLM condensation of DAILY_TALK.md.
DAILY_TALK_CONDENSE_TARGET = int(os.environ.get("LIGHTCLAW_DAILY_TALK_CONDENSE_TARGET", "7000"))

# Maximum characters for MEMORY.md before triggering LLM compaction.
MEMORY_MD_MAX_CHARS = int(os.environ.get("LIGHTCLAW_MEMORY_MD_MAX_CHARS", "20000"))
# Target length after LLM compaction of MEMORY.md.
MEMORY_MD_COMPACT_TARGET = int(os.environ.get("LIGHTCLAW_MEMORY_MD_COMPACT_TARGET", "14000"))

# ---------------------------------------------------------------------------
# Auto Memory Recall — proactive memory retrieval before each LLM call
# ---------------------------------------------------------------------------

# Master switch: set to "false" to disable automatic memory recall.
# Enabled by default — MEMORY.md is now injected on-demand via FTS+vector search
# rather than being dumped in full every turn.
AUTO_RECALL_ENABLED = os.environ.get("LIGHTCLAW_AUTO_RECALL_ENABLED", "true").lower() != "false"

# Maximum number of memory snippets injected per turn.
AUTO_RECALL_MAX_RESULTS = int(os.environ.get("LIGHTCLAW_AUTO_RECALL_MAX_RESULTS", "3"))

# Minimum cosine similarity score for a memory snippet to be injected.
AUTO_RECALL_MIN_SCORE = float(os.environ.get("LIGHTCLAW_AUTO_RECALL_MIN_SCORE", "0.2"))

# Maximum characters of the user query used as search input.
AUTO_RECALL_MAX_QUERY_LENGTH = int(os.environ.get("LIGHTCLAW_AUTO_RECALL_MAX_QUERY_LENGTH", "500"))

# Timeout in seconds for the auto-recall memory search.
# Prevents cold-start embedding model initialization from blocking the first
# conversation turn for minutes. On timeout the recall step is silently skipped.
AUTO_RECALL_TIMEOUT = float(os.environ.get("LIGHTCLAW_AUTO_RECALL_TIMEOUT", "8.0"))

# ---------------------------------------------------------------------------
# DAILY_TALK.md Injection — inject today's daily talk into system prompt
# ---------------------------------------------------------------------------

# Master switch: set to "false" to disable DAILY_TALK.md injection.
DAILY_TALK_INJECTION_ENABLED = os.environ.get("LIGHTCLAW_DAILY_TALK_INJECTION_ENABLED", "true").lower() != "false"

# Maximum characters from DAILY_TALK.md to inject per turn.
DAILY_TALK_INJECTION_MAX_CHARS = int(os.environ.get("LIGHTCLAW_DAILY_TALK_INJECTION_MAX_CHARS", "10000"))

# When True, DAILY_TALK.md is split into per-message chunks and only the
# top-N most relevant chunks (scored by keyword overlap with the current
# user query) are injected, instead of the full file content.
# Set to "false" to revert to the original full-content injection.
DAILY_TALK_INJECTION_FILTER_ENABLED = (
    os.environ.get("LIGHTCLAW_DAILY_TALK_INJECTION_FILTER_ENABLED", "true").lower() != "false"
)

# Maximum number of DAILY_TALK message chunks to inject per turn when
# relevance filtering is enabled.
DAILY_TALK_INJECTION_MAX_CHUNKS = int(os.environ.get("LIGHTCLAW_DAILY_TALK_INJECTION_MAX_CHUNKS", "5"))

# ---------------------------------------------------------------------------
# MEMORY.md Injection — inject MEMORY.md content into system prompt
# ---------------------------------------------------------------------------

# Master switch: set to "true" to re-enable full MEMORY.md injection into prompts.
# Disabled by default — MEMORY.md content is now retrieved on-demand via
# FTS+vector recall (AUTO_RECALL_ENABLED) instead of being injected in full.
MEMORY_MD_INJECTION_ENABLED = os.environ.get("LIGHTCLAW_MEMORY_MD_INJECTION_ENABLED", "false").lower() != "false"

# Maximum characters from MEMORY.md to inject per turn.
MEMORY_MD_INJECTION_MAX_CHARS = int(os.environ.get("LIGHTCLAW_MEMORY_MD_INJECTION_MAX_CHARS", "20000"))

# ---------------------------------------------------------------------------
# Session / Principal Isolation — recall scope & security defaults
# ---------------------------------------------------------------------------

# Recall scope: "current_session" (safe default) | "owner_memory" | "all_sessions"
RECALL_SCOPE = os.environ.get("LIGHTCLAW_RECALL_SCOPE", "current_session")

# Realtime DAILY_TALK injection: OFF by default (was ON, global, no session filter)
DAILY_TALK_REALTIME_INJECTION_ENABLED = (
    os.environ.get("LIGHTCLAW_DAILY_TALK_REALTIME_INJECTION_ENABLED", "false").lower() != "false"
)

# Realtime raw session search: OFF by default
INCLUDE_RAW_SESSIONS_IN_REALTIME_RECALL = (
    os.environ.get("LIGHTCLAW_INCLUDE_RAW_SESSIONS_IN_REALTIME_RECALL", "false").lower() != "false"
)

# Cross-scope recall gates (all OFF by default)
PRIVATE_TO_GROUP_RECALL_ENABLED = (
    os.environ.get("LIGHTCLAW_PRIVATE_TO_GROUP_RECALL_ENABLED", "false").lower() != "false"
)
GROUP_TO_PRIVATE_RECALL_ENABLED = (
    os.environ.get("LIGHTCLAW_GROUP_TO_PRIVATE_RECALL_ENABLED", "false").lower() != "false"
)
UNKNOWN_USER_MEMORY_PROMOTION_ENABLED = (
    os.environ.get("LIGHTCLAW_UNKNOWN_USER_MEMORY_PROMOTION_ENABLED", "false").lower() != "false"
)
GROUP_MEMORY_ENABLED = os.environ.get("LIGHTCLAW_GROUP_MEMORY_ENABLED", "false").lower() != "false"
SECRET_REALTIME_RECALL_ENABLED = (
    os.environ.get("LIGHTCLAW_SECRET_REALTIME_RECALL_ENABLED", "false").lower() != "false"
)
OLD_METADATA_REALTIME_RECALL_ENABLED = (
    os.environ.get("LIGHTCLAW_OLD_METADATA_REALTIME_RECALL_ENABLED", "false").lower() != "false"
)

# Per-user group sessions: ON by default
GROUP_SESSIONS_PER_USER = (
    os.environ.get("LIGHTCLAW_GROUP_SESSIONS_PER_USER", "true").lower() != "false"
)
GROUP_SHARED_SESSION_ENABLED = (
    os.environ.get("LIGHTCLAW_GROUP_SHARED_SESSION_ENABLED", "false").lower() != "false"
)

# ---------------------------------------------------------------------------
# Context Budget Firewall — hard limits
# ---------------------------------------------------------------------------

# Recall block total char limit
AUTO_RECALL_MAX_TOTAL_CHARS = _env_int("LIGHTCLAW_AUTO_RECALL_MAX_TOTAL_CHARS", 12000)

# Per-candidate char limit in recall
AUTO_RECALL_MAX_CANDIDATE_CHARS = _env_int("LIGHTCLAW_AUTO_RECALL_MAX_CANDIDATE_CHARS", 2000)

# Per session-search-result char limit
AUTO_RECALL_MAX_SESSION_RESULT_CHARS = _env_int(
    "LIGHTCLAW_AUTO_RECALL_MAX_SESSION_RESULT_CHARS", 2000
)

# DAILY_TALK max chars in recall (0 = disabled)
AUTO_RECALL_MAX_DAILY_TALK_CHARS = _env_int("LIGHTCLAW_AUTO_RECALL_MAX_DAILY_TALK_CHARS", 0)

# MEMORY.md max chars in recall
AUTO_RECALL_MAX_MEMORY_CHARS = _env_int("LIGHTCLAW_AUTO_RECALL_MAX_MEMORY_CHARS", 4000)

# Summary block max chars
SUMMARY_BLOCK_MAX_CHARS = _env_int("LIGHTCLAW_SUMMARY_BLOCK_MAX_CHARS", 12000)

# Extra parts total max chars
EXTRA_PARTS_MAX_TOTAL_CHARS = _env_int("LIGHTCLAW_EXTRA_PARTS_MAX_TOTAL_CHARS", 30000)

# Tool message single max chars
TOOL_MESSAGE_SINGLE_MAX_CHARS = _env_int("LIGHTCLAW_TOOL_MESSAGE_SINGLE_MAX_CHARS", 8000)

# Tool message total max chars across all tool messages
TOOL_MESSAGE_TOTAL_MAX_CHARS = _env_int("LIGHTCLAW_TOOL_MESSAGE_TOTAL_MAX_CHARS", 24000)

# Placeholder max length for tool messages replaced by total budget truncation
TOOL_MESSAGE_TRUNCATED_PLACEHOLDER_MAX_CHARS = _env_int(
    "LIGHTCLAW_TOOL_MESSAGE_TRUNCATED_PLACEHOLDER_MAX_CHARS", 800
)

# Final input guard: if True, refuse model call when final tokens > max_input_length
# after all trim rounds. If False, still fail-closed (do NOT send over-budget requests)
# but use a less severe error path.
FINAL_INPUT_GUARD_HARD_FAIL = _env_bool("LIGHTCLAW_FINAL_INPUT_GUARD_HARD_FAIL", True)

# Final input guard: max trim rounds before giving up
FINAL_INPUT_MAX_TRIM_ROUNDS = _env_int("LIGHTCLAW_FINAL_INPUT_MAX_TRIM_ROUNDS", 3)

# Session ID prefix for proactive (cron/memory-triggered) LLM runs.
# These sessions are isolated from the user-visible conversation history.
PROACTIVE_SESSION_PREFIX = "proactive_"

# ---------------------------------------------------------------------------
# Heartbeat runtime guard constants
# ---------------------------------------------------------------------------

# Error fingerprint cooldown: same fingerprint suppressed for this many seconds
ERROR_FINGERPRINT_COOLDOWN_SECONDS = _env_int(
    "LIGHTCLAW_ERROR_FINGERPRINT_COOLDOWN_SECONDS", 1800
)
# Consecutive heartbeat cycles with zero hits before marking a fingerprint resolved
ERROR_FINGERPRINT_RESOLVED_AFTER_CYCLES = _env_int(
    "LIGHTCLAW_ERROR_FINGERPRINT_RESOLVED_AFTER_CYCLES", 2
)
# Max entries in fingerprint state file (LRU eviction)
ERROR_FINGERPRINT_STATE_MAX_ENTRIES = _env_int(
    "LIGHTCLAW_ERROR_FINGERPRINT_STATE_MAX_ENTRIES", 200
)
# Remove fingerprint entries older than this many days
ERROR_FINGERPRINT_STATE_TTL_DAYS = _env_int(
    "LIGHTCLAW_ERROR_FINGERPRINT_STATE_TTL_DAYS", 7
)
# Heartbeat agent turn caps — enforced at runtime via runtime_context.
# When the runner/coordinator reads runtime_context.max_input_length_override /
# max_iters_override, these defaults apply.
HEARTBEAT_MAX_AGENT_ITERS = _env_int("LIGHTCLAW_HEARTBEAT_MAX_AGENT_ITERS", 5, min_value=1)
HEARTBEAT_MAX_INPUT_TOKENS = _env_int("LIGHTCLAW_HEARTBEAT_MAX_INPUT_TOKENS", 8000, min_value=512)
HEARTBEAT_TOTAL_TIMEOUT_SECONDS = _env_int("LIGHTCLAW_HEARTBEAT_TOTAL_TIMEOUT_SECONDS", 75, min_value=15)
HEARTBEAT_MODEL_REQUEST_TIMEOUT_SECONDS = _env_int("LIGHTCLAW_HEARTBEAT_MODEL_REQUEST_TIMEOUT_SECONDS", 60, min_value=10)

# ---------------------------------------------------------------------------
# P3.1 Readonly-worker delegation constants
# ---------------------------------------------------------------------------

DELEGATE_MAX_CONCURRENT_CHILDREN = _env_int(
    "LIGHTCLAW_DELEGATE_MAX_CONCURRENT_CHILDREN", 4, min_value=1, max_value=4
)
DELEGATE_MAX_TASKS = _env_int(
    "LIGHTCLAW_DELEGATE_MAX_TASKS", 5, min_value=1, max_value=5
)
DELEGATE_WORKER_TIMEOUT_SECONDS = _env_int(
    "LIGHTCLAW_DELEGATE_WORKER_TIMEOUT_SECONDS", 120, min_value=30, max_value=120
)
DELEGATE_TOTAL_TIMEOUT_SECONDS = _env_int(
    "LIGHTCLAW_DELEGATE_TOTAL_TIMEOUT_SECONDS", 300, min_value=60, max_value=600
)
DELEGATE_WORKER_MAX_INPUT_TOKENS = _env_int(
    "LIGHTCLAW_DELEGATE_WORKER_MAX_INPUT_TOKENS", 16000, min_value=512
)
DELEGATE_WORKER_MAX_ITERS = _env_int(
    "LIGHTCLAW_DELEGATE_WORKER_MAX_ITERS", 8, min_value=1, max_value=15
)
DELEGATE_RESULT_MAX_CHARS = _env_int(
    "LIGHTCLAW_DELEGATE_RESULT_MAX_CHARS", 12000, min_value=1000
)
DELEGATE_WORKER_SUMMARY_MAX_CHARS = _env_int(
    "LIGHTCLAW_DELEGATE_WORKER_SUMMARY_MAX_CHARS", 3000, min_value=100
)

# ---------------------------------------------------------------------------
# Channel availability — controlled by LIGHTCLAW_ENABLED_CHANNELS env var.
# When unset / empty, all registered channels (built-in + plugins) are
# available. Set to a comma-separated list to restrict, e.g.
#   LIGHTCLAW_ENABLED_CHANNELS=dingtalk,feishu,qq
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Secret Firewall
# ---------------------------------------------------------------------------

# Regex patterns for detecting secrets in text
_SECRET_PATTERNS: list[tuple[str, str]] = [
    (r'(?:api[-_]?key|apikey|API[-_]?KEY)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', "api_key_assignment"),
    (r'(?:bearer|Bearer)\s+[A-Za-z0-9\-._~+/]+={0,2}', "bearer_token"),
    (r'sk-[A-Za-z0-9\-_]{24,}', "openai_key"),
    (r'(?:secret|SECRET)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', "secret_assignment"),
    (r'(?:password|passwd|PASSWORD)\s*[:=]\s*[\'"][^\'"]{4,}[\'"]', "password_assignment"),
    (r'(?:access[-_]?key|ACCESS[-_]?KEY)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', "access_key"),
    (r'(?:private[-_]?key|PRIVATE[-_]?KEY)', "private_key_ref"),
    (r'(?:ssh[-]?key|SSH[-]?KEY)', "ssh_key_ref"),
    (r'(?:webhook|WEBHOOK)\s*[:=]\s*[\'"](?:https?://)?[^\'"]{20,}[\'"]', "webhook_url"),
    (r'(?:cookie|COOKIE)\s*[:=]\s*[\'"][^\'"]{16,}[\'"]', "cookie_assignment"),
    (r'(?:token|TOKEN)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', "token_assignment"),
    (r'SiliconFlow\s*(?:key|余额)', "siliconflow_ref"),
    (r'(?:s3|oss|cos|minio)[-_]?(?:access|secret|key)', "object_storage_key"),
    (r'(?:jwt|JWT)\s*[:=]\s*[\'"][^\'"]{20,}[\'"]', "jwt_token"),
]


def contains_secret(text: str) -> bool:
    """Return True if text matches any known secret/credential pattern."""
    return any(re.search(pat, text, re.IGNORECASE) for pat, _ in _SECRET_PATTERNS)


def is_memory_manager_enabled() -> bool:
    """Return True when the memory manager is not explicitly disabled.

    Controlled by the ``ENABLE_MEMORY_MANAGER`` environment variable.
    Set to ``"false"`` (case-insensitive) to disable; any other value
    (including unset) keeps it enabled.
    """
    return os.environ.get("ENABLE_MEMORY_MANAGER", "").lower() != "false"


def get_available_channels() -> tuple[str, ...]:
    """Return channel keys enabled for this run (built-in + entry point
    lightclaw.channels), filtered by LIGHTCLAW_ENABLED_CHANNELS when set.
    """
    from lightclaw.app.channels.registry import get_channel_registry

    registry = get_channel_registry()
    all_keys = tuple(registry.keys())
    raw = os.environ.get("LIGHTCLAW_ENABLED_CHANNELS", "").strip()
    if not raw:
        return all_keys
    enabled = tuple(ch.strip() for ch in raw.split(",") if ch.strip())
    return tuple(k for k in all_keys if k in enabled) or all_keys


# ---------------------------------------------------------------------------
# One-time migration: BASE_DIR/*.md + BASE_DIR/skills/ -> WORKING_DIR/
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Structured Fact Memory — Mem0-style per-turn fact extraction
# ---------------------------------------------------------------------------

# Master switch: set to "false" to disable fact extraction and injection.
FACT_EXTRACTION_ENABLED = os.environ.get("LIGHTCLAW_FACT_EXTRACTION_ENABLED", "true").lower() != "false"

# Maximum facts stored per user (triggers pruning when exceeded).
FACT_MAX_COUNT = int(os.environ.get("LIGHTCLAW_FACT_MAX_COUNT", "500"))

# Maximum facts extracted per conversation turn.
FACT_MAX_PER_TURN = int(os.environ.get("LIGHTCLAW_FACT_MAX_PER_TURN", "5"))

# Minimum user message length (chars) to trigger extraction.
# Set to 7 to be friendly to Chinese (high info density), while still filtering pure greetings.
FACT_MIN_MESSAGE_LENGTH = int(os.environ.get("LIGHTCLAW_FACT_MIN_MESSAGE_LENGTH", "7"))

# Minimum cosine similarity for matching existing facts.
FACT_SIMILARITY_THRESHOLD = float(os.environ.get("LIGHTCLAW_FACT_SIMILARITY_THRESHOLD", "0.7"))

# Token budget for the <structured-facts> injection block.
FACT_INJECTION_MAX_TOKENS = int(os.environ.get("LIGHTCLAW_FACT_INJECTION_MAX_TOKENS", "500"))


def migrate_to_workspace() -> bool:
    """Migrate legacy md files and skills from BASE_DIR into WORKING_DIR.

    This is a one-time migration that runs when WORKING_DIR does not yet exist
    (i.e. first start after upgrading to workspace-subdirectory layout).

    Moves:
    - ``BASE_DIR/*.md``  →  ``WORKING_DIR/*.md``
    - ``BASE_DIR/skills/``  →  ``WORKING_DIR/skills/``
    - ``BASE_DIR/memory/``  →  ``WORKING_DIR/memory/``
    - ``BASE_DIR/custom_channels/``  →  ``WORKING_DIR/custom_channels/``
    - ``BASE_DIR/uploads/``  →  ``WORKING_DIR/uploads/``

    Returns True if any migration was performed.
    """
    import logging
    import shutil

    logger = logging.getLogger(__name__)

    # Only migrate if WORKING_DIR doesn't exist yet (fresh layout transition)
    if WORKING_DIR.exists():
        return False

    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    migrated = False

    # Move *.md files from BASE_DIR root
    for md_file in BASE_DIR.glob("*.md"):
        target = WORKING_DIR / md_file.name
        try:
            shutil.move(str(md_file), str(target))
            logger.info("Migrated md file: %s -> %s", md_file, target)
            migrated = True
        except Exception as exc:
            logger.warning("Failed to migrate %s: %s", md_file, exc)

    # Move directories: skills, memory, custom_channels, uploads
    for dir_name in ("skills", "memory", "custom_channels", "uploads"):
        src = BASE_DIR / dir_name
        dst = WORKING_DIR / dir_name
        if src.is_dir() and not dst.exists():
            try:
                shutil.move(str(src), str(dst))
                logger.info("Migrated directory: %s -> %s", src, dst)
                migrated = True
            except Exception as exc:
                logger.warning("Failed to migrate directory %s: %s", src, exc)

    if migrated:
        logger.info("Migration complete: agent workspace files moved to %s", WORKING_DIR)
    return migrated
