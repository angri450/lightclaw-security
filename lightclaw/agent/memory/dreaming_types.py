"""Type definitions for the 3-phase dreaming pipeline (Light/REM/Deep).

DreamingConfig is NOT defined here — import from lightclaw.config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DreamCandidate:
    """One candidate extracted during Light Phase."""

    id: str  # "{source_file}:{normalized_hash[:8]}"
    source_file: str  # e.g. "DAILY_TALK.md" or "memory/2026-05-02.md"
    source_date: str  # "2026-05-02"
    raw_text: str  # original fragment from source
    normalized_text: str  # stripped + normalized, basis for evidence_hash
    summary: str  # LLM-compressed summary
    entry_type: str  # preference/constraint/decision/todo/config/fact_*/...
    persistence: str  # permanent/stable/transient/ephemeral
    keywords: list[str] = field(default_factory=list)
    confidence: float = 0.0  # Light Phase LLM confidence
    has_secret: bool = False
    evidence_hash: str = ""  # SHA256(normalized_text)[:16]
    line_range: str = ""  # optional source location marker, e.g. "L12-L15"
    # v2: Speaker/Principal attribution for promotion security
    source_channel: str = ""  # channel of origin (dashboard, yuanbao, weixin, etc.)
    source_chat_type: str = ""  # private / group / dashboard / cron
    source_chat_id: str = ""  # group chat ID if from group
    speaker_id: str = ""  # sender_id hash (not raw sender_id)
    principal_id: str = ""  # resolved principal
    trust_level: str = ""  # owner / trusted / group_member / unknown / system


@dataclass
class PhaseSignal:
    """Signal produced by REM Phase for a DreamCandidate."""

    candidate_id: str  # maps to DreamCandidate.id
    phase: str = "rem"
    signal_type: str = ""  # repeated_across_days | conflicts_with_existing_memory
    # | strengthens_existing_memory | user_profile_update_needed
    # | tool_rule_update_needed | agent_rule_update_needed
    # | proactivity_update_needed | no_action
    duplicate_of: str | None = None  # reference text of existing entry in MEMORY.md
    conflict_with: str | None = None  # reference text of conflicting entry
    pattern_category: str | None = None  # recurring_topic | new_preference | new_constraint | tool_usage
    cross_ref_count: int = 0  # number of distinct days with cross-references
    confidence: float = 0.0  # REM Phase LLM confidence
    should_promote: bool = False  # REM Phase recommendation to promote to MEMORY.md
    score_delta: float = 0.0  # REM Phase bonus score
    reason: str = ""  # human-readable reason
    metadata: dict | None = None  # raw metadata from LLM (duplicate_of, conflict_with, pattern_category, cross_ref_count, target_file)
    parse_error: str | None = None  # set when JSON parsing required fallback extraction


@dataclass
class DreamingResult:
    """Result of a full dreaming pipeline run."""

    date: str
    dry_run: bool
    started_at: str
    ended_at: str
    light_candidates_total: int
    light_candidates_kept: int
    light_secrets_filtered: int
    light_ephemeral_filtered: int
    rem_signals: list[PhaseSignal] = field(default_factory=list)
    promoted_count: int = 0
    promoted_entries: list[dict[str, Any]] = field(default_factory=list)
    pruned_count: int = 0
    pruned_entries: list[dict[str, Any]] = field(default_factory=list)
    updated_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report_text: str = ""
    duration_seconds: float = 0.0
