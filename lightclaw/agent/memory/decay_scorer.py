"""Hybrid regex+LLM memory decay scorer.

Inspired by OpenClaw dreaming Deep phase.  Regex handles the fast path
(recency / richness / persistence from parsed metadata, cross-reference
counting from daily digests).  LLM is only called for ambiguous entries
that regex cannot confidently classify — batched into a single call.

Weights mirror the dreaming scoring model:
  Recency        0.30  —  how recently was this fact confirmed?
  Frequency      0.25  —  how often does it appear across sources?
  Consolidation  0.20  —  how many unique days does it span?
  Persistence    0.15  —  what tier of longevity does this belong to?
  Richness       0.10  —  information density of the content

Each dimension is normalised 0–1, then weighted-sum → final_score.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from lightclaw.agent.memory.memory_parser import (
    MemoryEntry,
    MemoryParser,
    MemoryParseResult,
    compute_richness,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Scoring weights  (sum == 1.0)
# ═══════════════════════════════════════════════════════════════════

_WEIGHT_RECENCY = 0.30
_WEIGHT_FREQUENCY = 0.25
_WEIGHT_CONSOLIDATION = 0.20
_WEIGHT_PERSISTENCE = 0.15
_WEIGHT_RICHNESS = 0.10

# ═══════════════════════════════════════════════════════════════════
# Persistence → numeric score (regex fast path)
# ═══════════════════════════════════════════════════════════════════

_PERSISTENCE_SCORE_MAP: dict[str, float] = {
    "permanent": 0.95,
    "stable": 0.75,
    "transient": 0.40,
    "ephemeral": 0.10,
}

# Default when type is "meta" / boilerplate — these should decay fast
_META_PERSISTENCE_SCORE = 0.05

# ═══════════════════════════════════════════════════════════════════
# Recency decay parameters
# ═══════════════════════════════════════════════════════════════════

# Half-life in days by persistence tier — shorter half-life = faster decay
# ephemeral facts (weather, balances, versions) decay in ~3 days
# transient facts (current state, todos) decay in ~10 days
# stable facts (config, decisions) decay in ~21 days
# permanent facts (identity, core preferences) decay in ~60 days
_HALF_LIFE_MAP: dict[str, float] = {
    "ephemeral": 3.0,
    "transient": 10.0,
    "stable": 21.0,
    "permanent": 60.0,
}
_HALF_LIFE_DEFAULT = 14.0
# Maximum recency score — entries without dates get this floor
_RECENCY_FLOOR = 0.15
# Section-level date extraction for entries that lack their own dates
_SECTION_DATE_RE = re.compile(r"Distilled\s+on\s+(\d{4}-\d{2}-\d{2})")

# ═══════════════════════════════════════════════════════════════════
# Cross-reference matchers
# ═══════════════════════════════════════════════════════════════════

# When scanning daily digest files, extract what keywords to match against
_KEYWORD_RE = re.compile(r"[一-鿿]{2,}|\w{3,}")

# Minimum keyword overlap for a cross-reference hit
_MIN_KEYWORD_OVERLAP = 2

# ── Vector cross-reference ────────────────────────────────────────

# Cosine similarity threshold for considering a digest chunk "about" a
# memory entry.  0.7 is conservative — only clear semantic matches count.
_VECTOR_SIM_THRESHOLD = 0.70

# Maximum digest chunks to embed per day (limits compute cost)
_MAX_DIGEST_CHUNKS_PER_DAY = 20

# Minimum chunk length (chars) to consider embedding
_MIN_CHUNK_LENGTH = 20

# Chunk split: split daily digest by double-newline (paragraph boundaries)
_CHUNK_SPLIT_RE = re.compile(r"\n\n+")


# ═══════════════════════════════════════════════════════════════════
# LLM fallback: batch classify ambiguous entries
# ═══════════════════════════════════════════════════════════════════

_AMBIGUITY_PROMPT = """\
你是记忆分类助手。以下是从 MEMORY.md 中解析出的条目，每条都有一个初步分类，
但有些条目分类不准确（标记为 unknown 或可能误判）。

请对每条给出正确的分类和持久性等级。

分类选项（选一个）：
  preference  — 用户偏好、风格、习惯
  constraint  — 禁止事项、硬性要求
  decision    — 已做出的决定、选择
  todo        — 待办事项、未完成的任务
  config      — 工具配置、命令、API key、安装信息
  fact        — 客观事实（版本号、天气、状态）
  meta        — 模板说明、格式参考、无实际信息

持久性选项（选一个）：
  permanent   — 几乎不变（身份、名字、核心人格）
  stable      — 长期有效（偏好、配置、决策）
  transient   — 中等有效期（当前状态、短期任务）
  ephemeral   — 很快过时（天气、版本号、余额）

请严格按以下 JSON 格式输出，一个条目一行：
[ENTRY_ID] type=<分类> pers=<持久性>
...

条目列表：
{entries}"""


class DecayScorer:
    """Score MEMORY.md entries via regex heuristics + optional LLM batch.

    When ``embedding_fn`` is provided, cross-referencing uses vector
    similarity (semantic match) instead of regex keyword overlap.
    """

    def __init__(
        self,
        working_dir: str | Path,
        *,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self._root = Path(working_dir)
        self._parser = MemoryParser(working_dir)
        self._embedding_fn = embedding_fn
        self._daily_digest_cache: dict[str, str] | None = None
        # vector-mode cache: {day_key: [(chunk_text, embedding_vector), ...]}
        self._digest_chunk_embeddings: dict[str, list[tuple[str, list[float]]]] | None = None
        self._section_dates: dict[str, datetime] = {}

    # ── public API ──────────────────────────────────────────────────

    async def score(
        self,
        *,
        invoke_llm: Callable[[str, str], Awaitable[str]] | None = None,
        skip_llm: bool = False,
    ) -> MemoryParseResult:
        """Parse and score all entries.

        Set ``skip_llm=True`` to avoid LLM calls entirely (pure regex scoring).
        """
        result = self._parser.parse()
        if not result.entries:
            return result

        self._precompute_section_dates(result)
        await self._precompute_cross_references(result)

        # 1. Compute fast-path scores (regex only)
        for entry in result.entries:
            self._score_recency(entry)
            self._score_frequency(entry)
            self._score_consolidation(entry)
            self._score_persistence(entry)
            self._score_richness(entry)

        # 2. LLM fallback for ambiguous entries
        ambiguous = [e for e in result.entries if e.entry_type in ("unknown",)]
        if ambiguous and invoke_llm is not None and not skip_llm:
            try:
                await self._llm_reclassify(ambiguous, invoke_llm)
                # Re-score persistence now that LLM has refined types
                for ae in ambiguous:
                    self._score_persistence(ae)
            except Exception:
                logger.exception("LLM reclassify failed, keeping regex types")

        # 3. Weighted sum
        for entry in result.entries:
            entry.final_score = round(
                _WEIGHT_RECENCY * entry.recency_score
                + _WEIGHT_FREQUENCY * entry.frequency_score
                + _WEIGHT_CONSOLIDATION * entry.consolidation_score
                + _WEIGHT_PERSISTENCE * entry.persistence_score
                + _WEIGHT_RICHNESS * entry.richness_score,
                3,
            )

        return result

    def score_sync(self) -> MemoryParseResult:
        """Synchronous scoring: regex-only, no LLM, no embedding."""
        result = self._parser.parse()
        if not result.entries:
            return result

        self._precompute_section_dates(result)
        # Load daily digests and use keyword-only cross-referencing
        if self._daily_digest_cache is None:
            self._daily_digest_cache = self._load_daily_digests()
        if self._daily_digest_cache:
            self._keyword_cross_references(result)

        for entry in result.entries:
            self._score_recency(entry)
            self._score_frequency(entry)
            self._score_consolidation(entry)
            self._score_persistence(entry)
            self._score_richness(entry)
            entry.final_score = round(
                _WEIGHT_RECENCY * entry.recency_score
                + _WEIGHT_FREQUENCY * entry.frequency_score
                + _WEIGHT_CONSOLIDATION * entry.consolidation_score
                + _WEIGHT_PERSISTENCE * entry.persistence_score
                + _WEIGHT_RICHNESS * entry.richness_score,
                3,
            )
        return result

    # ── fast-path scorers ───────────────────────────────────────────

    def _score_recency(self, entry: MemoryEntry) -> None:
        """Compute recency score via exponential decay.

        Half-life is persistence-adjusted: ephemeral facts decay in 3 days,
        stable facts in 21 days.  Entries without dates inherit their
        section's date; entries with no date at all get ``_RECENCY_FLOOR``.
        """
        days = entry.days_since_newest
        if days is None:
            sd = self._section_dates.get(entry.section)
            if sd is not None:
                days = (datetime.now(timezone.utc) - sd).days
            else:
                entry.recency_score = _RECENCY_FLOOR
                return

        if days < 0:
            days = 0

        half_life = _HALF_LIFE_MAP.get(entry.persistence, _HALF_LIFE_DEFAULT)
        entry.recency_score = round(2.0 ** (-days / half_life), 3)

    def _score_frequency(self, entry: MemoryEntry) -> None:
        """Count keyword overlap hits in daily digest files.

        Saturates at 5 matches → score 1.0.
        """
        if entry.ref_count >= 5:
            entry.frequency_score = 1.0
        else:
            entry.frequency_score = round(entry.ref_count / 5.0, 3)

    def _score_consolidation(self, entry: MemoryEntry) -> None:
        """Score based on unique days this entry appears across.

        Saturates at 3 unique days → score 1.0.
        """
        days = len(entry.ref_days)
        if days >= 3:
            entry.consolidation_score = 1.0
        else:
            entry.consolidation_score = round(days / 3.0, 3)

    def _score_persistence(self, entry: MemoryEntry) -> None:
        """Map persistence tier and type to numeric score."""
        if entry.is_boilerplate or entry.entry_type == "meta":
            entry.persistence_score = _META_PERSISTENCE_SCORE
            return
        score = _PERSISTENCE_SCORE_MAP.get(entry.persistence)
        if score is not None:
            entry.persistence_score = score
            return
        # Fallback: map entry_type to reasonable persistence
        type_map = {
            "preference": 0.80,
            "constraint": 0.85,
            "decision": 0.70,
            "config": 0.65,
            "todo": 0.30,
            "fact_version": 0.10,
            "fact_weather": 0.05,
            "fact_transient": 0.25,
        }
        entry.persistence_score = type_map.get(entry.entry_type, 0.40)

    @staticmethod
    def _score_richness(entry: MemoryEntry) -> None:
        """Compute information density score."""
        if entry.is_boilerplate or entry.entry_type == "meta":
            entry.richness_score = 0.0
            return
        entry.richness_score = compute_richness(entry.raw_text)

    # ── precomputations ─────────────────────────────────────────────

    def _precompute_section_dates(self, result: MemoryParseResult) -> None:
        """Extract dates from section headers (e.g. 'Distilled on 2026-05-02')."""
        self._section_dates.clear()
        sections_seen = {e.section for e in result.entries}
        for sec in sections_seen:
            m = _SECTION_DATE_RE.search(sec)
            if m:
                try:
                    self._section_dates[sec] = datetime.strptime(
                        m.group(1), "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

    async def _precompute_cross_references(
        self, result: MemoryParseResult,
    ) -> None:
        """Count semantic hits in daily digest files for each entry.

        When ``embedding_fn`` is available, uses vector similarity.
        Otherwise falls back to regex keyword overlap.
        """
        if self._daily_digest_cache is None:
            self._daily_digest_cache = self._load_daily_digests()

        if not self._daily_digest_cache:
            return

        if self._embedding_fn is not None:
            await self._vector_cross_references(result)
        else:
            self._keyword_cross_references(result)

    def _load_daily_digests(self) -> dict[str, str]:
        """Load daily digest files from memory/ directory.

        Returns {date_key: lowercase_text} mapping.
        Skips MEMORY.md, DAILY_TALK.md, and binary files.
        """
        memory_dir = self._root / "memory"
        if not memory_dir.is_dir():
            return {}

        digests: dict[str, str] = {}
        skip_names = {"MEMORY.md", "DAILY_TALK.md"}
        skip_suffixes = {".db", ".db-shm", ".db-wal", ".tmp"}

        for f in memory_dir.iterdir():
            if not f.is_file():
                continue
            if f.name in skip_names:
                continue
            if any(f.name.endswith(s) for s in skip_suffixes):
                continue
            try:
                text = f.read_text(encoding="utf-8").lower()
                # Use stem (e.g. "2026-05-02") as date key
                day_key = f.stem
                digests[day_key] = text
            except Exception:
                logger.debug("Failed to read digest %s", f)

        return digests

    def _keyword_cross_references(self, result: MemoryParseResult) -> None:
        """Regex keyword overlap for frequency/consolidation (no embedding)."""
        for entry in result.entries:
            if entry.is_boilerplate or entry.entry_type == "meta":
                continue
            keywords = _KEYWORD_RE.findall(entry.raw_text.lower())
            if len(keywords) < 2:
                continue
            hits = 0
            hit_days: set[str] = set()
            for day_key, digest_text in self._daily_digest_cache.items():  # type: ignore[union-attr]
                overlap = sum(1 for kw in keywords if kw in digest_text)
                if overlap >= _MIN_KEYWORD_OVERLAP:
                    hits += 1
                    hit_days.add(day_key)
            entry.ref_count = hits
            entry.ref_days = hit_days

    async def _vector_cross_references(self, result: MemoryParseResult) -> None:
        """Vector similarity for frequency/consolidation scoring.

        1. Embed all memory entries (batch)
        2. Embed all digest chunks per day (batch per day)
        3. For each entry, count days where max cosine similarity > threshold
        """
        assert self._embedding_fn is not None
        assert self._daily_digest_cache is not None

        if self._digest_chunk_embeddings is None:
            self._digest_chunk_embeddings = {}
            for day_key, digest_text in self._daily_digest_cache.items():
                chunks = self._chunk_digest(digest_text)
                if not chunks:
                    self._digest_chunk_embeddings[day_key] = []
                    continue
                try:
                    vectors = await self._embedding_fn(chunks)
                    self._digest_chunk_embeddings[day_key] = list(zip(chunks, vectors))
                except Exception:
                    logger.exception("Failed to embed digest chunks for %s", day_key)
                    self._digest_chunk_embeddings[day_key] = []

        active_entries = [
            e for e in result.entries
            if not e.is_boilerplate and e.entry_type != "meta"
        ]
        if not active_entries:
            return

        entry_texts = [e.raw_text for e in active_entries]
        try:
            entry_vectors = await self._embedding_fn(entry_texts)
        except Exception:
            logger.exception("Failed to embed memory entries, falling back to keyword")
            self._keyword_cross_references(result)
            return

        # Compute keyword hits as floor for blending
        kw_ref_counts: dict[int, int] = {}
        kw_ref_days: dict[int, set[str]] = {}
        for i, entry in enumerate(active_entries):
            keywords = _KEYWORD_RE.findall(entry.raw_text.lower())
            if len(keywords) < 2:
                continue
            hits = 0
            hit_days: set[str] = set()
            for day_key in self._daily_digest_cache:  # type: ignore[union-attr]
                overlap = sum(1 for kw in keywords if kw in self._daily_digest_cache[day_key])
                if overlap >= _MIN_KEYWORD_OVERLAP:
                    hits += 1
                    hit_days.add(day_key)
            kw_ref_counts[i] = hits
            kw_ref_days[i] = hit_days

        for i, entry in enumerate(active_entries):
            ev = entry_vectors[i]
            vec_hits = 0
            vec_hit_days: set[str] = set()
            for day_key, chunk_vecs in self._digest_chunk_embeddings.items():
                best_sim = 0.0
                for _chunk_text, cv in chunk_vecs:
                    sim = _cosine_similarity(ev, cv)
                    if sim > best_sim:
                        best_sim = sim
                if best_sim >= _VECTOR_SIM_THRESHOLD:
                    vec_hits += 1
                    vec_hit_days.add(day_key)

            # Blend: max of vector and keyword for each dimension
            kw_hits = kw_ref_counts.get(i, 0)
            kw_days = kw_ref_days.get(i, set())
            entry.ref_count = max(vec_hits, kw_hits)
            entry.ref_days = vec_hit_days | kw_days

    @staticmethod
    def _chunk_digest(text: str) -> list[str]:
        """Split daily digest into embeddable chunks by paragraph.

        Returns at most ``_MAX_DIGEST_CHUNKS_PER_DAY`` chunks,
        each at least ``_MIN_CHUNK_LENGTH`` chars.
        """
        raw = _CHUNK_SPLIT_RE.split(text)
        chunks: list[str] = []
        for ch in raw:
            stripped = ch.strip()
            if len(stripped) >= _MIN_CHUNK_LENGTH:
                chunks.append(stripped)
            if len(chunks) >= _MAX_DIGEST_CHUNKS_PER_DAY:
                break
        return chunks

    # ── LLM fallback ────────────────────────────────────────────────

    async def _llm_reclassify(
        self,
        entries: list[MemoryEntry],
        invoke_llm: Callable[[str, str], Awaitable[str]],
    ) -> None:
        """Batch-reclassify ambiguous entries via a single LLM call."""
        if not entries:
            return

        lines = []
        for i, e in enumerate(entries):
            lines.append(
                f"[{i}] type={e.entry_type} pers={e.persistence} "
                f"section={e.section} subsection={e.subsection} "
                f"text={e.raw_text[:200]}"
            )
        user_msg = "\n".join(lines)
        system = _AMBIGUITY_PROMPT.format(entries=user_msg)

        response = await invoke_llm(system, user_msg)
        self._parse_llm_response(response, entries)

    @staticmethod
    def _parse_llm_response(
        response: str,
        entries: list[MemoryEntry],
    ) -> None:
        """Parse LLM batch classification output and update entries."""
        pattern = re.compile(
            r"\[(\d+)\]\s*type=(\w+)\s+pers=(\w+)", re.IGNORECASE
        )
        valid_types = {
            "preference", "constraint", "decision", "todo",
            "config", "fact", "meta",
        }
        valid_pers = {"permanent", "stable", "transient", "ephemeral"}

        for m in pattern.finditer(response):
            idx = int(m.group(1))
            etype = m.group(2).lower()
            pers = m.group(3).lower()
            if idx < len(entries) and etype in valid_types and pers in valid_pers:
                entries[idx].entry_type = etype
                entries[idx].persistence = pers


# ═══════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════

DEFAULT_PRUNE_THRESHOLD = 0.40


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
"""Entries below this final_score are candidates for pruning."""


def score_report(result: MemoryParseResult) -> str:
    """Format a human-readable scoring report."""
    lines = [
        f"Memory Decay Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Entries scored: {result.entry_count}",
        f"{'='*70}",
        "",
    ]

    # Sort by score ascending (worst first)
    sorted_entries = sorted(result.entries, key=lambda e: e.final_score)

    lines.append(
        f"{'Score':>6}  {'Pers':>6}  {'Rec':>6}  {'Frq':>6}  "
        f"{'Con':>6}  {'Rch':>6}  {'Type':12}  Content"
    )
    lines.append("-" * 100)

    for e in sorted_entries:
        flag = " [PRUNE]" if e.final_score < DEFAULT_PRUNE_THRESHOLD else ""
        lines.append(
            f"{e.final_score:>6.3f}  "
            f"{e.persistence_score:>6.3f}  "
            f"{e.recency_score:>6.3f}  "
            f"{e.frequency_score:>6.3f}  "
            f"{e.consolidation_score:>6.3f}  "
            f"{e.richness_score:>6.3f}  "
            f"{e.entry_type:<12}  "
            f"{e.raw_text[:60]}{flag}"
        )

    # Summary
    below = sum(1 for e in result.entries if e.final_score < DEFAULT_PRUNE_THRESHOLD)
    lines.append("")
    lines.append(
        f"Prune candidates: {below}/{result.entry_count} "
        f"({100*below/result.entry_count:.1f}%)  "
        f"(threshold={DEFAULT_PRUNE_THRESHOLD})"
    )

    # Distribution
    dist = Counter(
        "prune" if e.final_score < DEFAULT_PRUNE_THRESHOLD else "keep"
        for e in result.entries
    )
    lines.append(f"Keep/Prune: {dict(dist)}")

    return "\n".join(lines)
