"""Qdrant-backed memory searcher supporting hybrid (dense + BM25) and FTS-only modes.

Search pipeline:
  query
    ├─ [FTS-only, no embedding_fn]
    │    └─ extractKeywords(query) → FTS5 MATCH → deduplicate → return
    └─ [Hybrid, with embedding_fn]
         ├─ 1. dense:   embed(query) → Qdrant.search(limit=candidates)
         ├─ 2. BM25:    buildFtsQuery(query) → SQLite FTS5 MATCH → bm25RankToScore
         ├─ 3. merge:   score = 0.7 * vectorScore + 0.3 * textScore
         ├─ 4. temporal decay (optional, off by default)
         ├─ 5. sort + minScore filter
         └─ 6. MMR rerank (optional, off by default)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import sqlite3
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightclaw.agent.memory.query_expansion import build_fts_query, expand_query_for_fts, extract_keywords
from lightclaw.agent.memory.temporal_resolver import DateRange

if TYPE_CHECKING:
    from qdrant_client.models import Filter

    from lightclaw.agent.memory.qdrant_indexer import QdrantMemoryIndexer

logger = logging.getLogger(__name__)

# ── Default parameters ────────────────────────────────────────────────────────
VECTOR_WEIGHT = 0.7
TEXT_WEIGHT = 0.3
CANDIDATE_MULTIPLIER = 4  # candidates = maxResults * CANDIDATE_MULTIPLIER
MAX_CANDIDATES = 200
SNIPPET_MAX_CHARS = 700
HALF_LIFE_DAYS = 90.0
MMR_LAMBDA = 0.7
DAY_SECONDS = 86400.0

# Absolute minimum score below which results are considered pure noise.
# When all candidates fall below this floor, search returns empty rather
# than injecting irrelevant content into the LLM context.
ABSOLUTE_SCORE_FLOOR = float(os.environ.get("MEMORY_ABSOLUTE_SCORE_FLOOR", "0.15"))

# Score multiplier applied to results whose path date falls within a soft
# DateRange (e.g. "最近", "recently").  Hard ranges use filtering instead.
DATE_MATCH_BOOST = 1.5

# Regex matching memory/YYYY-MM-DD.md paths
_DATED_MEMORY_PATH_RE = re.compile(r"(?:^|/)memory/(\d{4})-(\d{2})-(\d{2})\.md$")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search result entry."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: str
    vector_score: float = 0.0
    text_score: float = 0.0
    importance: float = 0.5  # from payload, default neutral
    access_count: int = 0  # from payload, search-hit accumulator


@dataclass
class TemporalDecayConfig:
    """Configuration for time-based score decay."""

    enabled: bool = True
    half_life_days: float = 90.0  # 90 days: gentler than 30, avoids over-aggressive decay


@dataclass
class MMRConfig:
    """Configuration for Maximal Marginal Relevance reranking."""

    enabled: bool = False
    lambda_: float = MMR_LAMBDA


@dataclass
class SearchConfig:
    """Aggregated search parameters."""

    max_results: int = 5
    min_score: float = 0.3
    vector_weight: float = VECTOR_WEIGHT
    text_weight: float = TEXT_WEIGHT
    temporal_decay: TemporalDecayConfig = field(default_factory=TemporalDecayConfig)
    mmr: MMRConfig = field(default_factory=MMRConfig)
    date_range: DateRange | None = None

    # v2: scope filter
    scope_filter: str = ""  # "current_session" | "owner_memory" | "user_memory" | "" (no filter)
    current_session_id: str = ""
    current_principal_id: str = ""
    actor_trust_level: str = ""
    exclude_contains_secret: bool = True
    exclude_is_tool_output: bool = True
    exclude_is_raw_session: bool = True


# ── Utility functions ─────────────────────────────────────────────────────────


def bm25_rank_to_score(rank: float) -> float:
    """Convert SQLite FTS5 bm25() rank to a [0, 1] score.

    FTS5 bm25() returns negative values (more negative = more relevant)::

        rank < 0:  relevance = -rank,  score = relevance / (1 + relevance)
        rank >= 0: score = 1 / (1 + rank)
    """
    if not math.isfinite(rank):
        return 1 / (1 + 999)
    if rank < 0:
        relevance = -rank
        return relevance / (1 + relevance)
    return 1 / (1 + rank)


def _parse_memory_date_from_path(file_path: str) -> datetime | None:
    """Extract a date from a ``memory/YYYY-MM-DD.md`` path, or return None."""
    normalized = file_path.replace("\\", "/").lstrip("./")
    match = _DATED_MEMORY_PATH_RE.search(normalized)
    if not match:
        return None
    try:
        year, month, day = int(match[1]), int(match[2]), int(match[3])
        dt = datetime(year, month, day, tzinfo=UTC)
        if dt.year != year or dt.month != month or dt.day != day:
            return None
        return dt
    except (ValueError, OverflowError):
        return None


def _is_evergreen_memory_path(file_path: str) -> bool:
    """Return True if the path is evergreen (exempt from temporal decay).

    Evergreen paths: MEMORY.md, memory.md, and non-dated files under memory/.
    """
    normalized = file_path.replace("\\", "/").lstrip("./")
    if normalized in ("MEMORY.md", "memory.md"):
        return True
    if not normalized.startswith("memory/"):
        return False
    return not bool(_DATED_MEMORY_PATH_RE.search(normalized))


def _calculate_temporal_decay_multiplier(age_days: float, half_life_days: float) -> float:
    """Compute the exponential decay multiplier for a given age.

    Uses the formula::

        lambda_ = ln(2) / half_life_days
        multiplier = exp(-lambda_ * max(0, age_days))
    """
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        return 1.0
    lambda_ = math.log(2) / half_life_days
    clamped_age = max(0.0, age_days)
    if not math.isfinite(clamped_age):
        return 1.0
    return math.exp(-lambda_ * clamped_age)


def _get_file_mtime(abs_path: str) -> datetime | None:
    """Return the last-modified time of a file, or None on error."""
    try:
        mtime = Path(abs_path).stat().st_mtime
        if not math.isfinite(mtime):
            return None
        return datetime.fromtimestamp(mtime, tz=UTC)
    except OSError:
        return None


def _apply_temporal_decay(
    results: list[SearchResult],
    config: TemporalDecayConfig,
    working_dir: str | None = None,
    now: datetime | None = None,
) -> list[SearchResult]:
    """Apply time-based score decay to search results."""
    if not config.enabled:
        return results

    now_dt = now or datetime.now(UTC)
    decayed: list[SearchResult] = []

    for r in results:
        # 1. Try to extract date from path
        ts = _parse_memory_date_from_path(r.path)

        # 2. Evergreen paths are not decayed
        if ts is None and r.source == "memory" and _is_evergreen_memory_path(r.path):
            decayed.append(r)
            continue

        # 3. Fallback: use file mtime
        if ts is None and working_dir:
            abs_path = str(Path(working_dir) / r.path)
            ts = _get_file_mtime(abs_path)

        if ts is None:
            decayed.append(r)
            continue

        age_days = max(0.0, (now_dt - ts).total_seconds()) / DAY_SECONDS
        multiplier = _calculate_temporal_decay_multiplier(age_days, config.half_life_days)
        new_r = SearchResult(
            chunk_id=r.chunk_id,
            path=r.path,
            start_line=r.start_line,
            end_line=r.end_line,
            score=r.score * multiplier,
            snippet=r.snippet,
            source=r.source,
            vector_score=r.vector_score,
            text_score=r.text_score,
            importance=r.importance,
            access_count=r.access_count,
        )
        decayed.append(new_r)

    return decayed


# ── Date filtering / boosting ─────────────────────────────────────────────────


def _build_date_path_filter(date_range: DateRange) -> Filter | None:
    """Build a Qdrant payload filter matching memory file paths within a date range.

    Returns ``None`` for ranges spanning more than 31 days (too many OR
    conditions) — the caller should fall back to Python post-filtering.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    day_count = (date_range.end - date_range.start).days + 1
    if day_count > 31:
        return None  # Fall back to post-filtering

    path_values: list[str] = []
    current = date_range.start
    while current <= date_range.end:
        path_values.append(f"memory/{current.isoformat()}.md")
        current += timedelta(days=1)

    if len(path_values) == 1:
        return Filter(must=[FieldCondition(key="path", match=MatchValue(value=path_values[0]))])

    return Filter(
        should=[FieldCondition(key="path", match=MatchValue(value=p)) for p in path_values],
    )


def _path_in_date_range(path: str, date_range: DateRange) -> bool:
    """Return True if *path* is a dated memory file within *date_range*, or non-dated."""
    path_dt = _parse_memory_date_from_path(path)
    if path_dt is None:
        # Non-dated files (MEMORY.md, etc.) always pass.
        return True
    return date_range.start <= path_dt.date() <= date_range.end


def _apply_date_filter(
    results: list[SearchResult],
    date_range: DateRange,
) -> list[SearchResult]:
    """Keep only results whose path date falls within the date range.

    Non-dated files (MEMORY.md, etc.) are always included — they are
    evergreen long-term memory and should never be excluded by date.
    """
    filtered: list[SearchResult] = []
    for r in results:
        path_dt = _parse_memory_date_from_path(r.path)
        if path_dt is None:
            # Evergreen file — always include.
            filtered.append(r)
            continue
        if date_range.start <= path_dt.date() <= date_range.end:
            filtered.append(r)
    return filtered


def _apply_date_boost(
    results: list[SearchResult],
    date_range: DateRange,
) -> list[SearchResult]:
    """Boost scores for results whose path date falls within a soft date range.

    Non-dated files are left unchanged (no boost, no penalty).
    """
    boosted: list[SearchResult] = []
    for r in results:
        path_dt = _parse_memory_date_from_path(r.path)
        if path_dt is not None and date_range.start <= path_dt.date() <= date_range.end:
            boosted.append(replace(r, score=r.score * DATE_MATCH_BOOST))
        else:
            boosted.append(r)
    return boosted


# ── Importance boost ──────────────────────────────────────────────────────────


def _apply_importance_boost(results: list[SearchResult]) -> list[SearchResult]:
    """Apply importance and access-frequency boost to search scores.

    Formula::

        score *= importance_factor * frequency_factor

    Where:
        importance_factor = 0.5 + effective_importance
            importance=0.0 → treated as unset → ×1.0 (neutral)
            importance=0.5 → ×1.0 (neutral, unchanged)
            importance=1.0 → ×1.5 (50% boost)

        frequency_factor = 1.0 + log2(1 + access_count) * 0.15
            access_count=0  → ×1.0
            access_count=7  → ×1.45  (45% boost)
            access_count=31 → ×1.75  (75% boost, logarithmic cap)
    """
    boosted: list[SearchResult] = []
    for r in results:
        # Treat importance=0 as unset (payload default) and fall back to neutral 0.5.
        # importance=0 is indistinguishable from a missing field in older indexed chunks.
        # A stored importance of exactly 0.0 is indistinguishable from a missing field,
        # so we avoid penalising chunks that were indexed before importance was introduced.
        effective_importance = r.importance if r.importance > 0.0 else 0.5
        imp_factor = 0.5 + max(0.0, min(1.0, effective_importance))
        freq_factor = 1.0 + math.log2(1 + max(0, r.access_count)) * 0.15
        new_score = r.score * imp_factor * freq_factor
        boosted.append(replace(r, score=new_score))
    return boosted


# ── MMR reranking ─────────────────────────────────────────────────────────────


def _tokenize_for_mmr(text: str) -> frozenset[str]:
    """Extract lowercase alphanumeric tokens for MMR similarity comparison."""
    return frozenset(re.findall(r"[a-z0-9_]+", text.lower()))


def _jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compute_max_similarity(
    candidate: SearchResult,
    selected: list[SearchResult],
    token_cache: dict[str, frozenset[str]],
) -> float:
    """Return the maximum Jaccard similarity between a candidate and all selected results."""
    if not selected:
        return 0.0
    cand_tokens = token_cache[candidate.chunk_id]
    return max(_jaccard_similarity(cand_tokens, token_cache[sel.chunk_id]) for sel in selected)


def _select_best_mmr_candidate(
    remaining: list[SearchResult],
    selected: list[SearchResult],
    token_cache: dict[str, frozenset[str]],
    normalize: Callable[[float], float],
    lambda_: float,
) -> SearchResult | None:
    """Select the candidate with the highest MMR score."""
    best: SearchResult | None = None
    best_mmr = float("-inf")
    for candidate in remaining:
        max_sim = _compute_max_similarity(candidate, selected, token_cache)
        mmr_score = lambda_ * normalize(candidate.score) - (1 - lambda_) * max_sim
        best_score = best.score if best is not None else float("-inf")
        if mmr_score > best_mmr or (mmr_score >= best_mmr - 1e-9 and candidate.score > best_score):
            best_mmr = mmr_score
            best = candidate
    return best


def _mmr_rerank(results: list[SearchResult], config: MMRConfig) -> list[SearchResult]:
    """Rerank results using Maximal Marginal Relevance for diversity.

    Algorithm:
    1. Normalise scores to [0, 1].
    2. Iteratively select: mmr_score = lambda * relevance - (1 - lambda) * max_jaccard_sim_to_selected
    3. Jaccard similarity is computed over snippet token sets.
    """
    if not config.enabled or len(results) <= 1:
        return results

    lambda_ = max(0.0, min(1.0, config.lambda_))

    # lambda near 1 is equivalent to pure relevance sorting
    if lambda_ >= 1.0 - 1e-9:
        return sorted(results, key=lambda r: r.score, reverse=True)

    token_cache: dict[str, frozenset[str]] = {r.chunk_id: _tokenize_for_mmr(r.snippet) for r in results}

    scores = [r.score for r in results]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score

    def normalize(s: float) -> float:
        return 1.0 if score_range < 1e-9 else (s - min_score) / score_range

    selected: list[SearchResult] = []
    remaining = list(results)

    while remaining:
        best = _select_best_mmr_candidate(remaining, selected, token_cache, normalize, lambda_)
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    return selected


# ── Core searcher ─────────────────────────────────────────────────────────────


class QdrantMemorySearcher:
    """Memory searcher backed by Qdrant (dense) and SQLite FTS5 (BM25).

    Supports:
    - Hybrid mode (dense + BM25 fusion) when an embedding function is provided
    - FTS-only mode (automatic fallback when no embedding function)
    - Optional temporal decay
    - Optional MMR diversity reranking
    """

    def __init__(
        self,
        indexer: QdrantMemoryIndexer,
        embedding_fn: Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]] | None = None,
    ) -> None:
        self.indexer = indexer
        self.embedding_fn = embedding_fn

    async def search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.3,
        config: SearchConfig | None = None,
    ) -> list[SearchResult]:
        """Run a search and return ranked results.

        Triggers an incremental index sync if the indexer has pending changes.
        """
        if not query.strip():
            return []

        # Log dirty state for diagnostics; the file watcher already handles re-indexing
        # via index_file(force=True), so no additional sync is needed here.
        if self.indexer._dirty:
            logger.debug("onSearch: index has pending changes (watcher re-index in progress)")

        cfg = config or SearchConfig(max_results=max_results, min_score=min_score)
        candidates_limit = min(MAX_CANDIDATES, cfg.max_results * CANDIDATE_MULTIPLIER)

        # Build Qdrant payload filter for hard date ranges so the vector
        # engine can prune irrelevant candidates before scoring.
        qdrant_date_filter: Filter | None = None
        if cfg.date_range and not cfg.date_range.soft:
            qdrant_date_filter = _build_date_path_filter(cfg.date_range)

        # v2: build scope filter and merge with date filter
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        scope_must: list[Any] = []
        scope_must_not: list[Any] = []

        if cfg.scope_filter == "current_session" and cfg.current_session_id:
            scope_must.append(
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=cfg.current_session_id),
                )
            )
        elif cfg.scope_filter in ("owner_memory", "user_memory"):
            scope_must.append(
                FieldCondition(
                    key="scope",
                    match=MatchAny(any=["owner_memory", "user_memory"]),
                )
            )

        if cfg.exclude_is_raw_session:
            scope_must_not.append(
                FieldCondition(
                    key="is_raw_session",
                    match=MatchValue(value=True),
                )
            )

        if cfg.exclude_contains_secret:
            scope_must_not.append(
                FieldCondition(
                    key="contains_secret",
                    match=MatchValue(value=True),
                )
            )

        if cfg.exclude_is_tool_output:
            scope_must_not.append(
                FieldCondition(
                    key="is_tool_output",
                    match=MatchValue(value=True),
                )
            )

        # Merge date and scope filters into a single Qdrant Filter
        combined_must: list[Any] = list(scope_must)
        combined_must_not: list[Any] = list(scope_must_not)
        combined_should: list[Any] = []

        if qdrant_date_filter is not None:
            if qdrant_date_filter.must:
                combined_must.extend(qdrant_date_filter.must)
            if qdrant_date_filter.must_not:
                combined_must_not.extend(qdrant_date_filter.must_not)
            if qdrant_date_filter.should:
                combined_should.extend(qdrant_date_filter.should)

        combined_filter: Filter | None = None
        if combined_must or combined_must_not or combined_should:
            combined_filter = Filter(
                must=combined_must if combined_must else None,
                must_not=combined_must_not if combined_must_not else None,
                should=combined_should if combined_should else None,
            )

        if self.embedding_fn is not None:
            results = await self._hybrid_search(
                query,
                candidates_limit,
                cfg,
                query_filter=combined_filter,
            )
        else:
            results = self._fts_only_search(
                query,
                candidates_limit,
                date_range=cfg.date_range,
            )

        # v2: legacy fallback -- if no results with scope filter, try path-based for old data
        scope_applied = cfg.scope_filter in ("owner_memory", "user_memory", "current_session")
        if not results and scope_applied and self.embedding_fn is not None:
            if os.environ.get("OLD_METADATA_REALTIME_RECALL_ENABLED", "false").lower() == "true":
                legacy_must: list[Any] = [
                    FieldCondition(
                        key="path",
                        match=MatchAny(any=["MEMORY.md", "memory/"]),
                    ),
                ]
                if qdrant_date_filter is not None and qdrant_date_filter.must:
                    legacy_must.extend(qdrant_date_filter.must)
                legacy_filter = Filter(must=legacy_must)
                logger.debug(
                    "memory_search: scope filter returned 0 results, "
                    "falling back to path-based filter for legacy data"
                )
                results = await self._hybrid_search(
                    query, candidates_limit, cfg, query_filter=legacy_filter
                )

        # Date filtering / boosting (safety-net post-filter after merge).
        if cfg.date_range is not None:
            if cfg.date_range.soft:
                results = _apply_date_boost(results, cfg.date_range)
            else:
                results = _apply_date_filter(results, cfg.date_range)

        # Hard date ranges already constrain results to a specific time
        # window, so temporal decay would double-penalise older files.
        if cfg.date_range is not None and not cfg.date_range.soft:
            pass  # skip temporal decay
        else:
            results = _apply_temporal_decay(
                results,
                cfg.temporal_decay,
                working_dir=str(self.indexer.working_dir),
            )

        results = _apply_importance_boost(results)

        results.sort(key=lambda r: r.score, reverse=True)
        # High-importance memories (importance >= 0.7) use a relaxed threshold
        # to avoid filtering out valuable old memories after temporal decay.
        relaxed_threshold = cfg.min_score * 0.5
        filtered = [
            r for r in results if r.score >= cfg.min_score or (r.importance >= 0.7 and r.score >= relaxed_threshold)
        ]

        # Guarded single-best fallback: when no candidate passes the score
        # threshold, return at most the single best result if it exceeds the
        # absolute noise floor.  This avoids injecting bulk noise into the
        # LLM context while still surfacing a "maybe relevant" signal.
        if not filtered and results:
            best = results[0]
            if best.score >= ABSOLUTE_SCORE_FLOOR:
                logger.info(
                    "memory_search: no result above min_score=%.3f; returning single best candidate (score=%.3f)",
                    cfg.min_score,
                    best.score,
                )
                filtered = [best]
            else:
                logger.debug(
                    "memory_search: all %d candidates below absolute_floor=%.3f; returning empty (best score=%.3f)",
                    len(results),
                    ABSOLUTE_SCORE_FLOOR,
                    best.score,
                )

        results = filtered

        results = _mmr_rerank(results, cfg.mmr)

        final = results[: cfg.max_results]

        # Fire-and-forget: bump access_count for returned results
        if final and self.indexer.client is not None:
            loop = None
            with contextlib.suppress(RuntimeError):
                loop = asyncio.get_running_loop()
            if loop is not None and loop.is_running():
                asyncio.create_task(self._reinforce_results(final))

        return final

    async def _reinforce_results(self, results: list[SearchResult]) -> None:
        """Increment access_count and update last_accessed_at for search hits.

        Runs as a fire-and-forget background task — failures are silently
        logged at DEBUG level and never affect the search response.

        Uses the access_count already fetched during search (stored in SearchResult)
        to avoid a redundant retrieve round-trip. Concurrent searches may produce
        approximate counts, which is acceptable for a frequency-signal heuristic.
        """
        now_iso = datetime.now(UTC).isoformat()
        for r in results:
            try:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, r.chunk_id))
                await self.indexer.client.set_payload(
                    collection_name=self.indexer.collection_name,
                    payload={
                        "access_count": r.access_count + 1,
                        "last_accessed_at": now_iso,
                    },
                    points=[point_id],
                )
            except Exception as exc:
                logger.debug("Reinforce failed for chunk %s: %s", r.chunk_id, exc)

    async def _hybrid_search(
        self,
        query: str,
        candidates_limit: int,
        cfg: SearchConfig,
        query_filter: Filter | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search: merge dense vector results with BM25 keyword results."""
        vector_results = await self._search_vector(
            query,
            candidates_limit,
            query_filter=query_filter,
        )
        keyword_results = self._search_keyword(
            query,
            candidates_limit,
            date_range=cfg.date_range,
        )
        return _merge_hybrid_results(
            vector_results,
            keyword_results,
            cfg.vector_weight,
            cfg.text_weight,
        )

    async def _search_vector(
        self,
        query: str,
        limit: int,
        query_filter: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Run dense vector search against Qdrant."""
        if self.embedding_fn is None or limit <= 0:
            return []

        try:
            embeddings = await self.embedding_fn([query])
            query_vec = embeddings[0] if embeddings else []
        except Exception as exc:
            logger.warning("Embedding failed for query: %s", exc)
            return []

        if not query_vec:
            return []

        try:
            resp = await self.indexer.client.query_points(
                collection_name=self.indexer.collection_name,
                query=query_vec,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed: %s", exc)
            return []

        results = []
        for hit in resp.points:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", str(hit.id)),
                    "path": payload.get("path", ""),
                    "start_line": payload.get("start_line", 0),
                    "end_line": payload.get("end_line", 0),
                    "source": payload.get("source", "memory"),
                    "snippet": payload.get("snippet", "")[:SNIPPET_MAX_CHARS],
                    "vector_score": float(hit.score),
                    "text_score": 0.0,
                    "importance": float(payload.get("importance", 0.5)),
                    "access_count": int(payload.get("access_count", 0)),
                }
            )
        return results

    def _search_keyword(
        self,
        query: str,
        limit: int,
        date_range: DateRange | None = None,
    ) -> list[dict[str, Any]]:
        """Run BM25 keyword search via SQLite FTS5.

        Uses ``expand_query_for_fts`` to build an ``original OR keywords``
        expression, balancing exact-match precision with keyword recall.

        When *date_range* is a hard range, over-fetches and post-filters by
        path because the FTS ``path`` column is UNINDEXED.
        """
        if limit <= 0:
            return []

        fts_query = expand_query_for_fts(query)
        if not fts_query:
            return []

        # Over-fetch when date filtering is needed (path column is UNINDEXED).
        fetch_limit = limit * 3 if (date_range and not date_range.soft) else limit

        try:
            rows = self.indexer.fts_db.execute(
                "SELECT id, path, source, start_line, end_line, text, "
                "bm25(chunks_fts) AS rank "
                "FROM chunks_fts "
                "WHERE chunks_fts MATCH ? "
                "ORDER BY rank ASC "
                "LIMIT ?",
                (fts_query, fetch_limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 search failed: %s", exc)
            return []

        # Post-filter by date range when applicable.
        if date_range and not date_range.soft:
            rows = [r for r in rows if _path_in_date_range(r[1], date_range)]
            rows = rows[:limit]

        results = []
        for row in rows:
            chunk_id, path, source, start_line, end_line, text, rank = row
            text_score = bm25_rank_to_score(rank)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source": source,
                    "snippet": text[:SNIPPET_MAX_CHARS],
                    "vector_score": 0.0,
                    "text_score": text_score,
                }
            )
        return results

    def _fts_only_search(
        self,
        query: str,
        limit: int,
        date_range: DateRange | None = None,
    ) -> list[SearchResult]:
        """FTS-only search path (used when no embedding function is available).

        Two-pass retrieval:
        - Pass 1: ``expand_query_for_fts`` (exact phrase + keyword expansion)
        - Pass 2: ``extract_keywords`` individual keyword search (recall boost)

        When multiple keywords hit the same chunk, the max score is kept and
        ``hit_count`` is used as a tiebreaker::

            score = max_text_score * (1 + 0.1 * (hit_count - 1))
        """
        by_id: dict[str, dict[str, Any]] = {}

        def _upsert_row(row: tuple) -> None:
            chunk_id, path, source, start_line, end_line, text, rank = row
            text_score = bm25_rank_to_score(rank)
            if chunk_id not in by_id:
                by_id[chunk_id] = {
                    "chunk_id": chunk_id,
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source": source,
                    "snippet": text[:SNIPPET_MAX_CHARS],
                    "vector_score": 0.0,
                    "text_score": text_score,
                    "hit_count": 1,
                }
            else:
                # Keep max score; accumulate hit_count as tiebreaker
                by_id[chunk_id]["text_score"] = max(by_id[chunk_id]["text_score"], text_score)
                by_id[chunk_id]["hit_count"] += 1

        # Over-fetch when hard date filtering is needed.
        fetch_limit = limit * 3 if (date_range and not date_range.soft) else limit

        # Pass 1: expanded query (exact + keyword)
        expanded_query = expand_query_for_fts(query)
        if expanded_query:
            try:
                rows = self.indexer.fts_db.execute(
                    "SELECT id, path, source, start_line, end_line, text, "
                    "bm25(chunks_fts) AS rank "
                    "FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? "
                    "ORDER BY rank ASC "
                    "LIMIT ?",
                    (expanded_query, fetch_limit),
                ).fetchall()
                # Post-filter by date range when applicable.
                if date_range and not date_range.soft:
                    rows = [r for r in rows if _path_in_date_range(r[1], date_range)]
                for row in rows:
                    _upsert_row(row)
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 expand search failed: %s", exc)

        # Pass 2: individual keywords for additional recall.
        # Preserve temporal keywords so time-sensitive queries retain their
        # time dimension (e.g. "昨天" is not stripped as a stop word).
        keywords = extract_keywords(query, preserve_temporal=True)
        if not keywords:
            keywords = [query]

        for kw in keywords:
            fts_query = build_fts_query(kw)
            if not fts_query:
                continue
            try:
                rows = self.indexer.fts_db.execute(
                    "SELECT id, path, source, start_line, end_line, text, "
                    "bm25(chunks_fts) AS rank "
                    "FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? "
                    "ORDER BY rank ASC "
                    "LIMIT ?",
                    (fts_query, fetch_limit),
                ).fetchall()
                # Post-filter by date range when applicable.
                if date_range and not date_range.soft:
                    rows = [r for r in rows if _path_in_date_range(r[1], date_range)]
                for row in rows:
                    _upsert_row(row)
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 keyword search failed for '%s': %s", kw, exc)
                continue

        return [
            SearchResult(
                chunk_id=v["chunk_id"],
                path=v["path"],
                start_line=v["start_line"],
                end_line=v["end_line"],
                # More keyword hits → slight score boost as tiebreaker.
                # Cap at 5 hits so multi-hit noise cannot outrank high-quality
                # single-hit results (max multiplier: ×1.4).
                score=v["text_score"] * (1 + 0.1 * (min(v["hit_count"], 5) - 1)),
                snippet=v["snippet"],
                source=v["source"],
                vector_score=0.0,
                text_score=v["text_score"],
            )
            for v in by_id.values()
        ]


# ── Merge function ────────────────────────────────────────────────────────────


def _merge_hybrid_results(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    vector_weight: float,
    text_weight: float,
) -> list[SearchResult]:
    """Merge dense and BM25 results into a unified ranked list.

    score = vectorWeight * vectorScore + textWeight * textScore
    """
    by_id: dict[str, dict[str, Any]] = {}

    for r in vector_results:
        by_id[r["chunk_id"]] = {**r}

    for r in keyword_results:
        cid = r["chunk_id"]
        if cid in by_id:
            by_id[cid]["text_score"] = r["text_score"]
            # Prefer keyword snippet (usually more precise)
            if r["snippet"]:
                by_id[cid]["snippet"] = r["snippet"]
        else:
            by_id[cid] = {**r}

    results: list[SearchResult] = []
    for entry in by_id.values():
        score = vector_weight * entry["vector_score"] + text_weight * entry["text_score"]
        results.append(
            SearchResult(
                chunk_id=entry["chunk_id"],
                path=entry["path"],
                start_line=entry["start_line"],
                end_line=entry["end_line"],
                score=score,
                snippet=entry["snippet"],
                source=entry["source"],
                vector_score=entry["vector_score"],
                text_score=entry["text_score"],
                importance=float(entry.get("importance", 0.5)),
                access_count=int(entry.get("access_count", 0)),
            )
        )

    return results


# ── Factory ───────────────────────────────────────────────────────────────────


def create_search_config_from_env() -> SearchConfig:
    """Build a SearchConfig from environment variables.

    Supported environment variables:

    - ``MEMORY_VECTOR_WEIGHT``: weight for dense vector score in hybrid fusion
      (default: 0.7).  Must be in [0, 1].  ``text_weight`` is set to
      ``1 - vector_weight`` so the two always sum to 1.
    - ``MEMORY_TEMPORAL_DECAY_ENABLED``: enable/disable temporal decay
      (default: ``true``).
    - ``MEMORY_TEMPORAL_DECAY_HALF_LIFE_DAYS``: half-life in days for score
      decay (default: 90.0).
    - ``MEMORY_MMR_ENABLED``: enable/disable MMR diversity reranking
      (default: ``true``).
    - ``MEMORY_MMR_LAMBDA``: MMR lambda balancing relevance vs. diversity
      (default: 0.7).
    """
    temporal_decay_enabled = os.environ.get("MEMORY_TEMPORAL_DECAY_ENABLED", "true").lower() == "true"
    half_life_days = float(os.environ.get("MEMORY_TEMPORAL_DECAY_HALF_LIFE_DAYS", "90.0"))
    mmr_enabled = os.environ.get("MEMORY_MMR_ENABLED", "true").lower() == "true"
    mmr_lambda = float(os.environ.get("MEMORY_MMR_LAMBDA", str(MMR_LAMBDA)))

    # Allow runtime tuning of hybrid fusion weights via environment variables.
    # vector_weight + text_weight must sum to 1; text_weight is derived automatically.
    raw_vector_weight = os.environ.get("MEMORY_VECTOR_WEIGHT")
    if raw_vector_weight is not None:
        try:
            vector_weight = max(0.0, min(1.0, float(raw_vector_weight)))
        except ValueError:
            logger.warning(
                "Invalid MEMORY_VECTOR_WEIGHT=%r, using default %.1f",
                raw_vector_weight,
                VECTOR_WEIGHT,
            )
            vector_weight = VECTOR_WEIGHT
    else:
        vector_weight = VECTOR_WEIGHT
    text_weight = round(1.0 - vector_weight, 6)

    return SearchConfig(
        vector_weight=vector_weight,
        text_weight=text_weight,
        temporal_decay=TemporalDecayConfig(
            enabled=temporal_decay_enabled,
            half_life_days=half_life_days,
        ),
        mmr=MMRConfig(
            enabled=mmr_enabled,
            lambda_=mmr_lambda,
        ),
    )
