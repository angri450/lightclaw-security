"""Auto memory recall step — v2: Session/Principal isolation firewall edition."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage

from lightclaw.agent.bootstrap_state import is_bootstrap_pending
from lightclaw.agent.core.engines.langgraph.middleware.steps._message_utils import extract_latest_user_text
from lightclaw.constant import contains_secret

if TYPE_CHECKING:
    from lightclaw.agent.memory import MemoryManager

logger = logging.getLogger(__name__)


class AutoRecallInjector:
    """Search memory for context relevant to the current user message and inject it.

    v2 changes:
    - ActorContext-aware: respects session/principal scope
    - Secret firewall: filters out credentials before injection
    - Budget firewall: per-candidate + total hard limits
    - Source attribution: every candidate annotated with metadata
    - DAILY_TALK disabled by default for realtime recall
    - Session FTS filtered by current_session only
    - Old metadata rejected by default
    """

    def __init__(
        self,
        memory_manager: MemoryManager | None,
        language: str = "zh",
    ) -> None:
        self._memory_manager = memory_manager
        self._language = language

        from lightclaw.constant import (
            AUTO_RECALL_ENABLED,
            AUTO_RECALL_MAX_CANDIDATE_CHARS,
            AUTO_RECALL_MAX_DAILY_TALK_CHARS,
            AUTO_RECALL_MAX_MEMORY_CHARS,
            AUTO_RECALL_MAX_QUERY_LENGTH,
            AUTO_RECALL_MAX_RESULTS,
            AUTO_RECALL_MAX_SESSION_RESULT_CHARS,
            AUTO_RECALL_MAX_TOTAL_CHARS,
            AUTO_RECALL_MIN_SCORE,
            AUTO_RECALL_TIMEOUT,
            DAILY_TALK_INJECTION_FILTER_ENABLED,
            DAILY_TALK_INJECTION_MAX_CHARS,
            DAILY_TALK_INJECTION_MAX_CHUNKS,
            DAILY_TALK_REALTIME_INJECTION_ENABLED,
            INCLUDE_RAW_SESSIONS_IN_REALTIME_RECALL,
            MEMORY_MD_INJECTION_ENABLED,
            MEMORY_MD_INJECTION_MAX_CHARS,
            OLD_METADATA_REALTIME_RECALL_ENABLED,
            SECRET_REALTIME_RECALL_ENABLED,
            WORKING_DIR,
        )

        self._enabled = AUTO_RECALL_ENABLED
        self._max_results = AUTO_RECALL_MAX_RESULTS
        self._min_score = AUTO_RECALL_MIN_SCORE
        self._max_query_length = AUTO_RECALL_MAX_QUERY_LENGTH
        self._timeout = AUTO_RECALL_TIMEOUT
        self._max_total_chars = AUTO_RECALL_MAX_TOTAL_CHARS
        self._max_candidate_chars = AUTO_RECALL_MAX_CANDIDATE_CHARS
        self._max_session_result_chars = AUTO_RECALL_MAX_SESSION_RESULT_CHARS
        self._max_daily_talk_chars = AUTO_RECALL_MAX_DAILY_TALK_CHARS
        self._max_memory_chars = AUTO_RECALL_MAX_MEMORY_CHARS

        self._memory_md_injection_enabled = MEMORY_MD_INJECTION_ENABLED
        self._memory_md_injection_max_chars = MEMORY_MD_INJECTION_MAX_CHARS
        self._daily_talk_realtime_enabled = DAILY_TALK_REALTIME_INJECTION_ENABLED
        self._daily_talk_injection_max_chars = DAILY_TALK_INJECTION_MAX_CHARS
        self._daily_talk_injection_filter_enabled = DAILY_TALK_INJECTION_FILTER_ENABLED
        self._daily_talk_injection_max_chunks = DAILY_TALK_INJECTION_MAX_CHUNKS
        self._include_raw_sessions = INCLUDE_RAW_SESSIONS_IN_REALTIME_RECALL
        self._old_metadata_enabled = OLD_METADATA_REALTIME_RECALL_ENABLED
        self._secret_recall_enabled = SECRET_REALTIME_RECALL_ENABLED
        self._working_dir = WORKING_DIR
        self._bootstrap_pending = is_bootstrap_pending(WORKING_DIR)

    async def get_block(self, messages: list[BaseMessage], actor_context=None) -> str:
        """Return a recall-context block string, or empty string if not applicable."""
        from lightclaw.agent.core.actor_context import ActorContext
        from lightclaw.agent.memory.recall_candidate import RecallCandidate

        if actor_context is None:
            actor_context = ActorContext.dashboard_owner()

        trust_val = actor_context.trust_level
        if hasattr(trust_val, 'value'):
            trust_val = trust_val.value

        logger.info(
            "[ActorContext] channel=%s chat_type=%s chat_id=%s sender_id_hash=%s "
            "principal_id=%s trust_level=%s session_id=%s is_owner=%s",
            actor_context.channel,
            actor_context.chat_type.value if hasattr(actor_context.chat_type, 'value') else str(actor_context.chat_type),
            actor_context.chat_id,
            actor_context.sender_id_hash,
            actor_context.principal_id,
            trust_val,
            actor_context.session_id,
            actor_context.is_owner,
        )

        candidates: list[RecallCandidate] = []
        denied_count = 0
        denied_private_to_group = 0
        denied_unknown_owner_memory = 0
        dropped_secret = 0
        dropped_old_metadata = 0

        # --- MEMORY.md injection (owner only) ---
        memory_md_content = ""
        if self._memory_md_injection_enabled and actor_context.is_owner:
            io_started_at = time.perf_counter()
            memory_md_content = self._read_memory_md(
                self._working_dir,
                min(self._memory_md_injection_max_chars, self._max_memory_chars),
            )
            logger.info(
                "[Recall][io] op=read_memory_md chars=%d duration_ms=%.2f",
                len(memory_md_content),
                (time.perf_counter() - io_started_at) * 1000,
            )
            if memory_md_content:
                candidate = RecallCandidate(
                    text=memory_md_content,
                    source_type="memory_md",
                    scope="owner_memory",
                    principal_id=actor_context.principal_id,
                    trust_level=trust_val,
                    path="MEMORY.md",
                    char_count=len(memory_md_content),
                )
                candidates, denied_count, denied_private_to_group, denied_unknown_owner_memory, dropped_secret, dropped_old_metadata = (
                    self._apply_filters(
                        candidates, [candidate], actor_context,
                        denied_count, denied_private_to_group,
                        denied_unknown_owner_memory, dropped_secret, dropped_old_metadata,
                    )
                )

        # --- DAILY_TALK.md injection (v2: OFF by default) ---
        if self._daily_talk_realtime_enabled and not self._bootstrap_pending and self._max_daily_talk_chars > 0:
            io_started_at = time.perf_counter()
            daily_talk_content = self._read_daily_talk(
                self._working_dir,
                min(self._daily_talk_injection_max_chars, self._max_daily_talk_chars),
            )
            logger.info(
                "[Recall][io] op=read_daily_talk chars=%d duration_ms=%.2f",
                len(daily_talk_content),
                (time.perf_counter() - io_started_at) * 1000,
            )
            if daily_talk_content:
                if self._daily_talk_injection_filter_enabled:
                    user_text = extract_latest_user_text(messages)
                    if user_text and len(user_text.strip()) >= 5:
                        daily_talk_content = self._filter_daily_talk_by_relevance(
                            daily_talk_content,
                            query=user_text[: self._max_query_length],
                            max_chunks=self._daily_talk_injection_max_chunks,
                        )
                if daily_talk_content:
                    candidate = RecallCandidate(
                        text=daily_talk_content,
                        source_type="daily_talk",
                        scope="global",
                        path="DAILY_TALK.md",
                        char_count=len(daily_talk_content),
                        has_metadata=False,
                    )
                    if not self._old_metadata_enabled:
                        dropped_old_metadata += 1
                        logger.info(
                            "[Recall][old_metadata] dropped daily_talk (old_metadata_realtime_recall=false)"
                        )
                    else:
                        candidates.append(candidate)

        # --- Semantic recall ---
        if not self._enabled or self._memory_manager is None:
            return self._render_block(candidates, actor_context)

        for m in messages:
            if "<auto-recall-context>" in (m.content if isinstance(m.content, str) else ""):
                return self._render_block(candidates, actor_context)

        user_text = extract_latest_user_text(messages)
        if not user_text or len(user_text.strip()) < 5:
            return self._render_block(candidates, actor_context)

        query = user_text[: self._max_query_length]

        from lightclaw.agent.memory.temporal_resolver import resolve_temporal_expr
        date_range = resolve_temporal_expr(query)

        try:
            io_started_at = time.perf_counter()
            result = await asyncio.wait_for(
                self._memory_manager.memory_search(
                    query=query,
                    max_results=self._max_results,
                    min_score=self._min_score,
                    date_range=date_range,
                ),
                timeout=self._timeout,
            )
            logger.info(
                "[Recall][io] op=memory_search duration_ms=%.2f query_chars=%d result_chars=%d max_results=%d",
                (time.perf_counter() - io_started_at) * 1000,
                len(query),
                len(result or ""),
                self._max_results,
            )
        except TimeoutError:
            logger.warning("Auto-recall timed out after %.1fs", self._timeout)
            return self._render_block(candidates, actor_context)
        except Exception:
            logger.debug("Auto-recall search failed, continuing without recall")
            return self._render_block(candidates, actor_context)

        if (
            not result
            or result in ("No relevant memories found.", "No memory backend available.")
            or result.startswith("Error:")
        ):
            return self._render_block(candidates, actor_context)

        semantic_candidates = self._parse_semantic_result(result, actor_context)
        candidates, denied_count, denied_private_to_group, denied_unknown_owner_memory, dropped_secret, dropped_old_metadata = (
            self._apply_filters(
                candidates, semantic_candidates, actor_context,
                denied_count, denied_private_to_group,
                denied_unknown_owner_memory, dropped_secret, dropped_old_metadata,
            )
        )

        total_collected = (1 if memory_md_content else 0) + len(semantic_candidates)
        logger.info(
            "[Recall][authz] allowed=%d denied=%d denied_private_to_group=%d denied_unknown_owner_memory=%d",
            len(candidates), denied_count, denied_private_to_group, denied_unknown_owner_memory,
        )
        logger.info("[Recall][secret] dropped_secret_candidates=%d", dropped_secret)
        logger.info(
            "[Recall][candidate] collected=%d kept=%d dropped_cross_session=%d",
            total_collected, len(candidates),
            denied_count + denied_private_to_group + denied_unknown_owner_memory,
        )

        return self._render_block(candidates, actor_context)

    # --- Filter pipeline ---

    def _apply_filters(
        self,
        existing: list,
        new_candidates: list,
        actor_context,
        denied_count: int,
        denied_private_to_group: int,
        denied_unknown_owner_memory: int,
        dropped_secret: int,
        dropped_old_metadata: int,
    ):
        """Apply secret, authz, scope, and metadata filters."""
        results = list(existing)

        trust_val = actor_context.trust_level
        if hasattr(trust_val, 'value'):
            trust_val = trust_val.value

        for c in new_candidates:
            # Secret filter
            if contains_secret(c.text):
                if self._secret_recall_enabled:
                    results.append(c)
                else:
                    dropped_secret += 1
                    logger.info("[Recall][secret] dropped source=%s", c.source_type)
                continue

            # Old metadata filter
            if not c.has_metadata and not self._old_metadata_enabled:
                dropped_old_metadata += 1
                logger.info("[Recall][old_metadata] dropped source=%s", c.source_type)
                continue

            # Unknown/group_member cannot access owner_memory
            if trust_val in ("unknown", "group_member") and c.scope in ("owner_memory", "private_session"):
                denied_unknown_owner_memory += 1
                logger.info(
                    "[Recall][authz] denied: trust=%s scope=%s",
                    trust_val, c.scope,
                )
                continue

            # Private-to-group block
            if actor_context.is_group_chat and c.scope in ("owner_memory", "private_session"):
                from lightclaw.constant import PRIVATE_TO_GROUP_RECALL_ENABLED
                if not PRIVATE_TO_GROUP_RECALL_ENABLED:
                    denied_private_to_group += 1
                    continue

            # Group-to-private block
            if not actor_context.is_group_chat and c.scope == "group_memory":
                from lightclaw.constant import GROUP_TO_PRIVATE_RECALL_ENABLED
                if not GROUP_TO_PRIVATE_RECALL_ENABLED:
                    denied_count += 1
                    continue

            results.append(c)

        return results, denied_count, denied_private_to_group, denied_unknown_owner_memory, dropped_secret, dropped_old_metadata

    def _parse_semantic_result(self, result: str, actor_context) -> list:
        """Parse raw semantic search result into RecallCandidate list."""
        from lightclaw.agent.memory.recall_candidate import RecallCandidate

        candidates = []
        sections = re.split(r'\n\n---\n\n', result)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            match = re.match(r'\[([^\]]+)\]\s+\(score=([\d.]+)\)\s*\n(.*)', section, re.DOTALL)
            if match:
                path = match.group(1)
                score = float(match.group(2))
                text = match.group(3).strip()
                source_type = "qdrant_memory"
                scope = "user_memory"
            else:
                sess_match = re.match(r'\[([^\]]+)\]\n(.*)', section, re.DOTALL)
                if sess_match:
                    path = sess_match.group(1)
                    score = 0.5
                    text = sess_match.group(2).strip()
                    source_type = "session_fts"
                    scope = "current_session"
                else:
                    path = "unknown"
                    score = 0.5
                    text = section
                    source_type = "unknown"
                    scope = "unknown"

            candidate = RecallCandidate(
                text=text,
                source_type=source_type,
                scope=scope,
                session_id=actor_context.session_id if scope == "current_session" else "",
                path=path,
                score=score,
                char_count=len(text),
                has_metadata=(source_type != "session_fts"),
            )
            candidates.append(candidate)

        return candidates

    def _render_block(self, candidates: list, actor_context) -> str:
        """Render candidates into recall block with budget enforcement."""
        if not candidates:
            return ""

        # Deduplicate
        seen = set()
        deduped = []
        for c in candidates:
            norm = c.text.strip()[:200]
            if norm not in seen:
                seen.add(norm)
                deduped.append(c)

        # Sort by score
        deduped.sort(key=lambda c: c.score, reverse=True)

        # Trim each candidate
        for c in deduped:
            if len(c.text) > self._max_candidate_chars:
                c.text = c.text[:self._max_candidate_chars] + "\n[... truncated ...]"
                c.truncated = True

        # Apply total budget
        total_chars = 0
        kept = []
        dropped_budget = 0
        for c in deduped:
            block_line = c.render_for_block()
            if total_chars + len(block_line) > self._max_total_chars:
                dropped_budget += 1
                continue
            total_chars += len(block_line)
            kept.append(c)

        if dropped_budget > 0:
            logger.info(
                "[Recall][budget] dropped_budget=%d before_chars=%d after_chars=%d total_cap=%d",
                dropped_budget,
                sum(len(c.text) for c in deduped),
                total_chars,
                self._max_total_chars,
            )

        if not kept:
            return ""

        parts = [c.render_for_block() for c in kept]
        block = "<auto-recall-context>\n" + "\n\n".join(parts) + "\n</auto-recall-context>"

        logger.info(
            "[Recall][budget] before_chars=%d after_chars=%d total_cap=%d sources=memory_md:%d,daily_talk:%d,qdrant:%d,session_fts:%d",
            sum(len(c.text) for c in deduped),
            total_chars,
            self._max_total_chars,
            sum(1 for c in kept if c.source_type == "memory_md"),
            sum(1 for c in kept if c.source_type == "daily_talk"),
            sum(1 for c in kept if c.source_type == "qdrant_memory"),
            sum(1 for c in kept if c.source_type == "session_fts"),
        )

        return block

    # --- Static helpers ---

    @staticmethod
    def _read_memory_md(working_dir: Path, max_chars: int) -> str:
        memory_md_path = working_dir / "MEMORY.md"
        if not memory_md_path.is_file():
            return ""
        try:
            text = memory_md_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not text:
            return ""
        if len(text) > max_chars:
            return "[... truncated ...]\n\n" + text[-max_chars:]
        return text

    @staticmethod
    def _read_daily_talk(working_dir: Path, max_chars: int) -> str:
        daily_talk_path = working_dir / "DAILY_TALK.md"
        if not daily_talk_path.is_file():
            return ""
        try:
            text = daily_talk_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not text:
            return ""
        body = re.sub(r"<!--\s*daily_talk_date:\s*\d{4}-\d{2}-\d{2}\s*-->", "", text, count=1)
        lines = body.splitlines()
        if lines and lines[0].strip() == "# DAILY_TALK":
            lines = lines[1:]
        lines = [ln for ln in lines if not re.match(r"^## \d{2}:\d{2}:\d{2}\s*$", ln)]
        body = "\n".join(lines).strip()
        if not body:
            return ""
        if len(body) > max_chars:
            return "[... truncated ...]\n\n" + body[-max_chars:]
        return body

    @staticmethod
    def _filter_daily_talk_by_relevance(body: str, query: str, max_chunks: int) -> str:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        if not paragraphs:
            return body
        if len(paragraphs) <= max_chunks:
            return body
        cjk_tokens = set(re.findall(r"[一-鿿぀-ヿ가-힯]", query))
        ascii_tokens = set(w.lower() for w in re.findall(r"[a-zA-Z0-9]+", query) if len(w) >= 2)
        all_tokens = cjk_tokens | ascii_tokens
        if not all_tokens:
            return "\n\n".join(paragraphs[-max_chunks:])
        def _score(para: str) -> float:
            lower = para.lower()
            return sum(1.0 for tok in all_tokens if tok in lower)
        scored = [(idx, _score(p)) for idx, p in enumerate(paragraphs)]
        top_indices = sorted(
            [idx for idx, sc in scored if sc > 0],
            key=lambda i: (scored[i][1], i), reverse=True,
        )[:max_chunks]
        if not top_indices:
            return "\n\n".join(paragraphs[-max_chunks:])
        top_indices.sort()
        return "\n\n".join(paragraphs[i] for i in top_indices)

    @staticmethod
    def _build_daily_talk_block(content: str, language: str) -> str:
        from lightclaw.agent.prompt_catalog import get_daily_talk_block_template
        return get_daily_talk_block_template(language).format(content=content)

    @staticmethod
    def _build_memory_md_block(content: str, language: str) -> str:
        from lightclaw.agent.prompt_catalog import get_memory_md_block_template
        return get_memory_md_block_template(language).format(content=content)

    @staticmethod
    def _build_block(snippet: str, language: str) -> str:
        from lightclaw.agent.prompt_catalog import get_auto_recall_block_template
        return get_auto_recall_block_template(language).format(snippet=snippet)
