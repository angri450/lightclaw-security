"""Query/status/forget operations for memory backends."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightclaw.agent.memory.temporal_resolver import DateRange


class MemoryQueryService:
    """Encapsulate memory query and deletion workflows."""

    async def memory_search(
        self,
        *,
        query: str,
        max_results: int,
        min_score: float,
        memory_searcher: Any,
        logger: logging.Logger,
        session_distillation_service: Any = None,
        working_dir: str = "",
        session_search_enabled: bool = True,
        session_max_distilled: int = 30,
        session_search_timeout: float = 3.0,
        date_range: DateRange | None = None,
        # v2: session isolation params
        current_session_id: str = "",
        current_session_key: str = "",
        allow_all_sessions: bool = False,
        max_session_result_chars: int = 2000,
    ) -> str:
        if not query:
            return "Error: No query provided."

        if isinstance(max_results, int):
            max_results = min(max(max_results, 1), 100)
        else:
            max_results = 5

        if isinstance(min_score, float):
            min_score = min(max(min_score, 0.001), 0.999)
        else:
            min_score = 0.3

        if memory_searcher is None:
            return "No memory backend available."

        memory_result = await self.backend_memory_search(
            query=query,
            max_results=max_results,
            min_score=min_score,
            memory_searcher=memory_searcher,
            logger=logger,
            date_range=date_range,
        )

        if not session_search_enabled or not session_distillation_service or not working_dir:
            return memory_result

        # Run session FTS scan in a thread to avoid blocking the event loop.
        try:
            session_results = await asyncio.wait_for(
                asyncio.to_thread(
                    session_distillation_service.search_sessions,
                    working_dir,
                    query,
                    max_results,
                    session_max_distilled,
                    3,  # context_lines
                    current_session_id,
                    current_session_key,
                    allow_all_sessions,
                    max_session_result_chars,
                ),
                timeout=session_search_timeout,
            )
        except TimeoutError:
            logger.warning("session_search timed out (%.1fs), skipping", session_search_timeout)
            return memory_result
        except Exception as exc:
            logger.warning("session_search failed: %s", exc)
            return memory_result

        if not session_results:
            return memory_result

        # Filter out low-quality snippets (e.g. pure JSON structure lines that slipped
        # through text extraction) — a snippet shorter than 20 chars carries no useful signal.
        session_results = [r for r in session_results if len(r.get("snippet", "").strip()) >= 20]
        if not session_results:
            return memory_result

        session_lines: list[str] = []
        for r in session_results:
            session_lines.append(f"[{r['session']}]\n{r['snippet']}")
        session_section = "\n\n---\n\n".join(session_lines)

        parts: list[str] = []
        if memory_result and memory_result not in ("No relevant memories found.", "Memory search failed."):
            parts.append(memory_result)
        parts.append(f"### Session History Matches\n\n{session_section}")
        return "\n\n---\n\n".join(parts)

    async def backend_memory_search(
        self,
        *,
        query: str,
        max_results: int,
        min_score: float,
        memory_searcher: Any,
        logger: logging.Logger,
        date_range: DateRange | None = None,
    ) -> str:
        from lightclaw.agent.memory.qdrant_searcher import create_search_config_from_env

        cfg = create_search_config_from_env()
        cfg.max_results = max_results
        cfg.min_score = min_score
        cfg.date_range = date_range

        try:
            results = await memory_searcher.search(query=query, config=cfg)
        except Exception as exc:
            logger.warning("Memory backend search failed: %s", exc)
            return "Memory search failed."

        if not results:
            return "No relevant memories found."

        lines: list[str] = []
        for result in results:
            lines.append(f"[{result.path}:{result.start_line}] (score={result.score:.3f})\n{result.snippet}")
        return "\n\n---\n\n".join(lines)

    async def memory_get(
        self,
        *,
        path: str,
        offset: int | None,
        limit: int | None,
        memory_indexer: Any,
        logger: logging.Logger,
    ) -> str:
        if memory_indexer is not None:
            try:
                return memory_indexer.read_file(path, offset, limit)
            except Exception as exc:
                logger.warning("Memory backend memory_get failed: %s", exc)
        return ""

    async def memory_status(
        self,
        *,
        memory_indexer: Any,
        memory_searcher: Any,
        logger: logging.Logger,
    ) -> dict:
        vector_enabled = memory_searcher is not None and memory_searcher.embedding_fn is not None

        chunk_count = 0
        if memory_indexer is not None:
            try:
                chunk_count = await memory_indexer.get_chunk_count()
            except Exception as exc:
                logger.warning("memory_status: get_chunk_count failed: %s", exc)

        dirty = memory_indexer._dirty if memory_indexer is not None else False

        fts_count = 0
        if memory_indexer is not None:
            try:
                row = memory_indexer.fts_db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
                fts_count = row[0] if row else 0
            except Exception as exc:
                logger.warning("memory_status: fts_count failed: %s", exc)

        return {
            "chunk_count": chunk_count,
            "vector_enabled": vector_enabled,
            "dirty": dirty,
            "fts_count": fts_count,
        }

    async def memory_forget_by_query(
        self,
        *,
        query: str,
        max_candidates: int,
        min_score: float,
        auto_delete_threshold: float,
        memory_searcher: Any,
        memory_indexer: Any,
        logger: logging.Logger,
    ) -> str:
        if not query:
            return "Error: No query provided for memory_forget."
        if memory_searcher is None:
            return "Error: No memory search backend available."

        from lightclaw.agent.memory.qdrant_searcher import SearchResult, create_search_config_from_env

        cfg = create_search_config_from_env()
        cfg.max_results = max_candidates
        cfg.min_score = min_score

        try:
            results: list[SearchResult] = await memory_searcher.search(query=query, config=cfg)
        except Exception as exc:
            logger.warning("memory_forget_by_query search failed: %s", exc)
            return f"Error: Memory search failed: {exc}"

        if not results:
            return "No matching memories found for the given query."

        if len(results) == 1 and results[0].score >= auto_delete_threshold:
            result = results[0]
            if memory_indexer is not None:
                await memory_indexer.delete_chunks_by_ids([result.chunk_id])
            snippet_preview = result.snippet[:100].replace("\n", " ")
            return (
                f"Deleted memory (auto, score={result.score:.3f}):\n"
                f"  [{result.path}:{result.start_line}-{result.end_line}] {snippet_preview}"
            )

        lines: list[str] = [
            f"Found {len(results)} candidate memories. Please confirm which to delete by providing chunk_ids:\n"
        ]
        for idx, result in enumerate(results, 1):
            snippet_preview = result.snippet[:120].replace("\n", " ")
            lines.append(
                f"  {idx}. chunk_id={result.chunk_id} | [{result.path}:{result.start_line}-{result.end_line}] "
                f"(score={result.score:.3f})\n     {snippet_preview}"
            )
        return "\n".join(lines)

    async def memory_forget_by_file(
        self,
        *,
        file_name: str,
        memory_indexer: Any,
    ) -> str:
        from lightclaw.agent.memory.agent_md_manager import AGENT_MD_MANAGER

        if file_name.upper() in ("MEMORY.MD", "MEMORY"):
            AGENT_MD_MANAGER.write_working_md("MEMORY.md", "")
            if memory_indexer is not None:
                await memory_indexer.delete_file_chunks("MEMORY.md")
            return "MEMORY.md content cleared (file preserved)."

        try:
            deleted = AGENT_MD_MANAGER.delete_memory_md(file_name)
        except ValueError as exc:
            return f"Error: {exc}"

        if not deleted:
            return f"Memory file not found: {file_name}"

        if memory_indexer is not None:
            md_name = file_name if file_name.endswith(".md") else f"{file_name}.md"
            rel_path = f"memory/{md_name}"
            await memory_indexer.delete_file_chunks(rel_path)
        return f"Deleted memory file: {file_name} (and its vector chunks)."

    async def delete_vector_chunks_for_file(
        self,
        *,
        file_name: str,
        memory_indexer: Any,
    ) -> None:
        if memory_indexer is None:
            return
        md_name = file_name if file_name.endswith(".md") else f"{file_name}.md"
        rel_path = f"memory/{md_name}"
        await memory_indexer.delete_file_chunks(rel_path)

    async def memory_forget_by_chunk_ids(
        self,
        *,
        chunk_ids: list[str],
        memory_indexer: Any,
    ) -> str:
        if not chunk_ids:
            return "Error: No chunk_ids provided."
        if memory_indexer is None:
            return "Error: No memory indexer available."

        count = await memory_indexer.delete_chunks_by_ids(chunk_ids)
        if count == 0:
            return "No valid chunk_ids were deleted (invalid format?)."
        return f"Deleted {count} memory chunk(s) successfully."

    def session_search(
        self,
        *,
        query: str,
        working_dir: str,
        session_distillation_service: Any,
        max_results: int = 5,
        logger: logging.Logger,
        current_session_id: str = "",
        current_session_key: str = "",
        allow_all_sessions: bool = False,
    ) -> str:
        """Search raw session files for fine-grained details not captured by distillation.

        Complements ``memory_search`` for exact values, commands, and paths
        that may have been lost during the distillation step.

        Args:
            query:                        Search query string.
            working_dir:                  Agent working directory.
            session_distillation_service: ``SessionDistillationService`` instance.
            max_results:                  Maximum number of snippets to return.
            logger:                       Logger instance.

        Returns:
            Formatted string of matching session snippets, or a not-found message.
        """
        if not query:
            return "Error: No query provided."
        if session_distillation_service is None:
            return "Session search unavailable: no distillation service configured."

        try:
            results = session_distillation_service.search_sessions(
                working_dir=working_dir,
                query=query,
                max_results=max_results,
                current_session_id=current_session_id,
                current_session_key=current_session_key,
                allow_all_sessions=allow_all_sessions,
            )
        except Exception as exc:
            logger.warning("session_search failed: %s", exc)
            return "Session search failed."

        if not results:
            return "No matching content found in session history."

        lines: list[str] = []
        for r in results:
            lines.append(f"[{r['session']}]\n{r['snippet']}")
        return "\n\n---\n\n".join(lines)
