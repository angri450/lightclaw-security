"""
Qdrant memory indexer.
Aligns with openclaw: src/memory/manager.ts + manager-sync-ops.ts + manager-embedding-ops.ts

Features:
- File watching (watchdog, aligns with chokidar watcher)
- Markdown chunking (aligns with chunkMarkdown, CHUNK_SIZE=400 chars, OVERLAP=80 chars)
- Embedding (calls EMBEDDING_BASE_URL/EMBEDDING_MODEL_NAME)
- Upsert to Qdrant collection
- INSERT OR REPLACE to SQLite FTS5
- Full sync at startup (aligns with warmSession + onSessionStart sync)
- Incremental sync on file changes (aligns with chokidar watcher markDirty)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sqlite3
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Chunking parameters (aligns with openclaw chunkMarkdown) ─────────────────
# openclaw: chunking.tokens=100, maxChars = tokens * 4 = 400
CHUNK_SIZE = 400  # character count
CHUNK_OVERLAP = 80  # overlap characters (aligns with overlap * 4 = 20 * 4 = 80)
SNIPPET_MAX_CHARS = 700  # aligns with openclaw SNIPPET_MAX_CHARS

# ── Ignored directory names (aligns with openclaw IGNORED_MEMORY_WATCH_DIR_NAMES) ──
IGNORED_DIR_NAMES = {
    ".git",
    "node_modules",
    ".pnpm-store",
    ".venv",
    "venv",
    ".tox",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}

# ── File names that should not be indexed (system config files, not user memory) ──
IGNORED_FILE_NAMES = {
    "PROACTIVE.md",
    "IDENTITY.md",
    "AGENTS.md",
    "SOUL.md",
    "BOOTSTRAP.md",
    # DAILY_TALK.md is injected directly into the system prompt; no need to index it.
    "DAILY_TALK.md",
}

# ── Secret detection patterns (for index-time contains_secret tagging) ────────
_SECRET_PATTERNS = [
    re.compile(r'(?:api[_-]?key|apikey|secret|token|password)\s*[:=]\s*[\'"]?\S+', re.IGNORECASE),
    re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', re.IGNORECASE),
    re.compile(r'sk-[a-zA-Z0-9]{20,}', re.IGNORECASE),
    re.compile(r'ghp_[a-zA-Z0-9]{36}', re.IGNORECASE),
]


def _contains_secret(text: str) -> bool:
    """Simple secret detection for index-time tagging."""
    return any(p.search(text) for p in _SECRET_PATTERNS)


def _hash_text(text: str) -> str:
    """SHA256[:16], aligns with openclaw hashText."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _split_line_into_segments(raw_line: str, max_chars: int) -> list[str]:
    """Split a long line into segments of max_chars (aligns with openclaw segments logic)."""
    if len(raw_line) == 0:
        return [""]
    return [raw_line[start : start + max_chars] for start in range(0, len(raw_line), max_chars)]


def _carry_overlap(
    current: list[tuple[str, int]],
    overlap_chars: int,
) -> tuple[list[tuple[str, int]], int]:
    """Retain trailing overlap_chars worth of lines as prefix for next chunk (aligns with openclaw carryOverlap)."""
    if overlap_chars <= 0 or not current:
        return [], 0
    acc = 0
    kept: list[tuple[str, int]] = []
    for line, line_no in reversed(current):
        acc += len(line) + 1
        kept.insert(0, (line, line_no))
        if acc >= overlap_chars:
            break
    return kept, sum(len(line) + 1 for line, _ in kept)


def _flush_chunk(current: list[tuple[str, int]], chunks: list[dict[str, Any]]) -> None:
    """Flush the current buffer into the chunks list (aligns with openclaw flush)."""
    if not current:
        return
    text = "\n".join(line for line, _ in current)
    chunks.append(
        {
            "start_line": current[0][1],
            "end_line": current[-1][1],
            "text": text,
            "hash": _hash_text(text),
        }
    )


def _chunk_markdown(content: str) -> list[dict[str, Any]]:
    """
    Aligns with openclaw chunkMarkdown (internal.ts:334).
    Sliding window chunking, CHUNK_SIZE=400 chars, OVERLAP=80 chars.
    Returns list of {start_line, end_line, text, hash} (1-indexed).
    """
    if not content:
        return []
    lines = content.split("\n")
    if not lines:
        return []

    max_chars = max(32, CHUNK_SIZE)
    overlap_chars = max(0, CHUNK_OVERLAP)
    chunks: list[dict[str, Any]] = []
    current: list[tuple[str, int]] = []
    current_chars = 0

    for i, raw_line in enumerate(lines):
        line_no = i + 1
        for segment in _split_line_into_segments(raw_line, max_chars):
            line_size = len(segment) + 1
            if current_chars + line_size > max_chars and current:
                _flush_chunk(current, chunks)
                current, current_chars = _carry_overlap(current, overlap_chars)
            current.append((segment, line_no))
            current_chars += line_size

    _flush_chunk(current, chunks)
    return chunks


def _make_chunk_id(source: str, path: str, start_line: int, end_line: int, text_hash: str, model: str) -> str:
    """
    Aligns with openclaw indexFile id generation logic:
    hashText(`${source}:${path}:${startLine}:${endLine}:${hash}:${model}`)
    """
    raw = f"{source}:{path}:{start_line}:{end_line}:{text_hash}:{model}"
    return _hash_text(raw)


def _should_ignore_path(path_str: str) -> bool:
    """Aligns with openclaw shouldIgnoreMemoryWatchPath."""
    p = Path(path_str)
    if p.name in IGNORED_FILE_NAMES:
        return True
    return any(part.lower() in IGNORED_DIR_NAMES for part in p.parts)


class QdrantMemoryIndexer:
    """
    Qdrant memory indexer.
    Aligns with openclaw MemoryIndexManager (file watching + chunking + embedding + write operations).
    """

    def __init__(
        self,
        working_dir: str,
        embedding_fn: Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]] | None,
        qdrant_client: Any,  # AsyncQdrantClient
        fts_db: sqlite3.Connection,
        collection_name: str,
        model_name: str,
        vector_dims: int,
    ) -> None:
        self.working_dir = Path(working_dir).resolve()
        self.embedding_fn = embedding_fn
        self.client = qdrant_client
        self.fts_db = fts_db
        self.collection_name = collection_name
        self.model_name = model_name
        self.vector_dims = vector_dims

        self._watcher: Any = None
        self._closed = False
        self._sync_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dirty: bool = False  # whether there are pending file changes to sync

    # ── Public interface ───────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Initialize: full sync + start file watching.
        Aligns with openclaw MemoryIndexManager constructor's ensureWatcher + warmSession.
        """
        logger.info("QdrantMemoryIndexer: starting (working_dir=%s)", self.working_dir)
        self._loop = asyncio.get_event_loop()
        await self._initial_sync()
        self._start_watcher()
        logger.info("QdrantMemoryIndexer: started")

    def close(self) -> None:
        """Stop file watching and release resources."""
        self._closed = True
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher.join(timeout=5)
            self._watcher = None
        logger.info("QdrantMemoryIndexer: closed")

    async def index_file(self, abs_path: Path, force: bool = False) -> None:
        """
        Index a single .md file.
        Aligns with openclaw indexFile (manager-embedding-ops.ts).

        Args:
            abs_path: Absolute file path
            force:    When True, skip hash comparison and force re-index (used by file watcher)
        """
        if not abs_path.exists() or not abs_path.is_file():
            await self.delete_file_chunks(self._rel_path(abs_path))
            return

        rel = self._rel_path(abs_path)
        try:
            content = abs_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read file %s: %s", abs_path, exc)
            return

        # ── Incremental sync: hash comparison, skip if unchanged ─────────────
        current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not force:
            from lightclaw.agent.memory.qdrant_schema import get_indexed_file_hash

            stored_hash = get_indexed_file_hash(self.fts_db, rel)
            if stored_hash == current_hash:
                logger.debug("Incremental sync: skip unchanged file %s", rel)
                return

        raw_chunks = [c for c in _chunk_markdown(content) if c["text"].strip()]
        if not raw_chunks:
            await self.delete_file_chunks(rel)
            return

        embeddings = await self._get_embeddings(rel, [c["text"] for c in raw_chunks])

        # Dated daily memory files (memory/YYYY-MM-DD.md) are intentional
        # per-date historical records — skip cross-file dedup so they are
        # always indexed even when their content overlaps with MEMORY.md.
        _is_dated_memory = bool(re.search(r"(?:^|/)memory/\d{4}-\d{2}-\d{2}\.md$", rel))
        if not _is_dated_memory:
            # Cross-file dedup: filter out chunks that are near-duplicates of
            # content already indexed from OTHER files. Same-file dedup is
            # handled by the delete-then-reinsert pattern below.
            raw_chunks, embeddings = await self._filter_duplicate_chunks(
                rel,
                raw_chunks,
                embeddings,
            )
            if not raw_chunks:
                await self.delete_file_chunks(rel)
                logger.info("Dedup: all chunks in %s are duplicates, removed", rel)
                return

        now_iso = datetime.now(UTC).isoformat()
        qdrant_points, fts_rows = self._build_points_and_rows(rel, raw_chunks, embeddings, now_iso)

        # Delete old Qdrant chunks first (aligns with openclaw clearIndexedFileData)
        await self._delete_from_qdrant(rel)
        # FTS5: delete + insert atomic operation to avoid data loss on crash
        self._replace_fts_rows(rel, fts_rows)
        # Clear hash record then re-write
        from lightclaw.agent.memory.qdrant_schema import delete_indexed_file_hash

        delete_indexed_file_hash(self.fts_db, rel)

        await self._upsert_to_qdrant(rel, qdrant_points)

        # Record file hash (for incremental sync, reuse already-computed current_hash)
        from lightclaw.agent.memory.qdrant_schema import set_indexed_file_hash

        set_indexed_file_hash(self.fts_db, rel, current_hash)
        self._dirty = False  # Sync complete, clear dirty flag

        logger.info("Indexed %s: %d chunks, %d embeddings", rel, len(raw_chunks), len(qdrant_points))

    async def _get_embeddings(self, rel: str, texts: list[str]) -> list[list[float]]:
        """Get text embeddings, returning empty vectors on failure."""
        if self.embedding_fn is None:
            return [[] for _ in texts]
        try:
            return await self.embedding_fn(texts)
        except Exception as exc:
            logger.warning("Embedding failed for %s: %s", rel, exc)
            return [[] for _ in texts]

    async def _filter_duplicate_chunks(
        self,
        rel: str,
        raw_chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> tuple[list[dict[str, Any]], list[list[float]]]:
        """Remove chunks that are near-duplicates of existing indexed content.

        Compares each new chunk against chunks from OTHER files only.
        Same-file dedup is handled by the delete-then-reinsert pattern
        in index_file().

        Returns filtered (chunks, embeddings) tuple.
        """
        from lightclaw.agent.memory.dedup import check_chunk_duplicate, create_dedup_config_from_env

        config = create_dedup_config_from_env()
        if not config.enabled:
            return raw_chunks, embeddings

        kept_chunks: list[dict[str, Any]] = []
        kept_embeddings: list[list[float]] = []
        skipped = 0

        for i, chunk in enumerate(raw_chunks):
            emb = embeddings[i] if i < len(embeddings) else []
            verdict = await check_chunk_duplicate(
                text=chunk["text"],
                embedding=emb,
                exclude_path=rel,
                fts_db=self.fts_db,
                qdrant_client=self.client,
                collection_name=self.collection_name,
                config=config,
            )
            if verdict.action == "skip":
                skipped += 1
                logger.info(
                    "Dedup SKIP chunk %d of %s (%.3f %s sim with %s)",
                    i,
                    rel,
                    verdict.similarity,
                    verdict.method,
                    verdict.best_match_id,
                )
            else:
                if verdict.action == "warn":
                    logger.info(
                        "Dedup WARN chunk %d of %s (%.3f %s sim with %s)",
                        i,
                        rel,
                        verdict.similarity,
                        verdict.method,
                        verdict.best_match_id,
                    )
                kept_chunks.append(chunk)
                kept_embeddings.append(emb)

        if skipped:
            logger.info(
                "Dedup: skipped %d/%d chunks for %s",
                skipped,
                len(raw_chunks),
                rel,
            )

        return kept_chunks, kept_embeddings

    def _build_points_and_rows(
        self,
        rel: str,
        raw_chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        now_iso: str,
    ) -> tuple[list[Any], list[tuple]]:
        """
        Build Qdrant PointStruct list and FTS5 row list.
        Aligns with openclaw indexFile chunk loop logic.
        """
        import uuid

        # FTS-only mode (client=None) doesn't need PointStruct
        if self.client is not None:
            from qdrant_client.models import PointStruct

        source = "memory"
        qdrant_points: list[Any] = []
        fts_rows: list[tuple] = []

        # Evaluate importance for the file (same importance for all chunks in one file)
        from lightclaw.agent.memory.importance import evaluate_importance

        sample_text = raw_chunks[0]["text"] if raw_chunks else ""
        file_importance = evaluate_importance(sample_text, rel, source)

        # For dated daily summaries, re-evaluate per chunk (content varies)
        _is_dated = bool(re.search(r"(?:^|/)memory/\d{4}-\d{2}-\d{2}\.md$", rel))

        for i, chunk in enumerate(raw_chunks):
            chunk_id = _make_chunk_id(
                source, rel, chunk["start_line"], chunk["end_line"], chunk["hash"], self.model_name
            )
            embedding = embeddings[i] if i < len(embeddings) else []
            # FTS5 row (written regardless of embedding availability, supports FTS-only mode)
            fts_rows.append(
                (
                    chunk["text"],
                    chunk_id,
                    rel,
                    source,
                    self.model_name,
                    chunk["start_line"],
                    chunk["end_line"],
                )
            )
            # Qdrant point (only written when client is available and embedding exists)
            if self.client is not None and embedding:
                chunk_importance = evaluate_importance(chunk["text"], rel, source) if _is_dated else file_importance
                payload = {
                    "chunk_id": chunk_id,
                    "path": rel,
                    "source": source,
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "snippet": chunk["text"][:SNIPPET_MAX_CHARS],
                    "created_at": now_iso,
                    "model": self.model_name,
                    "hash": chunk["hash"],
                    # proactive_engine.py fetch_candidates
                    "proactive_enabled": True,
                    "proactive_time_window": "any",
                    "proactive_priority": 1 if source == "memory" else 0,
                    "importance": chunk_importance,
                    "access_count": 0,
                    "last_accessed_at": now_iso,
                }

                # v2: Session/Principal isolation -- add scope metadata
                # For existing memory/*.md and MEMORY.md content, scope as user_memory
                payload["source_type"] = "long_term_memory" if "MEMORY.md" in rel else "daily_memory"
                payload["scope"] = "user_memory"
                payload["session_id"] = ""
                payload["session_key"] = ""
                payload["channel"] = ""
                payload["chat_type"] = ""
                payload["chat_id"] = ""
                payload["sender_id"] = ""
                payload["principal_id"] = ""
                payload["trust_level"] = ""
                payload["user_id"] = ""
                payload["is_raw_session"] = False
                payload["is_tool_output"] = False
                payload["contains_secret"] = False

                # Secret detection
                if _contains_secret(chunk["text"]):
                    payload["contains_secret"] = True
                    logger.info(
                        "Secret detected in indexed chunk: %s line %d",
                        rel,
                        chunk["start_line"],
                    )

                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                qdrant_points.append(PointStruct(id=point_uuid, vector=embedding, payload=payload))

        return qdrant_points, fts_rows

    async def _upsert_to_qdrant(self, rel: str, points: list[Any]) -> None:
        """Batch upsert to Qdrant collection."""
        if not points or self.client is None:
            return
        try:
            await self.client.upsert(collection_name=self.collection_name, points=points)
            logger.debug("Qdrant upsert: %d points for %s", len(points), rel)
        except Exception as exc:
            logger.warning("Qdrant upsert failed for %s: %s", rel, exc)

    def _replace_fts_rows(self, rel: str, rows: list[tuple]) -> None:
        """
        Atomically replace all FTS5 rows for a given file (delete + insert in same transaction).
        Aligns with openclaw indexFile's db.exec("BEGIN") / db.exec("COMMIT") atomic write.

        Uses `with self.fts_db` context manager to auto-handle BEGIN / COMMIT / ROLLBACK,
        preventing permanent data loss if delete succeeds but insert fails.
        """
        try:
            with self.fts_db:
                self.fts_db.execute(
                    "DELETE FROM chunks_fts WHERE path = ?",
                    (rel,),
                )
                if rows:
                    self.fts_db.executemany(
                        "INSERT INTO chunks_fts (text, id, path, source, model, start_line, end_line) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
            logger.debug("FTS5 replace: %d rows for %s", len(rows), rel)
        except Exception as exc:
            logger.warning("FTS5 replace failed for %s: %s", rel, exc)

    async def delete_file_chunks(self, rel_path: str) -> None:
        """
        Delete all chunks for a given file (Qdrant + FTS5).
        Aligns with openclaw deleteFileChunks / clearIndexedFileData.
        """
        await self._delete_from_qdrant(rel_path)
        self._delete_from_fts(rel_path)
        # Also clean up hash record
        from lightclaw.agent.memory.qdrant_schema import delete_indexed_file_hash

        delete_indexed_file_hash(self.fts_db, rel_path)

    async def _delete_from_qdrant(self, rel_path: str) -> None:
        """Delete all chunks for a given path from Qdrant."""
        if self.client is None:
            return
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="path", match=MatchValue(value=rel_path))]),
            )
            logger.debug("Qdrant delete: path=%s", rel_path)
        except Exception as exc:
            logger.warning("Qdrant delete failed for %s: %s", rel_path, exc)

    def _delete_from_fts(self, rel_path: str) -> None:
        """Delete all chunks for a given path from SQLite FTS5."""
        try:
            self.fts_db.execute(
                "DELETE FROM chunks_fts WHERE path = ?",
                (rel_path,),
            )
            self.fts_db.commit()
            logger.debug("FTS5 delete: path=%s", rel_path)
        except Exception as exc:
            logger.warning("FTS5 delete failed for %s: %s", rel_path, exc)

    def read_file(
        self,
        rel_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        """
        Read .md file content under working_dir.
        Aligns with openclaw MemoryIndexManager.readFile.
        """
        abs_path = (self.working_dir / rel_path).resolve()
        # Security check: must be within working_dir
        try:
            abs_path.relative_to(self.working_dir)
        except ValueError:
            raise ValueError(f"Path not allowed: {rel_path}") from None
        if not str(abs_path).endswith(".md"):
            raise ValueError(f"Only .md files are allowed: {rel_path}")
        if not abs_path.exists():
            return ""
        content = abs_path.read_text(encoding="utf-8")
        if offset is None and limit is None:
            return content
        lines = content.split("\n")
        start = max(0, (offset or 1) - 1)
        count = max(1, limit or len(lines))
        return "\n".join(lines[start : start + count])

    async def get_chunk_count(self) -> int:
        """Return the total number of chunks in the Qdrant collection."""
        if self.client is None:
            return 0
        try:
            info = await self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    async def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        """Delete specific chunks from Qdrant and FTS5 by chunk_id.

        Args:
            chunk_ids: List of chunk_id values (SHA256[:16] hex strings).

        Returns:
            Number of chunk_ids that were requested for deletion.
        """
        if not chunk_ids:
            return 0

        # Validate chunk_id format (16-char hex strings)
        import re

        hex16_re = re.compile(r"^[0-9a-f]{16}$")
        valid_ids = [cid for cid in chunk_ids if hex16_re.match(cid)]
        if not valid_ids:
            logger.warning("delete_chunks_by_ids: no valid chunk_ids provided")
            return 0

        # Delete from Qdrant (by chunk_id payload filter)
        await self._delete_from_qdrant_by_chunk_ids(valid_ids)

        # Delete from FTS5 (by id column)
        self._delete_from_fts_by_chunk_ids(valid_ids)

        logger.info("Deleted %d chunks by chunk_id", len(valid_ids))
        return len(valid_ids)

    async def _delete_from_qdrant_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete points from Qdrant where payload.chunk_id is in chunk_ids."""
        if self.client is None or not chunk_ids:
            return
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))]),
            )
            logger.debug("Qdrant delete by chunk_ids: %d ids", len(chunk_ids))
        except Exception as exc:
            logger.warning("Qdrant delete by chunk_ids failed: %s", exc)

    def _delete_from_fts_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete rows from FTS5 where id is in chunk_ids."""
        if not chunk_ids:
            return
        try:
            placeholders = ",".join("?" for _ in chunk_ids)
            self.fts_db.execute(
                f"DELETE FROM chunks_fts WHERE id IN ({placeholders})",
                chunk_ids,
            )
            self.fts_db.commit()
            logger.debug("FTS5 delete by chunk_ids: %d ids", len(chunk_ids))
        except Exception as exc:
            logger.warning("FTS5 delete by chunk_ids failed: %s", exc)

    # ── Internal methods ───────────────────────────────────────────────────

    def _rel_path(self, abs_path: Path) -> str:
        """Convert absolute path to a path relative to working_dir (forward slashes).
        Note: on macOS, /var is a symlink to /private/var, so resolve() both sides before comparing.
        """
        resolved = abs_path.resolve()
        try:
            return str(resolved.relative_to(self.working_dir)).replace("\\", "/")
        except ValueError:
            # fallback: use filename directly
            return str(abs_path).replace("\\", "/")

    async def _initial_sync(self) -> None:
        """Incremental sync at startup.

        Scans MEMORY.md, memory.md, memory/*.md, AND any other .md files
        directly in the workspace root (agent-created topic files).
        Compares file SHA256 hash; only changed files are re-indexed.
        """
        targets: list[Path] = []

        # System-level prompt files that should NOT be indexed (they live
        # in the system prompt, not in the vector search layer).
        _prompt_files = {
            "AGENTS.md",
            "SOUL.md",
            "USER.md",
            "IDENTITY.md",
            "HEARTBEAT.md",
            "BOOTSTRAP.md",
            "PROACTIVE.md",
            "PROACTIVITY_ANALYSIS.md",
            # DAILY_TALK.md is injected directly into the system prompt; no need to index it.
            "DAILY_TALK.md",
        }

        # 1. MEMORY.md / memory.md (long-term memory core)
        for name in ("MEMORY.md", "memory.md"):
            p = self.working_dir / name
            if p.exists() and p.is_file():
                targets.append(p)

        # 2. All .md files in workspace root (agent-created topic files,
        #    e.g. TOOLS.md, custom archives). Excludes prompt
        #    files that are already injected via system prompt.
        for md_file in sorted(self.working_dir.glob("*.md")):
            if md_file.name in _prompt_files:
                continue
            if md_file.name in ("MEMORY.md", "memory.md"):
                continue  # Already added above
            if md_file.is_file():
                targets.append(md_file)

        # 3. memory/ subdirectory (daily summaries, recursively)
        memory_dir = self.working_dir / "memory"
        if memory_dir.exists() and memory_dir.is_dir():
            for md_file in sorted(memory_dir.rglob("*.md")):
                if not _should_ignore_path(str(md_file)):
                    targets.append(md_file)

        logger.info("Incremental sync: %d files to check", len(targets))
        for target in targets:
            try:
                await self.index_file(target, force=False)
            except Exception as exc:
                logger.warning("Initial sync failed for %s: %s", target, exc)
        logger.info("Incremental sync: done (%d files checked)", len(targets))

    def _start_watcher(self) -> None:
        """
        Start file watching (watchdog).
        Aligns with openclaw chokidar watcher (ensureWatcher).
        Watches: MEMORY.md / memory.md / .md files in memory/ directory.
        """
        try:
            from watchdog.events import FileSystemEvent, FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("watchdog not installed; file watching disabled. Install with: pip install watchdog")
            return

        indexer = self

        class _MemoryFileHandler(FileSystemEventHandler):
            def on_modified(self, event: FileSystemEvent) -> None:
                self._handle(event.src_path)

            def on_created(self, event: FileSystemEvent) -> None:
                self._handle(event.src_path)

            def on_deleted(self, event: FileSystemEvent) -> None:
                path_str = str(event.src_path)
                if not path_str.endswith(".md"):
                    return
                rel = indexer._rel_path(Path(path_str))
                loop = indexer._loop
                if loop is None or not loop.is_running():
                    return
                asyncio.run_coroutine_threadsafe(
                    indexer.delete_file_chunks(rel),
                    loop,
                )
                logger.info("File deleted, chunks removed: %s", rel)

            def _is_valid_md_path(self, src_path: str) -> Path | None:
                """Check if path is a valid .md file, return absolute path or None."""
                if not src_path.endswith(".md"):
                    return None
                if _should_ignore_path(src_path):
                    return None
                abs_path = Path(src_path).resolve()
                try:
                    abs_path.relative_to(indexer.working_dir)
                except ValueError:
                    return None
                return abs_path

            def _handle(self, src_path: str) -> None:
                abs_path = self._is_valid_md_path(src_path)
                if abs_path is None:
                    return
                loop = indexer._loop
                if loop is None or not loop.is_running():
                    return
                # Mark dirty; search will trigger incremental sync
                indexer._dirty = True
                # File watcher trigger uses force=True, skip hash comparison and re-index directly
                asyncio.run_coroutine_threadsafe(
                    indexer.index_file(abs_path, force=True),
                    loop,
                )
                logger.info("File changed, re-indexing: %s", src_path)

        observer = Observer()
        handler = _MemoryFileHandler()

        # Watch working_dir root directory (MEMORY.md / memory.md filtered by handler)
        observer.schedule(handler, str(self.working_dir), recursive=False)

        # Watch memory/ directory (recursive)
        memory_dir = self.working_dir / "memory"
        if memory_dir.exists():
            observer.schedule(handler, str(memory_dir), recursive=True)

        observer.start()
        self._watcher = observer
        logger.info("File watcher started for %s", self.working_dir)


# ── Factory function ──────────────────────────────────────────────────────────


async def create_qdrant_indexer(
    working_dir: str,
    embedding_fn: Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]] | None = None,
    qdrant_backend: str = "local",
    qdrant_path: str | None = None,
    qdrant_url: str | None = None,
    collection_name: str = "lightclaw_memory",
    model_name: str = "",
    vector_dims: int = 512,
    data_dir: str | None = None,
) -> QdrantMemoryIndexer:
    """
    Factory function: create and initialize a QdrantMemoryIndexer.
    Aligns with openclaw MemoryIndexManager.get() factory method.

    Args:
        working_dir     — Agent working directory (where MEMORY.md is located)
        embedding_fn    — Async embedding function, accepts list[str] returns list[list[float]]
        qdrant_backend  — "local" | "server" (default "local")
        qdrant_path     — Local mode data directory (default ~/.lightclaw/qdrant_data)
        qdrant_url      — Server mode URL (default http://localhost:6333)
        collection_name — Qdrant collection name
        model_name      — Embedding model name (used for chunk_id generation)
        vector_dims     — Embedding dimensions
        data_dir        — FTS5 database directory (default ~/.lightclaw)
    """
    from qdrant_client import AsyncQdrantClient

    from lightclaw.agent.memory.qdrant_schema import ensure_qdrant_collection, open_fts_db

    # Initialize Qdrant client
    if qdrant_backend == "server":
        url = qdrant_url or "http://localhost:6333"
        client = AsyncQdrantClient(url=url)
        logger.info("Qdrant client: server mode (%s)", url)
    else:
        path = qdrant_path or os.path.expanduser("~/.lightclaw/qdrant_data")
        client = AsyncQdrantClient(path=path)
        logger.info("Qdrant client: local mode (%s)", path)

    # initialze FTS5 DB
    _data_dir = data_dir or os.path.expanduser("~/.lightclaw")
    fts_db = open_fts_db(_data_dir)

    # make sure Qdrant collection exist （payload index included）
    await ensure_qdrant_collection(client, collection_name, vector_dims)

    indexer = QdrantMemoryIndexer(
        working_dir=working_dir,
        embedding_fn=embedding_fn,
        qdrant_client=client,
        fts_db=fts_db,
        collection_name=collection_name,
        model_name=model_name,
        vector_dims=vector_dims,
    )

    return indexer
