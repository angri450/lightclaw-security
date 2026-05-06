"""
Qdrant collection schema + SQLite FTS5 schema.
Aligns with openclaw: src/memory/memory-schema.ts
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import PayloadSchemaType

logger = logging.getLogger(__name__)

# ── SQLite version detection ───────────────────────────────────────────────────
_SQLITE_VER: tuple[int, ...] = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
_TRIGRAM_MIN_VER: tuple[int, ...] = (3, 43, 0)


def _supports_trigram() -> bool:
    """Return True if the linked SQLite supports the trigram tokenizer (>= 3.43.0)."""
    return _SQLITE_VER >= _TRIGRAM_MIN_VER


# ── Constants (aligns with openclaw manager.ts top-level constants) ────────────
COLLECTION_NAME = "lightclaw_memory"
FTS_DB_FILENAME = "memory_fts.db"
SESSIONS_FTS_DB_FILENAME = "sessions_fts.db"

# Payload field names (aligns with openclaw chunks table column names)
PAYLOAD_CHUNK_ID = "chunk_id"  # SHA256[:16], unique identifier
PAYLOAD_PATH = "path"  # relative path, e.g. "memory/2026-03-17.md"
PAYLOAD_SOURCE = "source"  # "memory" | "sessions"
PAYLOAD_START_LINE = "start_line"  # chunk start line (1-indexed)
PAYLOAD_END_LINE = "end_line"  # chunk end line
PAYLOAD_SNIPPET = "snippet"  # text truncated to SNIPPET_MAX_CHARS
PAYLOAD_CREATED_AT = "created_at"  # ISO-8601 timestamp
PAYLOAD_MODEL = "model"  # embedding model name
PAYLOAD_HASH = "hash"  # SHA256[:16] of chunk text

# Payload field names (memory importance scoring, aligns with importance.py)
PAYLOAD_IMPORTANCE = "importance"  # float, 0.0~1.0
PAYLOAD_ACCESS_COUNT = "access_count"  # integer, search-hit accumulator
PAYLOAD_LAST_ACCESSED_AT = "last_accessed_at"  # datetime, last search hit

# Payload field names (proactive outreach, aligns with proactive_engine.py)
PAYLOAD_PROACTIVE_ENABLED = "proactive_enabled"  # bool
PAYLOAD_PROACTIVE_TIME_WINDOW = "proactive_time_window"  # keyword
PAYLOAD_PROACTIVE_LAST_TRIGGERED_AT = "proactive_last_triggered_at"  # datetime
PAYLOAD_PROACTIVE_PRIORITY = "proactive_priority"  # integer
PAYLOAD_PROACTIVE_FILE_LAST_TRIGGERED_AT = "proactive_file_last_triggered_at"  # datetime

# ── v2: Session/Principal isolation payload fields ────────────────────────────
PAYLOAD_SOURCE_TYPE = "source_type"  # "memory_md" | "daily_memory" | "long_term_memory" | "raw_session" | "workspace"
PAYLOAD_SCOPE = "scope"  # "current_session" | "owner_memory" | "user_memory" | "group_memory" | "private_session" | "workspace" | "global"
PAYLOAD_SESSION_ID = "session_id"  # current session ID (empty for workspace-level content)
PAYLOAD_SESSION_KEY = "session_key"  # structured session key
PAYLOAD_CHANNEL = "channel"  # channel name (dashboard, yuanbao, weixin, etc.)
PAYLOAD_CHAT_TYPE = "chat_type"  # private | group | dashboard | cron | proactive
PAYLOAD_CHAT_ID = "chat_id"  # chat identifier (group id, etc.)
PAYLOAD_SENDER_ID = "sender_id"  # sender identifier hash (not raw sender_id for privacy)
PAYLOAD_PRINCIPAL_ID = "principal_id"  # resolved principal identity
PAYLOAD_TRUST_LEVEL = "trust_level"  # owner | trusted | group_member | unknown | system
PAYLOAD_USER_ID = "user_id"  # user identifier
PAYLOAD_IS_RAW_SESSION = "is_raw_session"  # bool: True if content from raw session JSONL
PAYLOAD_IS_TOOL_OUTPUT = "is_tool_output"  # bool: True if content is tool output
PAYLOAD_CONTAINS_SECRET = "contains_secret"  # bool: True if content contains credential


async def ensure_qdrant_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    vector_dims: int,
) -> None:
    """
    Ensure the Qdrant collection exists and create payload indexes.
    Aligns with openclaw: ensureMemoryIndexSchema + idx_chunks_source + idx_chunks_path
    """
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        VectorParams,
    )

    # Check if collection already exists
    existing = await client.get_collections()
    existing_names = {c.name for c in existing.collections}

    if collection_name not in existing_names:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dims,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Qdrant collection '%s' created (dims=%d)", collection_name, vector_dims)
    else:
        # Validate existing collection dimension matches expected dimension.
        # If mismatched (e.g. embedding model changed), recreate the collection.
        info = await client.get_collection(collection_name)
        existing_dims = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing_dims != vector_dims:
            logger.warning(
                "Qdrant collection '%s' dimension mismatch: existing=%d, expected=%d. "
                "Recreating collection to match current embedding model.",
                collection_name,
                existing_dims,
                vector_dims,
            )
            await client.delete_collection(collection_name)
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dims,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Qdrant collection '%s' recreated (dims=%d → %d)",
                collection_name,
                existing_dims,
                vector_dims,
            )
        else:
            logger.debug("Qdrant collection '%s' already exists (dims=%d)", collection_name, vector_dims)

    # Create payload indexes (aligns with openclaw idx_chunks_source + idx_chunks_path)
    # source: keyword index
    await _ensure_payload_index(client, collection_name, PAYLOAD_SOURCE, PayloadSchemaType.KEYWORD)
    # path: keyword index
    await _ensure_payload_index(client, collection_name, PAYLOAD_PATH, PayloadSchemaType.KEYWORD)
    # model: keyword index
    await _ensure_payload_index(client, collection_name, PAYLOAD_MODEL, PayloadSchemaType.KEYWORD)
    # created_at: datetime index (for time decay and proactive outreach filtering)
    await _ensure_payload_index(client, collection_name, PAYLOAD_CREATED_AT, PayloadSchemaType.DATETIME)

    # ── Memory importance scoring payload indexes (aligns with importance.py) ──
    # importance: float index (for pre-filtering low-importance chunks)
    await _ensure_payload_index(client, collection_name, PAYLOAD_IMPORTANCE, PayloadSchemaType.FLOAT)
    # access_count: integer index (for reinforcement tracking)
    await _ensure_payload_index(client, collection_name, PAYLOAD_ACCESS_COUNT, PayloadSchemaType.INTEGER)
    # last_accessed_at: datetime index (for access recency)
    await _ensure_payload_index(client, collection_name, PAYLOAD_LAST_ACCESSED_AT, PayloadSchemaType.DATETIME)

    # ── Proactive outreach payload indexes (aligns with proactive_engine.py _filter_candidates) ──
    # proactive_enabled: keyword index (bool uses keyword index)
    await _ensure_payload_index(client, collection_name, PAYLOAD_PROACTIVE_ENABLED, PayloadSchemaType.KEYWORD)
    # proactive_time_window: keyword index
    await _ensure_payload_index(client, collection_name, PAYLOAD_PROACTIVE_TIME_WINDOW, PayloadSchemaType.KEYWORD)
    # proactive_last_triggered_at: datetime index (for cooldown filtering)
    await _ensure_payload_index(
        client, collection_name, PAYLOAD_PROACTIVE_LAST_TRIGGERED_AT, PayloadSchemaType.DATETIME
    )
    # proactive_priority: integer index
    await _ensure_payload_index(client, collection_name, PAYLOAD_PROACTIVE_PRIORITY, PayloadSchemaType.INTEGER)
    # proactive_file_last_triggered_at: datetime index (file-level cooldown filtering)
    await _ensure_payload_index(
        client, collection_name, PAYLOAD_PROACTIVE_FILE_LAST_TRIGGERED_AT, PayloadSchemaType.DATETIME
    )

    # ── v2: Session/Principal isolation payload indexes ────────────────────────
    await _ensure_payload_index(client, collection_name, PAYLOAD_SOURCE_TYPE, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_SCOPE, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_SESSION_ID, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_CHANNEL, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_CHAT_TYPE, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_PRINCIPAL_ID, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_TRUST_LEVEL, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_IS_RAW_SESSION, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_IS_TOOL_OUTPUT, PayloadSchemaType.KEYWORD)
    await _ensure_payload_index(client, collection_name, PAYLOAD_CONTAINS_SECRET, PayloadSchemaType.KEYWORD)


async def _ensure_payload_index(
    client: AsyncQdrantClient,
    collection_name: str,
    field_name: str,
    schema_type: PayloadSchemaType,
) -> None:
    """Create payload field index (idempotent operation)."""
    try:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
        logger.debug("Payload index created: %s.%s", collection_name, field_name)
    except Exception as exc:
        # Index already exists — Qdrant raises an error, safe to ignore
        msg = str(exc).lower()
        if "already exists" in msg or "conflict" in msg:
            logger.debug("Payload index already exists: %s.%s", collection_name, field_name)
        else:
            logger.warning("Failed to create payload index %s.%s: %s", collection_name, field_name, exc)


def ensure_fts_schema(db: sqlite3.Connection) -> None:
    """
    Ensure the SQLite FTS5 virtual table exists.
    Aligns with openclaw: ensureMemoryIndexSchema chunks_fts section.

    Table structure:
        text        — full-text search content (BM25 indexed)
        id          — chunk_id (UNINDEXED, not part of full-text search)
        path        — relative path (UNINDEXED)
        source      — "memory" | "sessions" (UNINDEXED)
        model       — embedding model name (UNINDEXED)
        start_line  — chunk start line (UNINDEXED)
        end_line    — chunk end line (UNINDEXED)

    When SQLite >= 3.43.0, the trigram tokenizer is used for Chinese substring
    search support.  On older SQLite builds the table falls back to the default
    unicode61 tokenizer — behaviour is identical to the previous implementation.
    """
    tokenize_clause = ', tokenize="trigram"' if _supports_trigram() else ""
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            text,
            id UNINDEXED,
            path UNINDEXED,
            source UNINDEXED,
            model UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED
            {tokenize_clause}
        )
        """
    )
    # File-level hash metadata table (for incremental sync: skip unchanged files)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS file_index_meta (
            path        TEXT PRIMARY KEY,
            file_hash   TEXT NOT NULL,
            indexed_at  TEXT NOT NULL
        )
        """
    )
    db.commit()
    logger.debug("SQLite FTS5 table 'chunks_fts' ensured")


def get_indexed_file_hash(db: sqlite3.Connection, rel_path: str) -> str | None:
    """
    Query the SHA256 hash of an indexed file.
    Returns None if the file has never been indexed.
    """
    row = db.execute(
        "SELECT file_hash FROM file_index_meta WHERE path = ?",
        (rel_path,),
    ).fetchone()
    return row[0] if row else None


def set_indexed_file_hash(db: sqlite3.Connection, rel_path: str, file_hash: str) -> None:
    """
    Update (or insert) the indexed hash record for a file.
    Called after index_file completes successfully.
    """
    from datetime import datetime

    now_iso = datetime.now(UTC).isoformat()
    db.execute(
        """
        INSERT INTO file_index_meta (path, file_hash, indexed_at)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            file_hash  = excluded.file_hash,
            indexed_at = excluded.indexed_at
        """,
        (rel_path, file_hash, now_iso),
    )
    db.commit()


def delete_indexed_file_hash(db: sqlite3.Connection, rel_path: str) -> None:
    """Delete the hash record for a file (called when the file is deleted)."""
    db.execute("DELETE FROM file_index_meta WHERE path = ?", (rel_path,))
    db.commit()


def _needs_trigram_migration(db: sqlite3.Connection) -> bool:
    """Return True if chunks_fts exists but was built without the trigram tokenizer."""
    if not _supports_trigram():
        return False
    row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'").fetchone()
    if row is None:
        # Table does not exist yet — no migration needed.
        return False
    # Check whether the DDL already contains the trigram tokenizer declaration.
    return "trigram" not in str(row[0])


def open_fts_db(data_dir: str | Path) -> sqlite3.Connection:
    """
    Open (or create) the FTS5 database and return a sqlite3.Connection.
    Database file path: {data_dir}/memory_fts.db

    Migration: if the existing chunks_fts table was built without the trigram
    tokenizer and the current SQLite supports trigram (>= 3.43.0), the table is
    dropped and recreated so that Chinese substring search works correctly.
    The file_index_meta table is also cleared so the indexer will re-index all
    files on the next startup.
    """
    db_path = Path(data_dir) / FTS_DB_FILENAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    if _needs_trigram_migration(conn):
        logger.info(
            "SQLite %s supports trigram tokenizer; migrating chunks_fts to trigram. "
            "All memory files will be re-indexed on next startup.",
            sqlite3.sqlite_version,
        )
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        # Clear file hashes so the indexer re-indexes everything.
        conn.execute("DELETE FROM file_index_meta")
        conn.commit()

    ensure_fts_schema(conn)
    return conn


def ensure_sessions_fts_schema(db: sqlite3.Connection) -> None:
    """Ensure the SQLite FTS5 virtual table for session search exists.

    Table structure:
        text        — full-text search content (BM25 indexed)
        session     — session filename (UNINDEXED)
        indexed_at  — ISO-8601 timestamp when this session was indexed (UNINDEXED)

    When SQLite >= 3.43.0, the trigram tokenizer is used for Chinese substring
    search support.  On older builds the table falls back to unicode61.
    """
    tokenize_clause = ', tokenize="trigram"' if _supports_trigram() else ""
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts
        USING fts5(
            text,
            session UNINDEXED,
            indexed_at UNINDEXED
            {tokenize_clause}
        )
        """
    )
    db.commit()
    logger.debug("SQLite FTS5 table 'sessions_fts' ensured")


def _sessions_fts_needs_trigram_migration(db: sqlite3.Connection) -> bool:
    """Return True if sessions_fts exists but was built without the trigram tokenizer."""
    if not _supports_trigram():
        return False
    row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions_fts'").fetchone()
    if row is None:
        return False
    return "trigram" not in str(row[0])


def open_sessions_fts_db(data_dir: str | Path) -> sqlite3.Connection:
    """Open (or create) the sessions FTS5 database and return a sqlite3.Connection.

    Database file path: {data_dir}/sessions_fts.db

    Migration: if the existing sessions_fts table was built without the trigram
    tokenizer and the current SQLite supports trigram (>= 3.43.0), the table is
    dropped and recreated so that Chinese substring search works correctly.
    """
    db_path = Path(data_dir) / SESSIONS_FTS_DB_FILENAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    if _sessions_fts_needs_trigram_migration(conn):
        logger.info(
            "SQLite %s supports trigram tokenizer; migrating sessions_fts to trigram.",
            sqlite3.sqlite_version,
        )
        conn.execute("DROP TABLE IF EXISTS sessions_fts")
        conn.commit()

    ensure_sessions_fts_schema(conn)
    return conn
