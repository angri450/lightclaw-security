"""Session distillation service: sessions/* -> memory/YYYY-MM-DD.md."""

from __future__ import annotations

import contextlib
import datetime
import json
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from lightclaw.agent.memory.query_expansion import extract_keywords
from lightclaw.agent.utils.text_sanitizer import sanitize_summary_text


class SessionDistillationService:
    """Incrementally distill session files into today's daily memory note."""

    def __init__(self) -> None:
        self.interval_hours = 6
        self.max_chars_per_run = 30000
        self.max_sessions_per_run = 8
        self._format_priority = {
            ".jsonl": 0,
        }
        # Lazily opened per working_dir; keyed by working_dir string.
        self._fts_dbs: dict[str, sqlite3.Connection] = {}

    def configure(self, interval_hours: int, max_chars_per_run: int) -> None:
        self.interval_hours = max(1, interval_hours)
        self.max_chars_per_run = max(4000, max_chars_per_run)

    # ── Sessions FTS5 helpers ─────────────────────────────────────────────────

    def _get_fts_db(self, working_dir: str) -> sqlite3.Connection:
        """Return (and lazily open) the sessions FTS5 database for *working_dir*."""
        if working_dir not in self._fts_dbs:
            from lightclaw.agent.memory.qdrant_schema import open_sessions_fts_db

            self._fts_dbs[working_dir] = open_sessions_fts_db(working_dir)
        return self._fts_dbs[working_dir]

    def _fts_index_session(self, working_dir: str, session_name: str, text: str) -> None:
        """Insert or replace a session's full text into the FTS5 index.

        Replaces any existing row for *session_name* so re-indexing is idempotent.
        """
        if not text.strip():
            return
        db = self._get_fts_db(working_dir)
        indexed_at = datetime.datetime.now().isoformat()
        # FTS5 does not support UPDATE; delete then insert for idempotency.
        db.execute("DELETE FROM sessions_fts WHERE session = ?", (session_name,))
        db.execute(
            "INSERT INTO sessions_fts (text, session, indexed_at) VALUES (?, ?, ?)",
            (text, session_name, indexed_at),
        )
        db.commit()

    def _fts_search(
        self,
        working_dir: str,
        keywords: list[str],
        max_results: int,
        context_lines: int,
    ) -> list[dict[str, str]]:
        """Search the sessions FTS5 index and return snippet dicts.

        Returns a list of dicts with keys ``session`` and ``snippet``.
        """
        db = self._get_fts_db(working_dir)
        results: list[dict[str, str]] = []
        seen_sessions: set[str] = set()

        for kw in keywords:
            if len(results) >= max_results:
                break
            # Use FTS5 MATCH for BM25 ranking; wrap keyword in quotes for exact phrase.
            safe_kw = kw.replace('"', '""')
            try:
                rows = db.execute(
                    "SELECT text, session FROM sessions_fts WHERE sessions_fts MATCH ? ORDER BY rank LIMIT ?",
                    (f'"{safe_kw}"', max_results * 2),
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback: trigram not available or syntax error — skip FTS for this keyword.
                continue

            for text, session_name in rows:
                if len(results) >= max_results:
                    break
                if session_name in seen_sessions:
                    continue
                seen_sessions.add(session_name)

                # Extract a context window around the first matching line.
                lines = text.splitlines()
                kw_lower = kw.lower()
                match_idx = next(
                    (i for i, ln in enumerate(lines) if kw_lower in ln.lower()),
                    0,
                )
                start = max(0, match_idx - context_lines)
                end = min(len(lines), match_idx + context_lines + 1)
                snippet = "\n".join(lines[start:end])
                results.append({
                    "session": session_name,
                    "snippet": snippet,
                    "source_type": "session_fts",
                    "scope": "raw_session",
                })

        return results

    def _state_path(self, working_dir: str) -> Path:
        return Path(working_dir) / ".session_distill_state.json"

    def _load_state(self, working_dir: str) -> dict[str, Any]:
        path = self._state_path(working_dir)
        if not path.exists():
            return {"last_run_ts": 0.0, "files": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"last_run_ts": 0.0, "files": {}}
        if not isinstance(data, dict):
            return {"last_run_ts": 0.0, "files": {}}
        data.setdefault("last_run_ts", 0.0)
        data.setdefault("files", {})
        if not isinstance(data["files"], dict):
            data["files"] = {}
        return data

    def _save_state(self, working_dir: str, state: dict[str, Any], logger) -> None:
        path = self._state_path(working_dir)
        try:
            path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            logger.warning("Failed to save session distillation state")

    def should_distill(self, working_dir: str) -> bool:
        state = self._load_state(working_dir)
        last_ts = float(state.get("last_run_ts", 0.0) or 0.0)
        if last_ts <= 0:
            return True
        elapsed_hours = (datetime.datetime.now().timestamp() - last_ts) / 3600
        return elapsed_hours >= self.interval_hours

    @staticmethod
    def _extract_text(block: Any) -> str:
        if isinstance(block, str):
            return block.strip()
        if isinstance(block, list):
            parts: list[str] = []
            for item in block:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            return "\n".join(parts).strip()
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    @classmethod
    def _message_to_line(cls, msg: Any, max_content_chars: int = 2000) -> str:
        if not isinstance(msg, dict):
            return ""

        event_type = str(msg.get("type", "")).lower()
        if event_type == "message":
            message_payload = msg.get("message")
            if not isinstance(message_payload, dict):
                return ""
            role = str(message_payload.get("role", "unknown")).strip() or "unknown"
            content = cls._extract_text(message_payload.get("content"))
            if not content:
                return ""
            # Truncate tool output
            if role in ("tool", "system") or len(content) > max_content_chars:
                original_len = len(content)
                content = content[:max_content_chars]
                if original_len > max_content_chars:
                    content += f"\n[tool output truncated: original_chars={original_len}]"
            return f"{role}: {content}"

        data = msg.get("data", {}) if isinstance(msg.get("data"), dict) else {}
        content = cls._extract_text(data.get("content"))
        if not content:
            return ""

        role = "unknown"
        if "human" in event_type:
            role = "user"
        elif "ai" in event_type:
            role = "assistant"
        # Truncate long content
        if role in ("tool", "system") or len(content) > max_content_chars:
            original_len = len(content)
            content = content[:max_content_chars]
            if original_len > max_content_chars:
                content += f"\n[tool output truncated: original_chars={original_len}]"
        return f"{role}: {content}"

    def _suffix_for_path(self, path: Path) -> str:
        for suffix in self._format_priority:
            if path.name.endswith(suffix):
                return suffix
        return ""

    def _session_base_name(self, path: Path) -> str:
        suffix = self._suffix_for_path(path)
        if suffix:
            return path.name[: -len(suffix)]
        return path.stem

    def _list_candidates(self, sessions_dir: Path) -> list[Path]:
        raw_candidates = list(sessions_dir.glob("*.jsonl"))
        if not raw_candidates:
            return []

        selected: dict[str, Path] = {}
        for path in raw_candidates:
            base_name = self._session_base_name(path)
            current = selected.get(base_name)
            if current is None:
                selected[base_name] = path
                continue
            cur_priority = self._format_priority.get(self._suffix_for_path(current), 99)
            new_priority = self._format_priority.get(self._suffix_for_path(path), 99)
            if new_priority < cur_priority:
                selected[base_name] = path
                continue
            if new_priority == cur_priority and path.stat().st_mtime > current.stat().st_mtime:
                selected[base_name] = path

        return sorted(selected.values(), key=lambda p: p.stat().st_mtime, reverse=True)

    def _load_session_messages(self, fpath: Path, logger: Any) -> list[dict]:
        if fpath.name.endswith(".jsonl"):
            messages: list[dict] = []
            try:
                with open(fpath, encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        event_type = str(event.get("type", "")).lower()
                        # Accept the canonical "message" type as well as
                        # LangChain-style "human" / "ai" / "ai_message" /
                        # "human_message" event types so that _message_to_line
                        # can process all of them.  Previously only "message"
                        # was accepted, silently dropping the other formats.
                        if not event_type:
                            continue
                        if event_type not in {"message"} and "human" not in event_type and "ai" not in event_type:
                            continue
                        messages.append(event)
                return messages
            except OSError:
                logger.debug("Failed to read JSONL session file %s", fpath)
                return []

        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, dict):
            return []
        messages = data.get("messages", [])
        return messages if isinstance(messages, list) else []

    async def distill_sessions_to_daily_md(
        self,
        *,
        working_dir: str,
        language: str,
        logger,
        invoke_llm: Callable[[str, str], Awaitable[str]],
        dedup_summary_against_existing: Callable[[str], Awaitable[str]],
    ) -> str:
        state = self._load_state(working_dir)
        files_state: dict[str, Any] = state.get("files", {})

        sessions_dir = Path(working_dir) / "sessions"
        if not sessions_dir.exists():
            state["last_run_ts"] = datetime.datetime.now().timestamp()
            self._save_state(working_dir, state, logger)
            return ""

        candidates = self._list_candidates(sessions_dir)
        if not candidates:
            state["last_run_ts"] = datetime.datetime.now().timestamp()
            self._save_state(working_dir, state, logger)
            return ""

        source_chunks: list[str] = []
        chars = 0
        processed_sessions = 0
        truncated_sessions = 0
        skipped_sessions = 0

        today_date = datetime.date.today()

        for fpath in candidates:
            if processed_sessions >= self.max_sessions_per_run or chars >= self.max_chars_per_run:
                break

            messages = self._load_session_messages(fpath, logger)
            if not messages:
                continue

            rel_name = fpath.name
            cur_mtime = fpath.stat().st_mtime
            prev_state = files_state.get(rel_name, {})
            prev_count = int(prev_state.get("message_count", 0) or 0)
            prev_mtime = float(prev_state.get("mtime", 0.0) or 0.0)

            # Never-distilled session: only process if it was active today.
            # Historical sessions (mtime on a past date) are silently marked as
            # fully processed so they won't be picked up again, but their
            # content is NOT distilled into today's daily note.
            if not prev_state:
                session_date = datetime.date.fromtimestamp(cur_mtime)
                if session_date < today_date:
                    logger.debug(
                        "Skipping historical session %s (last active %s)",
                        rel_name,
                        session_date,
                    )
                    files_state[rel_name] = {"message_count": len(messages), "mtime": cur_mtime}
                    skipped_sessions += 1
                    continue

            if prev_count >= len(messages) and prev_mtime == cur_mtime:
                files_state[rel_name] = {"message_count": len(messages), "mtime": cur_mtime}
                skipped_sessions += 1
                continue

            new_msgs = messages[prev_count:]
            lines = [self._message_to_line(m) for m in new_msgs]
            lines = [line for line in lines if line]
            if not lines:
                files_state[rel_name] = {"message_count": len(messages), "mtime": cur_mtime}
                skipped_sessions += 1
                continue

            chunk = f"### {rel_name}\n" + "\n".join(lines)
            original_chunk_len = len(chunk)

            # Oversized session detection: if a single session dump exceeds
            # 3× the per-run budget, the file should eventually be processed
            # via chunked distillation rather than one-shot.
            oversized_threshold = self.max_chars_per_run * 3
            if original_chunk_len > oversized_threshold:
                logger.warning(
                    "Oversized session detected: %s (%d chars > %d threshold). "
                    "Consider chunked distillation for this session.",
                    rel_name, original_chunk_len, oversized_threshold,
                )

            # Per-chunk cap: if this session would push us past the
            # budget, truncate it so downstream LLM calls don't blow past
            # the model context window.
            remaining = self.max_chars_per_run - chars
            if len(chunk) > max(remaining, 0):
                if remaining <= 0:
                    break
                logger.warning(
                    "Session chunk truncated: %s | original=%d chars | "
                    "truncated=%d chars | remaining_quota=%d",
                    rel_name, original_chunk_len, remaining, remaining,
                )
                chunk = chunk[:remaining]
                truncated_sessions += 1
            source_chunks.append(chunk)
            chars += len(chunk)
            processed_sessions += 1
            files_state[rel_name] = {"message_count": len(messages), "mtime": cur_mtime}

        if not source_chunks:
            state["files"] = files_state
            state["last_run_ts"] = datetime.datetime.now().timestamp()
            self._save_state(working_dir, state, logger)
            logger.info(
                "Session distillation: no new content | processed=%d skipped=%d truncated=%d",
                processed_sessions, skipped_sessions, truncated_sessions,
            )
            return ""

        combined = "\n\n".join(source_chunks)
        final_payload_chars = len(combined)

        logger.info(
            "Session distillation payload: %d chars | processed=%d skipped=%d truncated=%d",
            final_payload_chars, processed_sessions, skipped_sessions, truncated_sessions,
        )

        from lightclaw.agent.prompt_catalog import get_session_distillation_system_prompt

        system = get_session_distillation_system_prompt(language)

        if language == "zh":
            user_msg = f"## 新增会话片段\n\n{combined}"
        else:
            user_msg = f"## Newly added session snippets\n\n{combined}"

        try:
            result = await invoke_llm(system, user_msg)
            result = sanitize_summary_text(result).strip()
        except Exception:
            logger.warning(
                "Session distillation LLM call failed (non-blocking): "
                "payload=%d chars | processed=%d skipped=%d truncated=%d — "
                "likely exceeds model context window",
                final_payload_chars, processed_sessions, skipped_sessions, truncated_sessions,
            )
            return ""

        if not result or "NOTHING_TO_DISTILL" in result:
            state["files"] = files_state
            state["last_run_ts"] = datetime.datetime.now().timestamp()
            self._save_state(working_dir, state, logger)
            return ""

        result = await dedup_summary_against_existing(result)
        if not result.strip():
            state["files"] = files_state
            state["last_run_ts"] = datetime.datetime.now().timestamp()
            self._save_state(working_dir, state, logger)
            return ""

        memory_dir = Path(working_dir) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        daily_path = memory_dir / f"{today}.md"
        block = (
            f"\n\n---\n\n## Session Distilled ({datetime.datetime.now().strftime('%H:%M:%S')})\n\n{result.strip()}\n"
        )
        try:
            with open(daily_path, "a", encoding="utf-8") as f:
                f.write(block)
        except OSError:
            logger.exception("Failed to append session distillation to %s", daily_path)
            return ""

        state["files"] = files_state
        state["last_run_ts"] = datetime.datetime.now().timestamp()
        self._save_state(working_dir, state, logger)
        logger.info(
            "Session distillation appended %d chars into %s from %d session files",
            len(result),
            daily_path,
            processed_sessions,
        )

        # Index the raw session text into sessions_fts.db so future searches
        # can use BM25 instead of brute-force scanning.
        for chunk in source_chunks:
            # Each chunk starts with "### <filename>\n"; extract name and body.
            first_newline = chunk.find("\n")
            if first_newline == -1:
                continue
            header = chunk[:first_newline].lstrip("# ").strip()
            body = chunk[first_newline + 1 :]
            if header and body.strip():
                try:
                    self._fts_index_session(working_dir, header, body)
                except Exception:
                    logger.debug("sessions_fts: failed to index %s", header)

        return result

    def mark_session_summarized(
        self,
        working_dir: str,
        session_id: str,
        message_count: int,
        logger: Any,
    ) -> None:
        """Mark a session as already summarized by ``summary_memory``.

        When ``summary_memory`` successfully writes a daily note it should call
        this method so that the session distillation pipeline skips the messages
        that have already been processed, preventing duplicate content in the
        daily memory file.

        The ``session_id`` is matched as a substring of the session filename
        (e.g. ``{channel}_{epoch_ms}.jsonl``).

        Args:
            working_dir:   Agent working directory.
            session_id:    Session identifier used to locate the session file.
            message_count: Number of messages already summarized (used as the
                           new ``prev_count`` baseline for incremental distill).
            logger:        Logger instance for diagnostic output.
        """
        if not session_id:
            return

        sessions_dir = Path(working_dir) / "sessions"
        if not sessions_dir.exists():
            return

        state = self._load_state(working_dir)
        files_state: dict[str, Any] = state.get("files", {})

        # Find all session files whose name contains the session_id.
        matched: list[Path] = []
        for suffix in self._format_priority:
            matched.extend(sessions_dir.glob(f"*{session_id}*{suffix}"))

        if not matched:
            logger.debug("mark_session_summarized: no file found for session_id=%s", session_id)
            return

        updated = False
        for fpath in matched:
            rel_name = fpath.name
            prev_state = files_state.get(rel_name, {})
            prev_count = int(prev_state.get("message_count", 0) or 0)

            # Only advance the counter — never go backwards.
            if message_count > prev_count:
                try:
                    cur_mtime = fpath.stat().st_mtime
                except OSError:
                    cur_mtime = prev_state.get("mtime", 0.0)
                files_state[rel_name] = {"message_count": message_count, "mtime": cur_mtime}
                logger.debug(
                    "mark_session_summarized: %s message_count %d -> %d",
                    rel_name,
                    prev_count,
                    message_count,
                )
                updated = True

        if updated:
            state["files"] = files_state
            self._save_state(working_dir, state, logger)

    def search_sessions(
        self,
        working_dir: str,
        query: str,
        max_results: int = 5,
        max_distilled_sessions: int = 100,
        context_lines: int = 3,
        current_session_id: str = "",
        current_session_key: str = "",
        allow_all_sessions: bool = False,
        max_result_chars: int = 2000,
    ) -> list[dict[str, str]]:
        """Search session history using FTS5 (distilled) + brute-force (undistilled).

        Strategy (Plan B):
        - Distilled sessions: BM25 search via sessions_fts.db (fast, full coverage).
        - Undistilled sessions: brute-force line scan (typically 0-2 files, negligible cost).

        Args:
            working_dir:            Agent working directory.
            query:                  Search query; multilingual keyword extraction used for CJK support.
            max_results:            Maximum number of matching snippets to return.
            max_distilled_sessions: Unused — kept for API compatibility; FTS covers all distilled sessions.
            context_lines:          Number of surrounding lines to include per match (brute-force path).

        Returns:
            List of dicts with keys: ``session``, ``snippet``.
        """
        sessions_dir = Path(working_dir) / "sessions"
        if not sessions_dir.exists():
            return []

        # Build keyword list with multilingual support.
        # Preserve temporal keywords so time-sensitive queries (e.g. "yesterday")
        # retain their time dimension in session search.
        extracted = extract_keywords(query, preserve_temporal=True)
        ascii_tokens = [k.lower() for k in query.split() if k.strip()]
        keywords = list(dict.fromkeys(extracted + ascii_tokens))
        if not keywords:
            keywords = [query.lower().strip()]
        if not keywords or not any(keywords):
            return []

        # ── Phase 1: FTS5 search over distilled sessions ──────────────────────
        fts_results: list[dict[str, str]] = []
        with contextlib.suppress(Exception):
            fts_results = self._fts_search(working_dir, keywords, max_results, context_lines)

        fts_sessions: set[str] = {r["session"] for r in fts_results}
        remaining = max_results - len(fts_results)

        # ── Phase 2: brute-force scan of undistilled sessions ─────────────────
        state = self._load_state(working_dir)
        distilled_names: set[str] = set(state.get("files", {}).keys())

        all_candidates = self._list_candidates(sessions_dir)
        undistilled: list[Path] = [
            p for p in all_candidates if p.name not in distilled_names and p.name not in fts_sessions
        ]

        brute_results: list[dict[str, str]] = []
        for fpath in undistilled:
            if len(brute_results) >= remaining:
                break
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            text_lines: list[tuple[int, str]] = []
            for i, raw in enumerate(lines):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                    text = self._message_to_line(obj)
                    if text:
                        text_lines.append((i, text))
                except (json.JSONDecodeError, ValueError):
                    text_lines.append((i, stripped))

            for idx, (_line_no, text) in enumerate(text_lines):
                text_lower = text.lower()
                if not any(kw in text_lower for kw in keywords):
                    continue

                start = max(0, idx - context_lines)
                end = min(len(text_lines), idx + context_lines + 1)
                snippet = "\n".join(tl for _, tl in text_lines[start:end])
                brute_results.append({"session": fpath.name, "snippet": snippet})
                if len(brute_results) >= remaining:
                    break

        # ── Session filter: restrict to current_session when provided ──────────
        all_results = fts_results + brute_results

        if not allow_all_sessions and (current_session_id or current_session_key):
            filtered = []
            for r in all_results:
                session_name = r.get("session", "")
                if current_session_key and current_session_key in session_name:
                    filtered.append(r)
                elif current_session_id and current_session_id in session_name:
                    filtered.append(r)
            if filtered:
                all_results = filtered
            elif current_session_id or current_session_key:
                # No results match current session — return empty
                return []
        elif not allow_all_sessions:
            # No session filter provided, realtime mode: return empty (safe default)
            return []

        # ── Truncate each result snippet ───────────────────────────────────────
        for r in all_results:
            snippet = r.get("snippet", "")
            if len(snippet) > max_result_chars:
                r["snippet"] = snippet[:max_result_chars] + "\n[... session snippet truncated ...]"

        return all_results
