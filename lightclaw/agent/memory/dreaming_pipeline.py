"""Dreaming pipeline: 3-phase nightly memory consolidation (Light/REM/Deep).

Inspired by OpenClaw dreaming.  Extracts long-term memory candidates from
daily conversation logs, cross-references them against existing memory, and
consolidates findings via promotion + decay pruning + managed file updates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightclaw.agent.memory.memory_parser import MemoryParser
from lightclaw.agent.memory.decay_scorer import DecayScorer
from lightclaw.agent.memory.memory_pruner import MemoryPruner
from lightclaw.agent.memory.dreaming_types import DreamCandidate, DreamingResult, PhaseSignal
from lightclaw.config import DreamingConfig

logger = logging.getLogger(__name__)

# ── Secret detection ─────────────────────────────────────────────────

_SECRET_PATTERN = re.compile(
    r"password|passwd|token|api[_-]?key|api_key|secret|bearer|"
    r"auth\.json|login\.txt|sk-[a-zA-Z0-9]{10,}|"
    r"AKIA[A-Z0-9]{16}|Authorization:|Cookie:|Set-Cookie:|"
    r"private_key|access_key|refresh_token|"
    r"-----BEGIN.*?PRIVATE KEY-----",
    re.IGNORECASE,
)

# ── Managed section markers ──────────────────────────────────────────

_MANAGED_BEGIN = "<!-- BEGIN DREAMING MANAGED -->"
_MANAGED_END = "<!-- END DREAMING MANAGED -->"

# ── Ephemeral filter: types that are pure noise ──────────────────────

_EPHEMERAL_TYPES = frozenset({
    "fact_weather", "fact_version", "fact_transient", "meta",
})

# ── JSON extraction: find JSON array in LLM output ───────────────────

_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{\s*\"candidate_id\".*?\}", re.DOTALL)
_JSON_WRAPPED_RE = re.compile(r"\{\s*\"signals\"\s*:\s*(\[.*?\])\s*\}", re.DOTALL)


def _extract_json_objects_aggressive(text: str) -> list[dict] | None:
    """Extract JSON objects with nested braces by scanning for candidate_id keys.

    Uses brace-depth counting to handle nested objects like metadata.
    Returns a list of parsed dicts, or None if nothing found.
    """
    # Find positions of "candidate_id" keys
    marker = '"candidate_id"'
    positions = []
    idx = 0
    while True:
        idx = text.find(marker, idx)
        if idx == -1:
            break
        positions.append(idx)
        idx += len(marker)

    if not positions:
        return None

    results = []
    for pos in positions:
        # Scan backwards from marker to find the opening brace
        start = text.rfind('{', 0, pos)
        if start == -1:
            continue

        # Scan forward from start with brace counting
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            continue

        candidate = text[start:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "candidate_id" in obj:
                results.append(obj)
        except (json.JSONDecodeError, TypeError):
            continue

    return results if results else None


def _extract_json_array(text: str) -> str:
    """Extract JSON array from LLM output, handling multiple formats."""
    # 1. Try ```json ... ``` fenced block first
    fence_matches = _JSON_FENCE_RE.findall(text)
    for block in fence_matches:
        block = block.strip()
        # Check if it's a wrapped {"signals": [...]}
        wrapped_m = _JSON_WRAPPED_RE.search(block)
        if wrapped_m:
            return wrapped_m.group(1)
        # Check if it's a bare JSON array
        arr_m = _JSON_ARRAY_RE.search(block)
        if arr_m:
            return arr_m.group(0)
        # Check if it's a bare JSON object
        obj_m = _JSON_OBJECT_RE.search(block)
        if obj_m:
            return "[" + obj_m.group(0) + "]"

    # 2. Try wrapped {"signals": [...]} in raw text
    wrapped_m = _JSON_WRAPPED_RE.search(text)
    if wrapped_m:
        return wrapped_m.group(1)

    # 3. Try bare JSON array in raw text
    arr_m = _JSON_ARRAY_RE.search(text)
    if arr_m:
        return arr_m.group(0)

    # 4. Try bare JSON object in raw text
    obj_m = _JSON_OBJECT_RE.search(text)
    if obj_m:
        return "[" + obj_m.group(0) + "]"

    return text


class DreamingPipeline:
    """Orchestrates Light → REM → Deep phases for nightly memory consolidation."""

    def __init__(
        self,
        working_dir: str | Path,
        *,
        invoke_llm: Callable[[str, str], Awaitable[str]],
        language: str = "zh",
        config: DreamingConfig,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self._root = Path(working_dir)
        self._invoke_llm = invoke_llm
        self._language = language
        self._config = config
        self._embedding_fn = embedding_fn
        self._memory_path = self._root / "MEMORY.md"
        self._daily_talk_path = self._root / "DAILY_TALK.md"
        self._memory_dir = self._root / "memory"
        self._dreams_dir = self._memory_dir / ".dreams"
        self._pruned_dir = self._memory_dir / ".pruned"
        self._candidates_dir = self._dreams_dir / "candidates"
        self._signals_dir = self._dreams_dir / "signals"
        self._reinforcement_path = self._dreams_dir / "reinforcement.jsonl"
        self._conflicts_path = self._dreams_dir / "conflicts.jsonl"

    # ── public API ────────────────────────────────────────────────────

    async def run(self, *, dry_run: bool = True) -> DreamingResult:
        started_at_ts = time.time()
        started_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today = datetime.now().strftime("%Y-%m-%d")

        result = DreamingResult(
            date=today,
            dry_run=dry_run,
            started_at=started_at_str,
            ended_at="",
            light_candidates_total=0,
            light_candidates_kept=0,
            light_secrets_filtered=0,
            light_ephemeral_filtered=0,
        )

        try:
            # ── Light Phase ──
            candidates = await self._run_light_phase()
            total = len(candidates)

            candidates, secrets_count = self._filter_secrets(candidates)
            candidates, ephemeral_count = self._filter_ephemeral(candidates)

            result.light_candidates_total = total
            result.light_candidates_kept = len(candidates)
            result.light_secrets_filtered = secrets_count
            result.light_ephemeral_filtered = ephemeral_count

            if not candidates:
                logger.info("Dreaming Light Phase: no candidates to process")
                result.ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result.duration_seconds = round(time.time() - started_at_ts, 1)
                result.report_text = self._generate_report(result)
                return result

            # ── REM Phase ──
            signals = await self._run_rem_phase(candidates)
            result.rem_signals = signals

            # ── Persist evidence (non-dry-run only) ──
            if not dry_run:
                self._persist_candidates(candidates, "full")
                self._persist_signals(signals)
                self._write_reinforcement(candidates, signals)
                self._write_conflicts(candidates, signals)

            # ── Deep Phase ──
            deep = await self._run_deep_phase(candidates, signals, dry_run=dry_run)
            result.promoted_count = deep.get("promoted_count", 0)
            result.promoted_entries = deep.get("promoted_entries", [])
            result.pruned_count = deep.get("pruned_count", 0)
            result.pruned_entries = deep.get("pruned_entries", [])
            result.updated_files = deep.get("updated_files", [])

        except Exception:
            logger.exception("Dreaming pipeline failed")
            result.errors.append("Dreaming pipeline encountered an unexpected error")

        result.ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result.duration_seconds = round(time.time() - started_at_ts, 1)
        result.report_text = self._generate_report(result)

        if not dry_run:
            if self._config.write_dream_report:
                self._write_dream_report(result)
            self._write_status(result)

        return result

    # ── Light-only ingest (no REM, no Deep, no writes except candidates) ──

    async def run_light_only(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Run only Light Phase: scan sources, extract candidates, persist to .dreams/.

        No REM. No Deep. No MEMORY.md / USER.md / TOOLS.md / AGENTS.md / HEARTBEAT.md updates.
        """
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        started_ts = time.time()
        today = datetime.now().strftime("%Y-%m-%d")

        info: dict[str, Any] = {
            "date": today,
            "dry_run": dry_run,
            "started_at": started_at,
            "ended_at": "",
            "run_type": "light_ingest",
            "candidates_total": 0,
            "candidates_kept": 0,
            "secrets_filtered": 0,
            "ephemeral_filtered": 0,
            "duration_seconds": 0.0,
        }

        try:
            candidates = await self._run_light_phase()
            total = len(candidates)

            candidates, secrets_count = self._filter_secrets(candidates)
            candidates, ephemeral_count = self._filter_ephemeral(candidates)

            info["candidates_total"] = total
            info["candidates_kept"] = len(candidates)
            info["secrets_filtered"] = secrets_count
            info["ephemeral_filtered"] = ephemeral_count

            if candidates and not dry_run:
                self._persist_candidates(candidates, "light-ingest")
                info["candidates_written"] = True

            if not dry_run:
                self._write_light_ingest_status(info)

        except Exception:
            logger.exception("Light-only ingest failed")
            info.setdefault("errors", []).append("Light-only ingest encountered an unexpected error")

        info["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info["duration_seconds"] = round(time.time() - started_ts, 1)
        return info

    def _write_light_ingest_status(self, info: dict[str, Any]) -> None:
        self._dreams_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "last_light_ingest_at": info["started_at"],
            "last_light_ingest_success_at": info["ended_at"],
            "light_ingest_candidates_total": info["candidates_total"],
            "light_ingest_candidates_kept": info["candidates_kept"],
            "light_ingest_secrets_filtered": info["secrets_filtered"],
            "light_ingest_ephemeral_filtered": info["ephemeral_filtered"],
            "light_ingest_duration_seconds": info["duration_seconds"],
        }
        # Update existing status.json to include light-ingest fields
        existing = {}
        status_path = self._dreams_dir / "status.json"
        if status_path.is_file():
            try:
                existing = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, TypeError):
                pass
        existing.update(status)
        self._atomic_write(status_path, json.dumps(existing, indent=2, ensure_ascii=False))

    async def _run_light_phase(self) -> list[DreamCandidate]:
        """Scan DAILY_TALK.md + recent memory/*.md, extract candidates via LLM."""
        sources: list[tuple[str, str, str]] = []  # (source_file, source_date, text)

        # 1. DAILY_TALK.md
        if self._daily_talk_path.is_file():
            raw = self._daily_talk_path.read_text(encoding="utf-8")
            body = self._daily_body(raw)
            if body.strip():
                truncated = body[:self._config.max_daily_talk_chars]
                sources.append(("DAILY_TALK.md", self._daily_date(raw) or "", truncated))

        # 2. memory/*.md — recent N days, exclude .pruned/ and .dreams/
        if self._memory_dir.is_dir():
            exclude_dirs = {".pruned", ".dreams"}
            now = datetime.now(timezone.utc)
            recent_files: list[tuple[str, Path]] = []
            for f in self._memory_dir.iterdir():
                if not f.is_file() or not f.suffix == ".md":
                    continue
                if any(p in exclude_dirs for p in f.parts[len(self._memory_dir.parts):]):
                    continue
                date_str = f.stem
                try:
                    fd = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if (now - fd).days <= self._config.recent_days:
                        recent_files.append((date_str, f))
                except ValueError:
                    continue
            recent_files.sort(key=lambda x: x[0], reverse=True)
            for date_str, fpath in recent_files[:self._config.recent_days]:
                text = fpath.read_text(encoding="utf-8")
                truncated = text[:self._config.max_recent_memory_chars]
                sources.append((fpath.name, date_str, truncated))

        if not sources:
            return []

        # Build LLM prompt — one call per source to keep output tractable
        from lightclaw.agent.prompt_catalog import get_daily_talk_summarize_prompt

        all_candidates: list[DreamCandidate] = []
        system_base = get_daily_talk_summarize_prompt(self._language)
        system = (
            system_base
            + "\n\n请以 JSON 数组输出。每个元素必须包含字段: "
            + "type(entry_type), persistence, text(原文片段), summary(LLM摘要), keywords(关键词列表)。"
            + "type 选项: preference, constraint, decision, todo, config, fact。"
            + "persistence 选项: permanent, stable, transient, ephemeral。"
            + "过滤: 天气、余额、版本号、file_id、boilerplate模板示例、一次性路径等临时信息。"
        )

        for source_file, source_date, text in sources:
            try:
                user = f"源文件: {source_file}\n日期: {source_date}\n\n{text[:8000]}"
                resp = await self._invoke_llm(system, user)
                candidates = self._parse_light_llm_output(resp, source_file, source_date)
                all_candidates.extend(candidates)
                logger.info("Light Phase: %s → %d candidates", source_file, len(candidates))
            except Exception:
                logger.exception("Light Phase LLM failed for %s", source_file)

        return all_candidates

    @staticmethod
    def _daily_date(text: str) -> str:
        m = re.search(r"daily_talk_date:\s*(\d{4}-\d{2}-\d{2})", text)
        return m.group(1) if m else ""

    @staticmethod
    def _daily_body(text: str) -> str:
        body = re.sub(r"<!--\s*daily_talk_date:.*?-->", "", text, count=1)
        lines = body.splitlines()
        if lines and lines[0].strip() == "# DAILY_TALK":
            lines = lines[1:]
        return "\n".join(lines).strip()

    @staticmethod
    def _parse_light_llm_output(
        response: str, source_file: str, source_date: str,
    ) -> list[DreamCandidate]:
        candidates: list[DreamCandidate] = []
        try:
            json_text = _extract_json_array(response)
            items = json.loads(json_text)
            if not isinstance(items, list):
                return candidates
        except (json.JSONDecodeError, TypeError):
            logger.debug("Light Phase: failed to parse LLM JSON for %s", source_file)
            return candidates

        for item in items:
            if not isinstance(item, dict):
                continue
            raw_text = str(item.get("text", "")).strip()
            if not raw_text:
                continue
            normalized = raw_text.strip()
            c = DreamCandidate(
                id=f"{source_file}:{_hash_8(normalized)}",
                source_file=source_file,
                source_date=source_date,
                raw_text=raw_text,
                normalized_text=normalized,
                summary=str(item.get("summary", raw_text[:100])).strip(),
                entry_type=str(item.get("type", "fact")).strip().lower(),
                persistence=str(item.get("persistence", "transient")).strip().lower(),
                keywords=[
                    kw.strip()
                    for kw in item.get("keywords", [])
                    if isinstance(kw, str) and kw.strip()
                ],
                confidence=float(item.get("confidence", 0.5)),
                evidence_hash=_hash_16(normalized),
            )
            candidates.append(c)

        return candidates

    # ── Secret filter ─────────────────────────────────────────────────

    @staticmethod
    def _filter_secrets(
        candidates: list[DreamCandidate],
    ) -> tuple[list[DreamCandidate], int]:
        filtered: list[DreamCandidate] = []
        dropped = 0
        for c in candidates:
            if _SECRET_PATTERN.search(c.raw_text) or _SECRET_PATTERN.search(c.summary):
                c.has_secret = True
                dropped += 1
                logger.info("Secret filtered: %s", c.id)
            else:
                filtered.append(c)
        return filtered, dropped

    # ── Ephemeral filter ──────────────────────────────────────────────

    @staticmethod
    def _filter_ephemeral(
        candidates: list[DreamCandidate],
    ) -> tuple[list[DreamCandidate], int]:
        filtered: list[DreamCandidate] = []
        dropped = 0
        for c in candidates:
            if c.persistence == "ephemeral" or c.entry_type in _EPHEMERAL_TYPES:
                dropped += 1
            else:
                filtered.append(c)
        return filtered, dropped

    # ── REM Phase ─────────────────────────────────────────────────────

    async def _run_rem_phase(
        self, candidates: list[DreamCandidate],
    ) -> list[PhaseSignal]:
        """Cross-reference candidates against existing memory and config files."""
        # Load reference corpora in parallel
        memory_entries_text, user_md_text, tools_md_text, agents_md_text, proactivity_text = \
            await asyncio.gather(
                asyncio.to_thread(self._load_memory_entries_text),
                asyncio.to_thread(self._read_file_if_exists, self._root / "USER.md"),
                asyncio.to_thread(self._read_file_if_exists, self._root / "TOOLS.md"),
                asyncio.to_thread(self._read_file_if_exists, self._root / "AGENTS.md"),
                asyncio.to_thread(self._read_file_if_exists, self._root / "PROACTIVITY_ANALYSIS.md"),
            )

        if not candidates:
            return []

        # Build prompt
        cand_lines = []
        for i, c in enumerate(candidates):
            cand_lines.append(
                f"[{i}] id={c.id} type={c.entry_type} pers={c.persistence} "
                f"src={c.source_file} date={c.source_date}\n"
                f"    summary={c.summary}\n"
                f"    keywords={', '.join(c.keywords[:8])}"
            )
        cand_text = "\n\n".join(cand_lines)

        system = (
            "你是记忆关联分析助手。分析候选长期记忆条目与现有记忆/配置的关联。\n"
            "对每个候选进行分析，即使结论是 no_action 也必须输出一个分析对象。\n"
            "\n"
            "【重要】只输出 JSON 数组，不要任何解释、注释或 Markdown 格式。\n"
            "不要使用 ```json 代码块包裹。直接输出 JSON 数组。\n"
            "\n"
            "输出格式（JSON 数组，每个元素对应一个候选条目）：\n"
            "[\n"
            "  {\n"
            '    "candidate_id": "对应的候选ID",\n'
            '    "signal_type": "信号类型(见下方选项)",\n'
            '    "score_delta": 0.0,\n'
            '    "should_promote": false,\n'
            '    "confidence": 0.0,\n'
            '    "reason": "分析理由(一句话)",\n'
            '    "metadata": {\n'
            '      "duplicate_of": "重复条目的引用文本(无则空字符串)",\n'
            '      "conflict_with": "冲突条目的引用文本(无则空字符串)",\n'
            '      "pattern_category": "模式类别(无则空字符串)",\n'
            '      "cross_ref_count": 0,\n'
            '      "target_file": "需要更新的文件名(无则空字符串)"\n'
            '    }\n'
            "  }\n"
            "]\n"
            "\n"
            "signal_type 选项：\n"
            "- repeated_across_days: 候选在多个日期反复出现 → score_delta 0.10~0.20, should_promote=true\n"
            "- strengthens_existing_memory: 候选与已有记忆一致/加强 → score_delta 0.05~0.15, should_promote=true\n"
            "- user_profile_update_needed: 候选提供了新的用户偏好/习惯 → score_delta 0.10, should_promote=true\n"
            "- tool_rule_update_needed: 候选揭示了工具使用规则或偏好 → score_delta 0.05, should_promote=true\n"
            "- agent_rule_update_needed: 候选涉及 agent 行为规则 → score_delta 0.05, should_promote=true\n"
            "- proactivity_update_needed: 候选涉及主动行为模式 → score_delta 0.05, should_promote=true\n"
            "- conflicts_with_existing_memory: 候选与已有记忆冲突 → score_delta=0.0, should_promote=false\n"
            "- no_action: 候选无需特殊处理 → score_delta=0.0, should_promote=false\n"
            "\n"
            "pattern_category 选项（填入 metadata.pattern_category）：\n"
            "- recurring_topic: 反复出现的话题\n"
            "- new_preference: 新发现的偏好\n"
            "- new_constraint: 新发现的约束\n"
            "- tool_usage: 工具使用模式\n"
            "\n"
            "评分规则：\n"
            "- 每个候选都必须有分析结果，最差也是 no_action\n"
            "- score_delta 必须在上述范围内，confidence 反映判断确信度 0.0~1.0\n"
            "- should_promote 为 true 表示建议提升到长期记忆\n"
            "- 输出数组长度必须等于候选条目数量\n"
        )

        user = (
            f"--- 候选长期记忆条目 ---\n{cand_text}\n\n"
            f"--- 现有 MEMORY.md ---\n{memory_entries_text[:5000]}\n\n"
            f"--- USER.md ---\n{user_md_text[:2000]}\n\n"
            f"--- TOOLS.md ---\n{tools_md_text[:2000]}\n\n"
            f"--- AGENTS.md ---\n{agents_md_text[:2000]}\n\n"
            f"--- PROACTIVITY_ANALYSIS.md ---\n{proactivity_text[:2000]}\n"
        )

        try:
            resp = await self._invoke_llm(system, user)
            signals = self._parse_rem_llm_output(resp)
            # Log signal type distribution
            type_counts: dict[str, int] = {}
            for s in signals:
                t = s.signal_type or "empty"
                type_counts[t] = type_counts.get(t, 0) + 1
            logger.info("REM Phase: %d signals from %d candidates — %s", len(signals), len(candidates), dict(sorted(type_counts.items())))
            return signals
        except Exception:
            logger.exception("REM Phase LLM failed")
            return []

    def _load_memory_entries_text(self) -> str:
        if not self._memory_path.is_file():
            return ""
        parser = MemoryParser(self._root)
        result = parser.parse()
        lines = []
        for e in result.entries[:50]:
            lines.append(f"[{e.entry_type}|{e.persistence}] {e.raw_text[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _read_file_if_exists(path: Path) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")[:5000]

    @staticmethod
    def _parse_rem_llm_output(response: str) -> list[PhaseSignal]:
        signals: list[PhaseSignal] = []
        parse_error: str | None = None

        # Stage 1: try standard extraction + normal JSON parse
        items = None
        raw_json_text = response
        try:
            json_text = _extract_json_array(response)
            parsed = json.loads(json_text)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                if "signals" in parsed and isinstance(parsed["signals"], list):
                    items = parsed["signals"]
                else:
                    items = [parsed]  # single object → wrap
            raw_json_text = json_text
        except (json.JSONDecodeError, TypeError) as e:
            parse_error = f"primary parse failed: {e}"
            logger.warning("REM Phase: primary JSON parse failed, attempting fallback extraction")

        # Stage 2: fallback — aggressive JSON block extraction
        if items is None:
            try:
                # Find all JSON objects with balanced braces by scanning for candidate_id
                objects = _extract_json_objects_aggressive(response)
                if objects:
                    items = objects
                    parse_error = (parse_error or "") + f"; recovered {len(items)} objects via aggressive extraction"
                    logger.info("REM Phase: recovered %d objects via aggressive extraction", len(items))
            except Exception:
                pass

        # Stage 3: last resort — try raw response as-is
        if items is None:
            try:
                direct = json.loads(response.strip())
                if isinstance(direct, list):
                    items = direct
                elif isinstance(direct, dict):
                    if "signals" in direct and isinstance(direct["signals"], list):
                        items = direct["signals"]
                    else:
                        items = [direct]
                parse_error = (parse_error or "") + "; parsed raw response directly"
            except (json.JSONDecodeError, TypeError):
                pass

        if items is None:
            logger.error("REM Phase: all JSON extraction attempts failed for response[:200]: %s", response[:200])
            # Return a parse_error signal so the pipeline knows something went wrong
            signals.append(PhaseSignal(
                candidate_id="parse_error",
                phase="rem",
                signal_type="parse_error",
                reason=f"Failed to parse REM LLM output. First 200 chars: {response[:200]}",
                parse_error=parse_error or "all extraction methods failed",
            ))
            return signals

        # Build signals from items
        for item in items:
            if not isinstance(item, dict):
                continue

            # Extract metadata sub-object
            meta = item.get("metadata")
            if isinstance(meta, dict):
                metadata = dict(meta)
                duplicate_of = str(meta.get("duplicate_of", "") or "")
                conflict_with = str(meta.get("conflict_with", "") or "")
                pattern_category = str(meta.get("pattern_category", "") or "")
                cross_ref_count = int(meta.get("cross_ref_count", 0))
            else:
                metadata = None
                # Fallback: read from flat fields (old prompt format)
                duplicate_of = item.get("duplicate_of")
                conflict_with = item.get("conflict_with")
                pattern_category = item.get("pattern_category")
                cross_ref_count = int(item.get("cross_ref_count", 0))

            candidate_id = str(item.get("candidate_id", ""))
            signal_type = str(item.get("signal_type", ""))
            score_delta = float(item.get("score_delta", 0.0))
            should_promote = bool(item.get("should_promote", False))
            confidence = float(item.get("confidence", 0.0))
            reason = str(item.get("reason", ""))

            # Validate signal_type against known values
            valid_signal_types = {
                "repeated_across_days", "conflicts_with_existing_memory",
                "strengthens_existing_memory", "user_profile_update_needed",
                "tool_rule_update_needed", "agent_rule_update_needed",
                "proactivity_update_needed", "no_action", "parse_error",
            }
            if signal_type and signal_type not in valid_signal_types:
                logger.debug("REM Phase: unknown signal_type '%s' for %s, defaulting to no_action", signal_type, candidate_id)
                signal_type = "no_action"

            signals.append(PhaseSignal(
                candidate_id=candidate_id,
                phase="rem",
                signal_type=signal_type,
                duplicate_of=duplicate_of if duplicate_of else None,
                conflict_with=conflict_with if conflict_with else None,
                pattern_category=pattern_category if pattern_category else None,
                cross_ref_count=cross_ref_count,
                confidence=confidence,
                should_promote=should_promote,
                score_delta=score_delta,
                reason=reason,
                metadata=metadata,
                parse_error=parse_error if items is not None and parse_error else None,
            ))

        return signals

    # ── Deep Phase ────────────────────────────────────────────────────

    async def _run_deep_phase(
        self,
        candidates: list[DreamCandidate],
        signals: list[PhaseSignal], *,
        dry_run: bool,
    ) -> dict[str, Any]:
        deep: dict[str, Any] = {
            "promoted_count": 0,
            "promoted_entries": [],
            "pruned_count": 0,
            "pruned_entries": [],
            "updated_files": [],
        }

        signal_map: dict[str, PhaseSignal] = {s.candidate_id: s for s in signals}

        # Task A: Archive DAILY_TALK + rollover
        if not dry_run:
            await self._deep_archive_daily_talk()
            deep["updated_files"].append("DAILY_TALK.md archived")

        # Task B: Promote candidates to MEMORY.md
        promoted = self._deep_promote_candidates(candidates, signal_map, dry_run=dry_run)
        deep["promoted_count"] = promoted["count"]
        deep["promoted_entries"] = promoted["entries"]
        if not dry_run and promoted["count"] > 0:
            deep["updated_files"].append("MEMORY.md (promoted)")

        # Task C: Decay pruning
        if not dry_run:
            prune = await self._deep_prune()
            deep["pruned_count"] = prune["count"]
            deep["pruned_entries"] = prune["entries"]
            if prune["count"] > 0:
                deep["updated_files"].append("MEMORY.md (pruned)")

        # Task D: Distillation (migrated from heartbeat)
        if not dry_run:
            await self._deep_distillation()
            deep["updated_files"].append("distillation checked")

        # Task E–I: Managed section updates
        if not dry_run:
            updated = await self._deep_update_managed_files(signals, candidates)
            deep["updated_files"].extend(updated)

        return deep

    # ── Deep: Task A — DAILY_TALK archive ─────────────────────────────

    async def _deep_archive_daily_talk(self) -> None:
        if not self._daily_talk_path.is_file():
            return
        daily_text = self._daily_talk_path.read_text(encoding="utf-8")
        body = self._daily_body(daily_text).strip()
        date_str = self._daily_date(daily_text) or datetime.now().strftime("%Y-%m-%d")
        if not body:
            return
        archive_path = self._memory_dir / f"{date_str}.md"
        self._atomic_write(archive_path, body + "\n")
        logger.info("Archived DAILY_TALK.md to memory/%s.md", date_str)
        # Rollover
        try:
            from lightclaw.app.runner.session.daily_talk_store import DailyTalkStore
            store = DailyTalkStore(working_dir=self._root)
            await store.maybe_rollover(model=None, language=self._language)
        except Exception:
            logger.exception("Daily talk rollover failed during dreaming")

    # ── Deep: Task B — Promote candidates ─────────────────────────────

    def _deep_promote_candidates(
        self,
        candidates: list[DreamCandidate],
        signal_map: dict[str, PhaseSignal], *,
        dry_run: bool,
    ) -> dict[str, Any]:
        promoted_entries: list[dict[str, Any]] = []
        pending_review_entries: list[dict[str, Any]] = []
        new_lines: list[str] = []

        for c in candidates:
            if c.has_secret:
                continue

            # v2: Promotion security — check speaker trust_level
            trust = c.trust_level or "unknown"
            is_group = c.source_chat_type == "group"

            # Unknown/group_member speakers → pending_review only (not owner memory)
            from lightclaw.constant import UNKNOWN_USER_MEMORY_PROMOTION_ENABLED, GROUP_MEMORY_ENABLED
            if trust in ("unknown", "group_member") and not UNKNOWN_USER_MEMORY_PROMOTION_ENABLED:
                logger.info(
                    "Dreaming promotion: skipped %s — trust_level=%s speaker (not owner)",
                    c.id, trust,
                )
                continue

            # Group chat content goes to group_memory, not owner private memory
            if is_group and not trust == "owner" and not GROUP_MEMORY_ENABLED:
                logger.info(
                    "Dreaming promotion: skipped %s — group content from non-owner (group_memory disabled)",
                    c.id,
                )
                continue

            sig = signal_map.get(c.id)
            should_promote = sig.should_promote if sig else False
            score_delta = sig.score_delta if sig else 0.0
            promotion_score = (
                c.confidence * 0.5
                + (1.0 if should_promote else 0.0) * 0.3
                + score_delta * 0.2
            )

            if promotion_score < 0.6:
                continue

            # Rehydrate verification
            if not self._rehydrate_verify(c):
                logger.info("Rehydrate failed for %s — evidence gone", c.id)
                continue

            entry_hash = c.evidence_hash[:8] or _hash_8(c.normalized_text)
            # v2: Add speaker attribution to promoted entry
            attribution = ""
            if trust in ("unknown", "group_member"):
                attribution = f"  speaker: {trust} (pending review)\n"
            elif is_group:
                attribution = f"  speaker: owner (from group: {c.source_chat_id})\n"
            new_lines.append(
                f"- [id:{entry_hash}] {c.summary}\n"
                f"  type: {c.entry_type}\n"
                f"  persistence: {c.persistence}\n"
                f"  created_at: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"  source: {c.source_file}\n"
                f"  evidence_hash: {c.evidence_hash}\n"
                f"{attribution}"
            )
            promoted_entries.append({
                "id": c.id,
                "summary": c.summary,
                "entry_type": c.entry_type,
                "persistence": c.persistence,
                "promotion_score": round(promotion_score, 3),
                "source_file": c.source_file,
                "trust_level": trust,
                "chat_type": c.source_chat_type,
            })

        if new_lines and not dry_run:
            existing = ""
            if self._memory_path.is_file():
                existing = self._memory_path.read_text(encoding="utf-8").rstrip()
            prefix = "\n\n## Dreaming Promoted\n\n" if existing else "# MEMORY\n\n"
            self._atomic_write(self._memory_path, existing + prefix + "\n".join(new_lines) + "\n")
            logger.info("Promoted %d candidates to MEMORY.md", len(promoted_entries))

        return {"count": len(promoted_entries), "entries": promoted_entries}

    def _rehydrate_verify(self, c: DreamCandidate) -> bool:
        """Re-read source file and verify the original fragment still exists."""
        source_path = self._root / c.source_file
        if not source_path.is_file():
            return False

        try:
            content = source_path.read_text(encoding="utf-8")
        except Exception:
            return False

        # Look for the normalized text fragment
        if c.normalized_text in content:
            return True

        # Fuzzy: look for keywords overlap
        if c.keywords:
            matches = sum(1 for kw in c.keywords if kw in content)
            if matches >= max(1, len(c.keywords) // 2):
                return True

        return False

    # ── Deep: Task C — Decay pruning ──────────────────────────────────

    async def _deep_prune(self) -> dict[str, Any]:
        if not self._memory_path.is_file():
            return {"count": 0, "entries": []}

        parser = MemoryParser(self._root)
        result = parser.parse()
        total = len(result.entries)
        if total == 0:
            return {"count": 0, "entries": []}

        # Dynamic threshold
        effective_threshold = max(
            self._config.dynamic_threshold_base,
            min(
                self._config.dynamic_threshold_max,
                self._config.dynamic_threshold_base + (total / 100) * 0.05,
            ),
        )

        # Score entries via DecayScorer
        scorer = DecayScorer(self._root, embedding_fn=self._embedding_fn)
        try:
            scored = await scorer.score(invoke_llm=self._invoke_llm, skip_llm=False)
        except Exception:
            logger.exception("Decay scoring failed in dreaming")
            return {"count": 0, "entries": []}

        # Collect prune-eligible entries (exclude protected)
        protected = set(self._config.protected_types)
        prunable = []
        for e in scored.entries:
            if e.entry_type in protected or e.persistence == "permanent":
                continue
            if e.final_score < effective_threshold:
                prunable.append(e)

        if not prunable:
            return {"count": 0, "entries": []}

        # Single-run cap: max_prune_ratio × total
        max_prune = max(1, int(total * self._config.max_prune_ratio))
        prunable.sort(key=lambda e: e.final_score)
        to_prune = prunable[:max_prune]

        # Archive + rewrite using existing MemoryPruner
        archived = []
        if to_prune:
            pruned_text = self._build_pruned_text(to_prune)
            self._write_prune_archive(pruned_text)

            # Remove pruned entries from MEMORY.md
            pruned_ids = {id(e) for e in to_prune}
            new_lines = []
            for line in self._memory_path.read_text(encoding="utf-8").splitlines(keepends=True):
                # Simple heuristic: if the line contains pruned text, skip
                should_skip = False
                for e in to_prune:
                    sig = e.raw_text.strip()[:80]
                    if sig and sig in line:
                        should_skip = True
                        break
                if not should_skip:
                    new_lines.append(line)

            self._atomic_write(self._memory_path, "".join(new_lines))

            for e in to_prune:
                archived.append({
                    "entry_type": e.entry_type,
                    "persistence": e.persistence,
                    "final_score": e.final_score,
                    "raw_text": e.raw_text[:200],
                    "section": e.section,
                })

            logger.info("Pruned %d/%d entries (threshold=%.3f)", len(to_prune), total, effective_threshold)

        return {"count": len(to_prune), "entries": archived}

    def _build_pruned_text(self, entries: list) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"# Pruned by Dreaming Deep Phase — {ts}", ""]
        for e in entries:
            parts.append(f"## [{e.entry_type}|{e.persistence}] score={e.final_score:.3f}")
            parts.append(f"  section={e.section}")
            parts.append(f"  {e.raw_text}")
            parts.append("")
        return "\n".join(parts)

    def _write_prune_archive(self, text: str) -> None:
        self._pruned_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        path = self._pruned_dir / f"{today}.md"
        existing = ""
        if path.exists():
            existing = path.read_text(encoding="utf-8")
        combined = existing + "\n" + text if existing else text
        self._atomic_write(path, combined)

    # ── Deep: Task D — Distillation ───────────────────────────────────

    async def _deep_distillation(self) -> None:
        """Run session + memory distillation if due (migrated from heartbeat)."""
        from lightclaw.agent.memory.memory_manager import MemoryManager
        mm = None
        try:
            # Get the global memory_manager instance — it's registered on the runner
            # but here we have no runner; probe via the app state instead.
            # This is a best-effort call — we log and continue on failure.
            from lightclaw.config import load_config
            import sys
            # Access the running app's state via sys.modules
            app_mod = sys.modules.get("lightclaw.app._app")
            if app_mod is not None:
                runner = getattr(app_mod, "runner", None)
                if runner is not None:
                    mm = getattr(runner, "memory_manager", None)
        except Exception:
            mm = None

        if mm is None:
            logger.debug("Deep distillation skipped: no memory_manager available")
            return

        try:
            if mm.should_distill_sessions():
                logger.info("Dreaming: session distillation triggered")
                await mm.distill_sessions_to_daily_md()
        except Exception:
            logger.exception("Session distillation failed in dreaming")

        try:
            if mm.should_distill():
                logger.info("Dreaming: memory distillation triggered")
                await mm.distill_to_memory_md()
        except Exception:
            logger.exception("Memory distillation failed in dreaming")

    # ── Deep: Tasks E–I — Managed section updates ─────────────────────

    async def _deep_update_managed_files(
        self, signals: list[PhaseSignal], candidates: list[DreamCandidate],
    ) -> list[str]:
        updated: list[str] = []
        signal_types = {s.signal_type for s in signals}

        # Task E: USER.md
        if self._config.auto_update_user_profile and "user_profile_update_needed" in signal_types:
            await self._update_managed_file(
                self._root / "USER.md",
                self._build_user_profile_update(candidates, signals),
            )
            updated.append("USER.md")

        # Task F: PROACTIVITY_ANALYSIS.md
        if self._config.auto_update_proactivity and "proactivity_update_needed" in signal_types:
            await self._update_managed_file(
                self._root / "PROACTIVITY_ANALYSIS.md",
                self._build_proactivity_update(candidates, signals),
            )
            updated.append("PROACTIVITY_ANALYSIS.md")

        # Task G: TOOLS.md
        if self._config.auto_update_tools and "tool_rule_update_needed" in signal_types:
            await self._update_managed_file(
                self._root / "TOOLS.md",
                self._build_tools_update(candidates, signals),
            )
            updated.append("TOOLS.md")

        # Task H: AGENTS.md
        if self._config.auto_update_agents and "agent_rule_update_needed" in signal_types:
            await self._update_managed_file(
                self._root / "AGENTS.md",
                self._build_agents_update(candidates, signals),
            )
            updated.append("AGENTS.md")

        # Task I: HEARTBEAT.md
        if self._config.auto_update_heartbeat and any(
            s.signal_type in ("agent_rule_update_needed", "tool_rule_update_needed")
            for s in signals
        ):
            await self._update_managed_file(
                self._root / "HEARTBEAT.md",
                self._build_heartbeat_update(),
            )
            updated.append("HEARTBEAT.md")

        return updated

    async def _update_managed_file(self, path: Path, content: str) -> None:
        if not content.strip():
            return
        self._upsert_managed_section(path, content)

    def _build_user_profile_update(
        self, candidates: list[DreamCandidate], signals: list[PhaseSignal],
    ) -> str:
        relevant = [
            c for c in candidates
            if any(s.candidate_id == c.id and s.signal_type == "user_profile_update_needed"
                   for s in signals)
        ]
        if not relevant:
            return ""
        lines = ["## Dreaming 自动更新", ""]
        for c in relevant[:5]:
            lines.append(f"- {c.summary}")
        lines.append(f"\n_最后更新: {datetime.now().strftime('%Y-%m-%d')}_")
        return "\n".join(lines)

    def _build_proactivity_update(
        self, candidates: list[DreamCandidate], signals: list[PhaseSignal],
    ) -> str:
        return (
            f"## Dreaming 活跃度检查\n\n"
            f"最近做梦分析于 {datetime.now().strftime('%Y-%m-%d')} 完成。\n"
        )

    def _build_tools_update(
        self, candidates: list[DreamCandidate], signals: list[PhaseSignal],
    ) -> str:
        relevant = [
            c for c in candidates
            if any(s.candidate_id == c.id and s.signal_type == "tool_rule_update_needed"
                   for s in signals)
        ]
        if not relevant:
            return ""
        lines = ["## Dreaming 工具使用模式", ""]
        for c in relevant[:5]:
            lines.append(f"- {c.summary}")
        return "\n".join(lines)

    def _build_agents_update(
        self, candidates: list[DreamCandidate], signals: list[PhaseSignal],
    ) -> str:
        return (
            f"## 自动做梦机制\n\n"
            f"### 心跳规则\n"
            f"- heartbeat 只做轻量检查，禁止 LLM 调用\n"
            f"- heartbeat 不写 MEMORY.md，不执行 embedding\n"
            f"- 输出 HEARTBEAT_OK 表示无需关注\n\n"
            f"### 夜间做梦（每日 {self._config.cron} {self._config.timezone}）\n"
            f"系统每天凌晨执行三阶段记忆巩固：\n"
            f"- Light Phase: 扫描 DAILY_TALK.md 和 memory/*.md，提取候选长期记忆\n"
            f"- REM Phase: 跨文件关联，发现重复/冲突/模式\n"
            f"- Deep Phase: 提升候选到 MEMORY.md，衰减淘汰过时条目，更新配置\n\n"
            f"### 记忆写入规则\n"
            f"- 所有写入使用结构化格式 (type/persistence/source/evidence_hash)\n"
            f"- 正则优先，LLM 补位\n"
            f"- 敏感信息永不写入任何文件\n"
            f"- 临时信息(天气/余额/版本号)自动过滤\n"
        )

    def _build_heartbeat_update(self) -> str:
        # HEARTBEAT.md is now manually controlled (auto_update_heartbeat defaults to False).
        # Dreaming can write suggestions to DREAMS.md but must not auto-patch HEARTBEAT.md.
        return (
            "# 心跳任务\n\n"
            "每次心跳只做轻量检查，禁止执行 LLM 总结、embedding 批量计算或 MEMORY.md 写入。\n"
            "无异常则记录 HEARTBEAT_OK。\n\n"
            "## 轻量检查\n\n"
            "- 服务状态正常\n"
            "- memory system 已初始化\n"
            "- DAILY_TALK.md 是否有内容\n"
            "- MEMORY.md 是否超过大小阈值\n"
            "- 每日做梦最近运行状态 (memory/.dreams/status.json)\n\n"
            "## 禁止事项\n\n"
            "心跳不执行以下操作(全部迁移至每日 dreaming cron):\n"
            "- session 蒸馏与记忆蒸馏\n"
            "- DAILY_TALK.md LLM 压缩\n"
            "- MEMORY.md LLM 压缩\n"
            "- 记忆衰减评分与淘汰\n"
            "- 长期记忆提取与写入\n"
        )

    # ── Atomic write helpers ──────────────────────────────────────────

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _upsert_managed_section(file_path: Path, new_content: str) -> None:
        if file_path.is_file():
            text = file_path.read_text(encoding="utf-8")
        else:
            text = ""
        managed_block = f"\n{_MANAGED_BEGIN}\n{new_content}\n{_MANAGED_END}\n"
        if _MANAGED_BEGIN in text and _MANAGED_END in text:
            start = text.index(_MANAGED_BEGIN)
            end = text.index(_MANAGED_END) + len(_MANAGED_END)
            new_text = text[:start] + managed_block.strip() + "\n" + text[end:]
        else:
            new_text = (text.rstrip() + "\n\n" + managed_block.strip() + "\n")
        DreamingPipeline._atomic_write(file_path, new_text)

    # ── Report generation ─────────────────────────────────────────────

    def _generate_report(self, result: DreamingResult) -> str:
        lines = [
            f"# Dreaming Report — {result.date}",
            f"模式: {'DRY RUN (仅预览)' if result.dry_run else '正式执行'}",
            f"开始: {result.started_at}",
            f"结束: {result.ended_at}",
            f"耗时: {result.duration_seconds}s",
            "",
            "## Light Phase",
            f"- 候选总数: {result.light_candidates_total}",
            f"- 保留: {result.light_candidates_kept}",
            f"- 敏感信息过滤: {result.light_secrets_filtered}",
            f"- 临时信息过滤: {result.light_ephemeral_filtered}",
            "",
            "## REM Phase",
            f"- 信号总数: {len(result.rem_signals)}",
        ]

        # Signal statistics
        if result.rem_signals:
            type_counts: dict[str, int] = {}
            for s in result.rem_signals:
                t = s.signal_type or "unknown"
                type_counts[t] = type_counts.get(t, 0) + 1
            lines.append("- 信号分类:")
            for t, count in sorted(type_counts.items()):
                lines.append(f"  - {t}: {count}")
            promote_count = sum(1 for s in result.rem_signals if s.should_promote)
            lines.append(f"- 建议提升: {promote_count}")

        lines += [
            "",
            "## Deep Phase",
            f"- 提升到 MEMORY.md: {result.promoted_count}",
            f"- 衰减淘汰: {result.pruned_count}",
            f"- 文件更新: {', '.join(result.updated_files) if result.updated_files else '无'}",
        ]

        if result.promoted_entries:
            lines.append("\n### 提升条目")
            for e in result.promoted_entries:
                lines.append(f"- [{e['entry_type']}] {e['summary'][:80]}")

        if result.pruned_entries:
            lines.append("\n### 淘汰条目")
            for e in result.pruned_entries:
                lines.append(f"- [{e['entry_type']}|{e.get('final_score', 0):.3f}] {e['raw_text'][:80]}")

        if result.errors:
            lines.append("\n## 错误")
            for err in result.errors:
                lines.append(f"- {err}")

        # Conflicts section
        if result.rem_signals:
            conflicts_section = self._build_conflicts_report(result.rem_signals)
            if conflicts_section:
                lines.append(conflicts_section)

        return "\n".join(lines)

    def _write_dream_report(self, result: DreamingResult) -> None:
        report_path = self._root / "DREAMS.md"
        self._dreams_dir.mkdir(parents=True, exist_ok=True)
        existing = ""
        if report_path.is_file():
            existing = report_path.read_text(encoding="utf-8")
        divider = "\n\n---\n\n" if existing else ""
        self._atomic_write(report_path, existing + divider + result.report_text)

    def _write_status(self, result: DreamingResult) -> None:
        self._dreams_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "last_run_at": result.started_at,
            "last_success_at": result.ended_at if not result.errors else None,
            "last_error": result.errors[0] if result.errors else None,
            "run_type": "full",
            "promoted_count": result.promoted_count,
            "pruned_count": result.pruned_count,
            "updated_files": result.updated_files,
            "duration_seconds": result.duration_seconds,
        }
        self._atomic_write(
            self._dreams_dir / "status.json",
            json.dumps(status, indent=2, ensure_ascii=False),
        )

    # ── Evidence persistence ───────────────────────────────────────────

    def _persist_candidates(self, candidates: list[DreamCandidate], tag: str) -> None:
        """Write candidates to memory/.dreams/candidates/{tag}-YYYY-MM-DD-HHMM.json."""
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        fname = f"{tag}-{ts}.json"
        payload = [
            {
                "id": c.id,
                "source_file": c.source_file,
                "source_date": c.source_date,
                "summary": c.summary,
                "entry_type": c.entry_type,
                "persistence": c.persistence,
                "keywords": c.keywords,
                "confidence": c.confidence,
                "evidence_hash": c.evidence_hash,
                "has_secret": c.has_secret,
                "raw_text": c.raw_text[:500],
            }
            for c in candidates
        ]
        self._atomic_write(self._candidates_dir / fname, json.dumps(payload, indent=2, ensure_ascii=False))

    def _persist_signals(self, signals: list[PhaseSignal]) -> None:
        """Write signals to memory/.dreams/signals/YYYY-MM-DD-HHMM.json."""
        if not signals:
            return
        self._signals_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        fname = f"{ts}.json"
        payload = [
            {
                "candidate_id": s.candidate_id,
                "signal_type": s.signal_type,
                "score_delta": s.score_delta,
                "should_promote": s.should_promote,
                "confidence": s.confidence,
                "reason": s.reason,
                "conflict_with": s.conflict_with,
                "pattern_category": s.pattern_category,
                "cross_ref_count": s.cross_ref_count,
                "parse_error": s.parse_error,
            }
            for s in signals
        ]
        self._atomic_write(self._signals_dir / fname, json.dumps(payload, indent=2, ensure_ascii=False))

    def _write_reinforcement(
        self, candidates: list[DreamCandidate], signals: list[PhaseSignal],
    ) -> None:
        """Append strengthens_existing_memory / repeated_across_days evidence to reinforcement.jsonl."""
        cand_map = {c.id: c for c in candidates}
        reinforcing = [
            s for s in signals
            if s.signal_type in ("strengthens_existing_memory", "repeated_across_days")
        ]
        if not reinforcing:
            return
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        lines = []
        for s in reinforcing:
            c = cand_map.get(s.candidate_id)
            entry = {
                "seen_at": now,
                "candidate_id": s.candidate_id,
                "signal_type": s.signal_type,
                "source": c.source_file if c else "",
                "evidence_hash": c.evidence_hash if c else "",
                "score_delta": s.score_delta,
                "confidence": s.confidence,
                "reason": s.reason,
                "memory_id": None,
            }
            lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
        self._dreams_dir.mkdir(parents=True, exist_ok=True)
        with open(self._reinforcement_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def _write_conflicts(
        self, candidates: list[DreamCandidate], signals: list[PhaseSignal],
    ) -> None:
        """Append conflicts_with_existing_memory evidence to conflicts.jsonl."""
        cand_map = {c.id: c for c in candidates}
        conflicting = [
            s for s in signals
            if s.signal_type == "conflicts_with_existing_memory"
        ]
        if not conflicting:
            return
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        lines = []
        for s in conflicting:
            c = cand_map.get(s.candidate_id)
            entry = {
                "seen_at": now,
                "candidate_id": s.candidate_id,
                "source": c.source_file if c else "",
                "evidence_hash": c.evidence_hash if c else "",
                "conflict_with": s.conflict_with or "",
                "reason": s.reason,
                "confidence": s.confidence,
                "status": "pending_review",
            }
            lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
        self._dreams_dir.mkdir(parents=True, exist_ok=True)
        with open(self._conflicts_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def _build_conflicts_report(self, signals: list[PhaseSignal]) -> str:
        """Build human-readable conflicts section for DREAMS.md."""
        conflicting = [
            s for s in signals
            if s.signal_type == "conflicts_with_existing_memory"
        ]
        if not conflicting:
            return ""
        lines = ["\n## Conflicts Pending Review", ""]
        for s in conflicting:
            lines.append(f"- candidate_id: {s.candidate_id}")
            lines.append(f"  conflict_with: {s.conflict_with or 'N/A'}")
            lines.append(f"  reason: {s.reason}")
            lines.append(f"  confidence: {s.confidence:.2f}")
            lines.append(f"  status: pending_review")
            lines.append("")
        return "\n".join(lines)


# ── Hash helpers ─────────────────────────────────────────────────────

def _hash_8(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:8]


def _hash_16(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]
