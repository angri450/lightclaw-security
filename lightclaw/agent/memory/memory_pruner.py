"""Memory pruner: orchestrate parse → score → prune with dry-run support.

Inspired by OpenClaw dreaming Deep phase — scores every entry, archives
candidates whose final_score falls below the threshold, and writes a
human-readable pruning report.

Supports:
  - --dry-run: preview what would be pruned without making changes
  - archive: pruned entries land in memory/.pruned/YYYY-MM-DD.md
  - report: audit trail with reasoning for each pruned entry
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from lightclaw.agent.memory.decay_scorer import (
    DEFAULT_PRUNE_THRESHOLD,
    DecayScorer,
)
from lightclaw.agent.memory.memory_parser import MemoryEntry, MemoryParseResult

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Prune result
# ═══════════════════════════════════════════════════════════════════


class PruneResult:
    """Result of a pruning run."""

    def __init__(self) -> None:
        self.dry_run: bool = True
        self.threshold: float = DEFAULT_PRUNE_THRESHOLD
        self.total_entries: int = 0
        self.kept: list[MemoryEntry] = []
        self.pruned: list[MemoryEntry] = []
        self.pruned_text: str = ""        # original lines to remove
        self.new_memory_text: str = ""    # MEMORY.md after pruning
        self.errors: list[str] = []

    @property
    def pruned_count(self) -> int:
        return len(self.pruned)

    @property
    def kept_count(self) -> int:
        return len(self.kept)


# ═══════════════════════════════════════════════════════════════════
# Pruner
# ═══════════════════════════════════════════════════════════════════


class MemoryPruner:
    """Score, threshold, archive, and rewrite MEMORY.md."""

    def __init__(
        self,
        working_dir: str | Path,
        *,
        threshold: float = DEFAULT_PRUNE_THRESHOLD,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self._root = Path(working_dir)
        self._memory_path = self._root / "MEMORY.md"
        self._pruned_dir = self._root / "memory" / ".pruned"
        self.threshold = threshold
        self._scorer = DecayScorer(working_dir, embedding_fn=embedding_fn)

    # ── public API ──────────────────────────────────────────────────

    async def run(
        self,
        *,
        invoke_llm: Callable[[str, str], Awaitable[str]] | None = None,
        dry_run: bool = True,
    ) -> PruneResult:
        """Score and prune entries.

        When ``dry_run=True`` (default), produces a full report but
        does NOT modify MEMORY.md or write archive files.

        LLM is only called for ambiguous entries (type="unknown") to
        improve classification accuracy before thresholding.
        """
        result = PruneResult()
        result.dry_run = dry_run
        result.threshold = self.threshold

        if not self._memory_path.exists():
            result.errors.append(f"MEMORY.md not found at {self._memory_path}")
            return result

        # 1. Parse + Score
        try:
            parsed = await self._scorer.score(
                invoke_llm=invoke_llm,
                skip_llm=(invoke_llm is None),
            )
        except Exception as exc:
            result.errors.append(f"Scoring failed: {exc}")
            logger.exception("Memory scoring failed")
            return result

        result.total_entries = parsed.entry_count

        # 2. Split: keep vs prune
        for entry in parsed.entries:
            if entry.final_score < self.threshold:
                result.pruned.append(entry)
            else:
                result.kept.append(entry)

        if not result.pruned:
            logger.info("Memory prune: nothing below threshold %.2f", self.threshold)
            return result

        # 3. Build new MEMORY.md text
        result.pruned_text, result.new_memory_text = self._remove_pruned_lines(
            parsed, result.pruned
        )

        # 4. Apply (if not dry-run)
        if not dry_run:
            try:
                await self._apply(result)
            except Exception as exc:
                result.errors.append(f"Apply failed: {exc}")
                logger.exception("Memory prune apply failed")

        return result

    def run_sync(self, *, dry_run: bool = True) -> PruneResult:
        """Synchronous scoring (regex-only, no LLM)."""
        result = PruneResult()
        result.dry_run = dry_run
        result.threshold = self.threshold

        if not self._memory_path.exists():
            result.errors.append(f"MEMORY.md not found at {self._memory_path}")
            return result

        try:
            parsed = self._scorer.score_sync()
        except Exception as exc:
            result.errors.append(f"Scoring failed: {exc}")
            logger.exception("Memory scoring failed")
            return result

        result.total_entries = parsed.entry_count

        for entry in parsed.entries:
            if entry.final_score < self.threshold:
                result.pruned.append(entry)
            else:
                result.kept.append(entry)

        if not result.pruned:
            return result

        result.pruned_text, result.new_memory_text = self._remove_pruned_lines(
            parsed, result.pruned
        )

        if not dry_run:
            try:
                self._apply_sync(result)
            except Exception as exc:
                result.errors.append(f"Apply failed: {exc}")
                logger.exception("Memory prune apply failed")

        return result

    # ── text-level removal ──────────────────────────────────────────

    @staticmethod
    def _remove_pruned_lines(
        parsed: MemoryParseResult,
        pruned: list[MemoryEntry],
    ) -> tuple[str, str]:
        """Remove pruned entries from MEMORY.md via text matching.

        Since line numbers from the parser may be offset by code-fence
        removal, we match on the raw text of each pruned entry.  To avoid
        false matches we anchor on the full line containing the entry.

        Returns (pruned_text_for_archive, new_memory_text).
        """
        if not parsed.path.exists():
            return "", ""

        lines = parsed.path.read_text(encoding="utf-8").splitlines(keepends=True)
        pruned_indices: set[int] = set()
        pruned_collector: list[str] = []

        for entry in pruned:
            # Build a signature: look for the raw_text substring, anchored
            # to a line that starts with "- " or a digit-list prefix
            sig = entry.raw_text.strip()
            if not sig:
                continue
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Match bullet/numbered entries
                if stripped.startswith(("- ", "* ")) or re.match(r"^\d+\.\s", stripped):
                    body = re.sub(r"^[-*\d]+\\.?\\s*", "", stripped, count=1)
                    body = body.strip()
                    if sig in body or body in sig:
                        pruned_indices.add(i)
                        break
            pruned_collector.append(
                f"# [{entry.entry_type}|{entry.persistence}] score={entry.final_score:.3f}\n"
                f"# sec={entry.section} | sub={entry.subsection}\n"
                f"# {entry.raw_text[:200]}\n"
            )

        new_lines = [line for i, line in enumerate(lines) if i not in pruned_indices]
        pruned_text = "".join(pruned_collector)
        new_text = "".join(new_lines)
        return pruned_text, new_text

    # ── apply ───────────────────────────────────────────────────────

    async def _apply(self, result: PruneResult) -> None:
        """Write archived entries and overwrite MEMORY.md."""
        # Archive
        archive_text = self._build_archive(result)
        await self._write_archive(archive_text)

        # Overwrite
        self._memory_path.write_text(result.new_memory_text, encoding="utf-8")
        logger.info(
            "Memory pruned: %d entries removed, %d kept",
            result.pruned_count,
            result.kept_count,
        )

    def _apply_sync(self, result: PruneResult) -> None:
        archive_text = self._build_archive(result)
        self._write_archive_sync(archive_text)
        self._memory_path.write_text(result.new_memory_text, encoding="utf-8")
        logger.info(
            "Memory pruned: %d entries removed, %d kept",
            result.pruned_count,
            result.kept_count,
        )

    # ── archive ─────────────────────────────────────────────────────

    def _build_archive(self, result: PruneResult) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [
            f"# Pruned Memory — {ts}",
            f"# Threshold: {result.threshold:.2f}",
            f"# Removed: {result.pruned_count}/{result.total_entries} entries",
            "",
        ]
        for e in result.pruned:
            parts.append(
                f"## [{e.entry_type}|{e.persistence}] score={e.final_score:.3f}"
            )
            parts.append(
                f"  rec={e.recency_score:.3f} frq={e.frequency_score:.3f} "
                f"con={e.consolidation_score:.3f} per={e.persistence_score:.3f} "
                f"rch={e.richness_score:.3f}"
            )
            parts.append(f"  section={e.section} | subsection={e.subsection}")
            parts.append(f"  {e.raw_text}")
            parts.append("")
        return "\n".join(parts)

    async def _write_archive(self, text: str) -> None:
        self._pruned_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        path = self._pruned_dir / f"{today}.md"
        existing = ""
        if path.exists():
            existing = path.read_text(encoding="utf-8")
        path.write_text(existing + "\n" + text if existing else text, encoding="utf-8")

    def _write_archive_sync(self, text: str) -> None:
        self._pruned_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        path = self._pruned_dir / f"{today}.md"
        existing = ""
        if path.exists():
            existing = path.read_text(encoding="utf-8")
        path.write_text(existing + "\n" + text if existing else text, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# Report formatter
# ═══════════════════════════════════════════════════════════════════


def format_prune_report(result: PruneResult) -> str:
    """Human-readable pruning report."""
    if result.errors:
        return "ERRORS:\n" + "\n".join(f"  - {e}" for e in result.errors)

    mode = "DRY RUN" if result.dry_run else "APPLIED"
    lines = [
        f"Memory Prune Report — {mode}",
        f"Threshold: {result.threshold:.2f}",
        f"Total entries: {result.total_entries}",
        f"Kept: {result.kept_count}  |  Pruned: {result.pruned_count}",
        "=" * 70,
    ]

    if not result.pruned:
        lines.append("No entries below threshold — nothing to prune.")
        return "\n".join(lines)

    lines.append("")
    lines.append("PRUNED:")
    lines.append("-" * 70)
    for e in result.pruned:
        lines.append(
            f"  [{e.final_score:.3f}] [{e.entry_type:14}] "
            f"{e.raw_text[:100]}"
        )
        lines.append(
            f"         rec={e.recency_score:.3f} frq={e.frequency_score:.3f} "
            f"con={e.consolidation_score:.3f} per={e.persistence_score:.3f} "
            f"rch={e.richness_score:.3f}"
        )
        lines.append(f"         → section={e.section} | {e.subsection}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("KEPT (lowest scores first):")
    lines.append("-" * 70)
    show_kept = sorted(result.kept, key=lambda e: e.final_score)[:10]
    for e in show_kept:
        lines.append(
            f"  [{e.final_score:.3f}] [{e.entry_type:14}] "
            f"{e.raw_text[:100]}"
        )

    if not result.dry_run:
        lines.append("")
        lines.append("Archived to: memory/.pruned/")
        lines.append("MEMORY.md has been rewritten.")

    return "\n".join(lines)
