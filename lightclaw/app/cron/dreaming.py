"""Dreaming cron entry points. Full dreaming at 02:00, light-only ingest at 14:30."""

from __future__ import annotations

import logging
from typing import Any

from lightclaw.agent.memory.dreaming_pipeline import DreamingPipeline
from lightclaw.agent.memory.dreaming_types import DreamingResult
from lightclaw.config import get_dreaming_config
from lightclaw.constant import WORKING_DIR

logger = logging.getLogger(__name__)


def _build_pipeline(runner: Any) -> DreamingPipeline | None:
    config = get_dreaming_config()
    if not config.enabled:
        return None
    mm = getattr(runner, "memory_manager", None)
    if mm is None:
        return None
    invoke_llm = getattr(mm, "_invoke_llm", None)
    if invoke_llm is None:
        return None
    language = getattr(mm, "language", "zh")
    indexer = getattr(mm, "_memory_indexer", None)
    embedding_fn = getattr(indexer, "embedding_fn", None) if indexer else None

    return DreamingPipeline(
        working_dir=WORKING_DIR,
        invoke_llm=invoke_llm,
        language=language or "zh",
        config=config,
        embedding_fn=embedding_fn,
    )


async def run_dreaming_once(
    *,
    runner: Any,
    channel_manager: Any,
    dry_run: bool = False,
) -> DreamingResult | None:
    pipeline = _build_pipeline(runner)
    if pipeline is None:
        logger.info("Dreaming skipped: pipeline not available")
        return None
    result = await pipeline.run(dry_run=dry_run)
    logger.info(
        "Dreaming complete (dry_run=%s): %d candidates -> %d promoted, %d pruned",
        dry_run,
        result.light_candidates_total,
        result.promoted_count,
        result.pruned_count,
    )
    return result


async def run_light_ingest_once(
    *,
    runner: Any,
    channel_manager: Any,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run light-only ingest: Light Phase only, no REM, no Deep, no file writes except candidates."""
    pipeline = _build_pipeline(runner)
    if pipeline is None:
        logger.info("Light ingest skipped: pipeline not available")
        return None
    result = await pipeline.run_light_only(dry_run=dry_run)
    logger.info(
        "Light ingest complete (dry_run=%s): %d candidates extracted, %d kept",
        dry_run,
        result.get("candidates_total", 0),
        result.get("candidates_kept", 0),
    )
    return result
