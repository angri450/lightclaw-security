from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from lightclaw.app.cron.executor import CronExecutor
from lightclaw.app.cron.heartbeat import parse_heartbeat_every, run_heartbeat_once
from lightclaw.app.cron.models import CronJobSpec, CronJobState
from lightclaw.app.cron.proactivity import run_daily_analysis, run_proactivity_once, setup_proactive_engine
from lightclaw.app.cron.repo.base import BaseJobRepository
from lightclaw.app.stores.push import append as push_store_append
from lightclaw.config import get_heartbeat_config, get_proactivity_config

HEARTBEAT_JOB_ID = "_heartbeat"
PROACTIVITY_JOB_ID = "_proactivity"
DAILY_ANALYSIS_JOB_ID = "_daily_proactivity_analysis"
DREAMING_JOB_ID = "_dreaming"
DREAMING_LIGHT_INGEST_JOB_ID = "_dreaming_light_ingest"

logger = logging.getLogger(__name__)


@dataclass
class _Runtime:
    sem: asyncio.Semaphore


class CronManager:
    def __init__(
        self,
        *,
        repo: BaseJobRepository,
        runner: Any,
        channel_manager: Any,
        timezone: str = "UTC",
        task_registry: Any = None,
    ):
        self._repo = repo
        self._runner = runner
        self._channel_manager = channel_manager
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._executor = CronExecutor(
            runner=runner,
            channel_manager=channel_manager,
            task_registry=task_registry,
        )

        self._lock = asyncio.Lock()
        self._states: dict[str, CronJobState] = {}
        self._rt: dict[str, _Runtime] = {}
        self._started = False

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            jobs_file = await self._repo.load()

            self._scheduler.start()
            for job in jobs_file.jobs:
                await self._register_or_update(job)

            # Heartbeat: only schedule if enabled (config not None)
            hb = get_heartbeat_config()
            if hb is not None:
                interval_seconds = parse_heartbeat_every(hb.every)
                self._scheduler.add_job(
                    self._heartbeat_callback,
                    trigger=IntervalTrigger(seconds=interval_seconds),
                    id=HEARTBEAT_JOB_ID,
                    replace_existing=True,
                )

            # Proactivity: engine-driven (ProactiveMemoryEngine._run_loop)
            pa = get_proactivity_config()
            if pa.enabled:
                setup_proactive_engine(self._runner, self._channel_manager)

                mm = getattr(self._runner, "memory_manager", None)
                engine = getattr(mm, "proactive_engine", None) if mm else None
                engine_running = engine is not None and getattr(engine, "_task", None) is not None

                # Fallback: if engine didn't start, use APScheduler with legacy flow
                if not engine_running:
                    from lightclaw.app.cron.proactivity import _parse_interval

                    every_str = getattr(pa, "every", "600") or "600"
                    interval_seconds = _parse_interval(every_str)
                    self._scheduler.add_job(
                        self._proactivity_callback,
                        trigger=IntervalTrigger(seconds=interval_seconds),
                        id=PROACTIVITY_JOB_ID,
                        replace_existing=True,
                    )
                    logger.info(
                        "proactivity fallback: APScheduler interval=%ds (no engine)",
                        interval_seconds,
                    )

                # Daily analysis via APScheduler (separate schedule)
                hour, minute = self._parse_time(pa.daily_analysis_time)
                self._scheduler.add_job(
                    self._daily_analysis_callback,
                    trigger=CronTrigger(hour=hour, minute=minute),
                    id=DAILY_ANALYSIS_JOB_ID,
                    replace_existing=True,
                )
                logger.info(
                    "proactivity scheduled: engine=%s, daily_analysis=%02d:%02d",
                    "running" if engine_running else "not_started",
                    hour,
                    minute,
                )
            else:
                logger.info("proactivity not scheduled: config=%s", pa)

            # Dreaming pipeline: 3-phase nightly memory consolidation (full)
            from lightclaw.config import get_dreaming_config

            dreaming_cfg = get_dreaming_config()
            if dreaming_cfg.enabled:
                try:
                    trigger = CronTrigger.from_crontab(
                        dreaming_cfg.cron, timezone=dreaming_cfg.timezone
                    )
                except Exception:
                    logger.warning(
                        "Invalid dreaming cron '%s', using default 0 2 * * *",
                        dreaming_cfg.cron,
                    )
                    trigger = CronTrigger.from_crontab("0 2 * * *", timezone="Asia/Shanghai")
                self._scheduler.add_job(
                    self._dreaming_callback,
                    trigger=trigger,
                    id=DREAMING_JOB_ID,
                    replace_existing=True,
                )
                logger.info(
                    "dreaming scheduled: cron=%s tz=%s",
                    dreaming_cfg.cron,
                    dreaming_cfg.timezone,
                )

                # Light-only ingest (14:30 daily, separate from full dreaming)
                if dreaming_cfg.light_ingest_enabled:
                    try:
                        light_trigger = CronTrigger.from_crontab(
                            dreaming_cfg.light_ingest_cron, timezone=dreaming_cfg.timezone
                        )
                    except Exception:
                        logger.warning(
                            "Invalid light-ingest cron '%s', using default 30 14 * * *",
                            dreaming_cfg.light_ingest_cron,
                        )
                        light_trigger = CronTrigger.from_crontab(
                            "30 14 * * *", timezone="Asia/Shanghai"
                        )
                    self._scheduler.add_job(
                        self._dreaming_light_ingest_callback,
                        trigger=light_trigger,
                        id=DREAMING_LIGHT_INGEST_JOB_ID,
                        replace_existing=True,
                    )
                    logger.info(
                        "dreaming light-ingest scheduled: cron=%s tz=%s",
                        dreaming_cfg.light_ingest_cron,
                        dreaming_cfg.timezone,
                    )
            else:
                logger.info("dreaming not scheduled: disabled in config")

            self._started = True

    async def stop(self) -> None:
        async with self._lock:
            if not self._started:
                return
            # Stop proactive engine if running
            mm = getattr(self._runner, "memory_manager", None)
            engine = getattr(mm, "proactive_engine", None) if mm else None
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    logger.debug("proactive engine stop failed", exc_info=True)
            self._scheduler.shutdown(wait=False)
            self._started = False

    async def reschedule_heartbeat(self) -> None:
        """Re-read heartbeat config and reschedule (or remove) the heartbeat job."""
        async with self._lock:
            if not self._started:
                return
            hb = get_heartbeat_config()
            if self._scheduler.get_job(HEARTBEAT_JOB_ID):
                self._scheduler.remove_job(HEARTBEAT_JOB_ID)
            if hb is None:
                logger.info("heartbeat disabled: job removed")
                return
            interval_seconds = parse_heartbeat_every(hb.every)
            self._scheduler.add_job(
                self._heartbeat_callback,
                trigger=IntervalTrigger(seconds=interval_seconds),
                id=HEARTBEAT_JOB_ID,
                replace_existing=True,
            )
            logger.info(
                "heartbeat rescheduled: every=%ss",
                interval_seconds,
            )

    async def reschedule_proactivity(self) -> None:
        """Re-read proactivity config and reschedule (or remove) the jobs."""
        async with self._lock:
            if not self._started:
                return
            pa = get_proactivity_config()

            # Stop existing engine if running
            mm = getattr(self._runner, "memory_manager", None)
            engine = getattr(mm, "proactive_engine", None) if mm else None
            if engine is not None:
                engine.stop()

            # Remove existing APScheduler jobs
            for job_id in (PROACTIVITY_JOB_ID, DAILY_ANALYSIS_JOB_ID):
                if self._scheduler.get_job(job_id):
                    self._scheduler.remove_job(job_id)
            if pa is None or not pa.enabled:
                logger.info("proactivity disabled: jobs removed, engine stopped")
                return

            # Re-setup engine (will restart with new interval)
            setup_proactive_engine(self._runner, self._channel_manager)

            engine = getattr(mm, "proactive_engine", None) if mm else None
            engine_running = engine is not None and getattr(engine, "_task", None) is not None

            # Re-add daily analysis job
            hour, minute = self._parse_time(pa.daily_analysis_time)
            self._scheduler.add_job(
                self._daily_analysis_callback,
                trigger=CronTrigger(hour=hour, minute=minute),
                id=DAILY_ANALYSIS_JOB_ID,
                replace_existing=True,
            )
            logger.info(
                "proactivity rescheduled: engine=%s, daily_analysis=%02d:%02d",
                "running" if engine_running else "not_started",
                hour,
                minute,
            )

    async def reschedule_dreaming(self) -> None:
        """Re-read dreaming config and reschedule (or remove) both dreaming jobs."""
        async with self._lock:
            if not self._started:
                return
            from lightclaw.config import get_dreaming_config

            dreaming_cfg = get_dreaming_config()

            # Full dreaming
            if self._scheduler.get_job(DREAMING_JOB_ID):
                self._scheduler.remove_job(DREAMING_JOB_ID)
            if dreaming_cfg.enabled:
                try:
                    trigger = CronTrigger.from_crontab(
                        dreaming_cfg.cron, timezone=dreaming_cfg.timezone
                    )
                except Exception:
                    logger.warning(
                        "Invalid dreaming cron '%s', using default 0 2 * * *",
                        dreaming_cfg.cron,
                    )
                    trigger = CronTrigger.from_crontab("0 2 * * *", timezone="Asia/Shanghai")
                self._scheduler.add_job(
                    self._dreaming_callback,
                    trigger=trigger,
                    id=DREAMING_JOB_ID,
                    replace_existing=True,
                )
                logger.info(
                    "dreaming rescheduled: cron=%s tz=%s",
                    dreaming_cfg.cron,
                    dreaming_cfg.timezone,
                )
            else:
                logger.info("dreaming disabled: job removed")

            # Light-only ingest
            if self._scheduler.get_job(DREAMING_LIGHT_INGEST_JOB_ID):
                self._scheduler.remove_job(DREAMING_LIGHT_INGEST_JOB_ID)
            if dreaming_cfg.enabled and dreaming_cfg.light_ingest_enabled:
                try:
                    light_trigger = CronTrigger.from_crontab(
                        dreaming_cfg.light_ingest_cron, timezone=dreaming_cfg.timezone
                    )
                except Exception:
                    logger.warning(
                        "Invalid light-ingest cron '%s', using default 30 14 * * *",
                        dreaming_cfg.light_ingest_cron,
                    )
                    light_trigger = CronTrigger.from_crontab(
                        "30 14 * * *", timezone="Asia/Shanghai"
                    )
                self._scheduler.add_job(
                    self._dreaming_light_ingest_callback,
                    trigger=light_trigger,
                    id=DREAMING_LIGHT_INGEST_JOB_ID,
                    replace_existing=True,
                )
                logger.info(
                    "dreaming light-ingest rescheduled: cron=%s tz=%s",
                    dreaming_cfg.light_ingest_cron,
                    dreaming_cfg.timezone,
                )
            else:
                logger.info("dreaming light-ingest disabled: job removed")

    # ----- read/state -----

    async def list_jobs(self) -> list[CronJobSpec]:
        return await self._repo.list_jobs()

    async def get_job(self, job_id: str) -> CronJobSpec | None:
        return await self._repo.get_job(job_id)

    def get_state(self, job_id: str) -> CronJobState:
        return self._states.get(job_id, CronJobState())

    # ----- write/control -----

    async def create_or_replace_job(self, spec: CronJobSpec) -> None:
        async with self._lock:
            await self._repo.upsert_job(spec)
            if self._started:
                await self._register_or_update(spec)

    async def delete_job(self, job_id: str) -> bool:
        async with self._lock:
            if self._started and self._scheduler.get_job(job_id):
                self._scheduler.remove_job(job_id)
            self._states.pop(job_id, None)
            self._rt.pop(job_id, None)
            return await self._repo.delete_job(job_id)

    async def pause_job(self, job_id: str) -> None:
        async with self._lock:
            self._scheduler.pause_job(job_id)
            # Persist enabled=False so the job stays paused after restart
            await self._set_job_enabled(job_id, enabled=False)

    async def resume_job(self, job_id: str) -> None:
        async with self._lock:
            self._scheduler.resume_job(job_id)
            # Persist enabled=True so the job stays active after restart
            await self._set_job_enabled(job_id, enabled=True)

    async def _set_job_enabled(self, job_id: str, *, enabled: bool) -> None:
        """Update the ``enabled`` flag in persistent storage."""
        job = await self._repo.get_job(job_id)
        if job is None:
            return
        updated = job.model_copy(update={"enabled": enabled})
        await self._repo.upsert_job(updated)

    async def run_job(self, job_id: str) -> None:
        """Trigger a job to run in the background (fire-and-forget).

        Raises KeyError if the job does not exist.
        The actual execution happens asynchronously; errors are logged
        and reflected in the job state but NOT propagated to the caller.
        """
        job = await self._repo.get_job(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")
        logger.info(
            "cron run_job (async): job_id=%s channel=%s task_type=%s target_user_id=%s target_session_id=%s",
            job_id,
            job.dispatch.channel,
            job.task_type,
            (job.dispatch.target.user_id or "")[:40],
            (job.dispatch.target.session_id or "")[:40],
        )
        task = asyncio.create_task(
            self._execute_once(job),
            name=f"cron-run-{job_id}",
        )
        task.add_done_callback(lambda t: self._task_done_cb(t, job))

    # ----- callbacks -----

    def _task_done_cb(self, task: asyncio.Task, job: CronJobSpec) -> None:
        """Suppress and log exceptions from fire-and-forget tasks.

        On failure, push an error message to the console push store so
        the frontend can display it.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "cron background task %s failed: %s",
                task.get_name(),
                repr(exc),
            )
            # Push error to the console for the frontend to display
            session_id = job.dispatch.target.session_id
            if session_id:
                error_text = f"❌ Cron job [{job.name}] failed: {exc}"
                asyncio.ensure_future(
                    push_store_append(session_id, error_text),
                )

    # ----- internal -----

    async def _register_or_update(self, spec: CronJobSpec) -> None:
        # per-job concurrency semaphore
        self._rt[spec.id] = _Runtime(
            sem=asyncio.Semaphore(spec.runtime.max_concurrency),
        )

        trigger = self._build_trigger(spec)

        # replace existing
        if self._scheduler.get_job(spec.id):
            self._scheduler.remove_job(spec.id)

        self._scheduler.add_job(
            self._scheduled_callback,
            trigger=trigger,
            id=spec.id,
            args=[spec.id],
            misfire_grace_time=spec.runtime.misfire_grace_seconds,
            replace_existing=True,
        )

        if not spec.enabled:
            self._scheduler.pause_job(spec.id)

        # update next_run
        aps_job = self._scheduler.get_job(spec.id)
        st = self._states.get(spec.id, CronJobState())
        st.next_run_at = aps_job.next_run_time if aps_job else None
        self._states[spec.id] = st

    def _build_trigger(self, spec: CronJobSpec) -> CronTrigger:
        # enforce 5 fields (no seconds)
        parts = [p for p in spec.schedule.cron.split() if p]
        if len(parts) != 5:
            raise ValueError(
                f"cron must have 5 fields, got {len(parts)}: {spec.schedule.cron}",
            )

        minute, hour, day, month, day_of_week = parts
        return CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone=spec.schedule.timezone,
        )

    async def _scheduled_callback(self, job_id: str) -> None:
        job = await self._repo.get_job(job_id)
        if not job:
            return

        await self._execute_once(job)

        # refresh next_run
        aps_job = self._scheduler.get_job(job_id)
        st = self._states.get(job_id, CronJobState())
        st.next_run_at = aps_job.next_run_time if aps_job else None
        self._states[job_id] = st

    async def _heartbeat_callback(self) -> None:
        """Run one heartbeat (HEARTBEAT.md as query, optional dispatch)."""
        try:
            await run_heartbeat_once(
                runner=self._runner,
                channel_manager=self._channel_manager,
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("heartbeat run failed")

    async def _proactivity_callback(self) -> None:
        """APScheduler fallback callback — runs legacy flow when engine is unavailable."""
        try:
            await run_proactivity_once(self._runner, self._channel_manager)
        except Exception:
            logger.exception("proactivity callback failed")

    async def _daily_analysis_callback(self) -> None:
        """Run daily proactivity analysis (LLM-driven activity_level adjustment)."""
        try:
            await run_daily_analysis()
        except Exception:
            logger.exception("daily proactivity analysis failed")

    async def _dreaming_callback(self) -> None:
        """Run nightly dreaming pipeline (Light/REM/Deep phases)."""
        from lightclaw.app.cron.dreaming import run_dreaming_once

        try:
            await run_dreaming_once(
                runner=self._runner,
                channel_manager=self._channel_manager,
            )
        except Exception:
            logger.exception("Dreaming pipeline failed")

    async def _dreaming_light_ingest_callback(self) -> None:
        """Run light-only ingest (Light Phase only, no REM/Deep)."""
        from lightclaw.app.cron.dreaming import run_light_ingest_once

        try:
            await run_light_ingest_once(
                runner=self._runner,
                channel_manager=self._channel_manager,
            )
        except Exception:
            logger.exception("Dreaming light-ingest failed")

    @staticmethod
    def _parse_time(time_str: str) -> tuple[int, int]:
        """Parse 'HH:MM' to (hour, minute). Defaults to (23, 0) on error."""
        try:
            parts = time_str.strip().split(":")
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return 23, 0

    async def _execute_once(self, job: CronJobSpec) -> None:
        rt = self._rt.get(job.id)
        if not rt:
            rt = _Runtime(sem=asyncio.Semaphore(job.runtime.max_concurrency))
            self._rt[job.id] = rt

        async with rt.sem:
            st = self._states.get(job.id, CronJobState())
            st.last_status = "running"
            self._states[job.id] = st

            try:
                await self._executor.execute(job)
                st.last_status = "success"
                st.last_error = None
                logger.info(
                    "cron _execute_once: job_id=%s status=success",
                    job.id,
                )
            except Exception as e:  # pylint: disable=broad-except
                st.last_status = "error"
                st.last_error = repr(e)
                logger.warning(
                    "cron _execute_once: job_id=%s status=error error=%s",
                    job.id,
                    repr(e),
                )
                raise
            finally:
                st.last_run_at = datetime.now(UTC)
                self._states[job.id] = st
