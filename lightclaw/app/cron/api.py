from __future__ import annotations

import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from lightclaw.app.cron.manager import CronManager
from lightclaw.app.cron.models import CronJobSpec, CronJobView
from lightclaw.constant import WORKING_DIR

router = APIRouter(prefix="/cron", tags=["cron"])


def get_cron_manager(request: Request) -> CronManager:
    mgr = getattr(request.app.state, "cron_manager", None)
    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="cron manager not initialized",
        )
    return mgr


@router.get("/jobs", response_model=list[CronJobSpec])
async def list_jobs(mgr: CronManager = Depends(get_cron_manager)):
    return await mgr.list_jobs()


@router.get("/jobs/{job_id}", response_model=CronJobView)
async def get_job(job_id: str, mgr: CronManager = Depends(get_cron_manager)):
    job = await mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return CronJobView(spec=job, state=mgr.get_state(job_id))


@router.post("/jobs", response_model=CronJobSpec)
async def create_job(
    spec: CronJobSpec,
    mgr: CronManager = Depends(get_cron_manager),
):
    # server generates id; ignore client-provided spec.id
    today = datetime.now(UTC).strftime("%Y%m%d")
    job_id = f"job-{today}-{secrets.token_hex(2)}"
    created = spec.model_copy(update={"id": job_id})
    await mgr.create_or_replace_job(created)
    return created


@router.put("/jobs/{job_id}", response_model=CronJobSpec)
async def replace_job(
    job_id: str,
    spec: CronJobSpec,
    mgr: CronManager = Depends(get_cron_manager),
):
    if spec.id != job_id:
        raise HTTPException(status_code=400, detail="job_id mismatch")
    await mgr.create_or_replace_job(spec)
    return spec


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    mgr: CronManager = Depends(get_cron_manager),
):
    ok = await mgr.delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"deleted": True}


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str, mgr: CronManager = Depends(get_cron_manager)):
    try:
        await mgr.pause_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"paused": True}


@router.post("/jobs/{job_id}/resume")
async def resume_job(
    job_id: str,
    mgr: CronManager = Depends(get_cron_manager),
):
    try:
        await mgr.resume_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"resumed": True}


@router.post("/jobs/{job_id}/run")
async def run_job(job_id: str, mgr: CronManager = Depends(get_cron_manager)):
    try:
        await mgr.run_job(job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="job not found") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"started": True}


@router.get("/jobs/{job_id}/state")
async def get_job_state(
    job_id: str,
    mgr: CronManager = Depends(get_cron_manager),
):
    job = await mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return mgr.get_state(job_id).model_dump(mode="json")


@router.post("/proactivity/run")
async def run_proactivity(
    bypass_gate: bool = False,
    mgr: CronManager = Depends(get_cron_manager),
):
    """Trigger an immediate proactivity evaluation (fire-and-forget).

    Set ``bypass_gate=true`` to skip the probability dice roll and forced
    dispatch for testing/debugging.
    """
    from lightclaw.app.cron.proactivity import run_proactivity_once

    runner = getattr(mgr, "_runner", None)
    channel_manager = getattr(mgr, "_channel_manager", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="runner not initialized")

    import asyncio

    if bypass_gate:
        # Bypass the full engine pipeline (dice roll, mode selection, etc.)
        # and directly run the agent with the PROACTIVE prompt plus any
        # memory candidates found by the engine.
        from lightclaw.app.cron.proactivity import (
            _format_candidates_section,
            _load_proactive_prompt,
        )
        from lightclaw.app.cron.proactivity import _run_cron_agent as run_agent
        from lightclaw.config import load_config

        mm = getattr(runner, "memory_manager", None)
        engine = getattr(mm, "proactive_engine", None) if mm else None

        query_text = _load_proactive_prompt()
        if not query_text:
            raise HTTPException(status_code=500, detail="PROACTIVE.md not found or empty")

        candidates = []
        if engine is not None:
            try:
                candidates = await engine.fetch_candidates()
            except Exception:
                candidates = []
        if candidates:
            lang = getattr(engine.config, "language", "zh") if engine else "zh"
            section = _format_candidates_section(candidates, lang)
            if section:
                query_text += "\n\n" + section

        cfg = load_config()
        ld = cfg.last_dispatch
        dispatch_user_id = (ld.user_id if ld else "") or "main"
        dispatch_session_id = (ld.session_id if ld else "") or "main"

        asyncio.create_task(
            run_agent(
                query_text=query_text,
                runner=runner,
                channel_manager=channel_manager,
                config=cfg,
                target="last",
                source="proactivity",
                user_id=dispatch_user_id,
                session_id=dispatch_session_id,
            )
        )

        return {
            "status": "started",
            "bypass_gate": True,
            "candidates": len(candidates),
            "message": "Agent dispatched directly (bypassing engine gate)",
        }
    else:
        asyncio.create_task(
            run_proactivity_once(runner=runner, channel_manager=channel_manager)
        )

    return {
        "status": "started",
        "bypass_gate": bypass_gate,
        "message": "Proactivity evaluation started in background",
    }


@router.get("/proactivity/status")
async def get_proactivity_status(mgr: CronManager = Depends(get_cron_manager)):
    """Get proactivity engine status."""
    from lightclaw.config import get_proactivity_config

    cfg = get_proactivity_config()
    runner = getattr(mgr, "_runner", None)
    mm = getattr(runner, "memory_manager", None) if runner else None
    engine = getattr(mm, "proactive_engine", None) if mm else None

    return {
        "enabled": cfg.enabled,
        "target": cfg.target,
        "every": cfg.every,
        "active_hours": cfg.active_hours,
        "engine_available": engine is not None,
        "engine_enabled": engine.config.enabled if engine else False,
    }


@router.post("/dreaming/run")
async def run_dreaming(
    dry_run: bool = False,
    mgr: CronManager = Depends(get_cron_manager),
):
    """Trigger an immediate dreaming pipeline run (fire-and-forget)."""
    from lightclaw.app.cron.dreaming import run_dreaming_once

    runner = getattr(mgr, "_runner", None)
    channel_manager = getattr(mgr, "_channel_manager", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="runner not initialized")

    import asyncio

    asyncio.create_task(
        run_dreaming_once(
            runner=runner, channel_manager=channel_manager, dry_run=dry_run,
        )
    )
    return {
        "status": "started",
        "dry_run": dry_run,
        "message": "Dreaming pipeline started in background",
    }


@router.get("/dreaming/status")
async def get_dreaming_status(mgr: CronManager = Depends(get_cron_manager)):
    """Get dreaming pipeline status."""
    from lightclaw.config import get_dreaming_config

    cfg = get_dreaming_config()
    status_path = WORKING_DIR / "memory" / ".dreams" / "status.json"
    status = {}
    if status_path.is_file():
        import json

        status = json.loads(status_path.read_text(encoding="utf-8"))
    return {
        "enabled": cfg.enabled,
        "cron": cfg.cron,
        "light_ingest_cron": cfg.light_ingest_cron,
        "timezone": cfg.timezone,
        "last_run_at": status.get("last_run_at"),
        "last_success_at": status.get("last_success_at"),
        "last_error": status.get("last_error"),
        "last_light_ingest_at": status.get("last_light_ingest_at"),
        "last_light_ingest_success_at": status.get("last_light_ingest_success_at"),
        "run_type": status.get("run_type"),
        "promoted_count": status.get("promoted_count", 0),
        "pruned_count": status.get("pruned_count", 0),
    }


@router.post("/dreaming/light-ingest")
async def run_light_ingest(
    dry_run: bool = False,
    mgr: CronManager = Depends(get_cron_manager),
):
    """Trigger an immediate light-only ingest (fire-and-forget)."""
    from lightclaw.app.cron.dreaming import run_light_ingest_once

    runner = getattr(mgr, "_runner", None)
    channel_manager = getattr(mgr, "_channel_manager", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="runner not initialized")

    import asyncio

    asyncio.create_task(
        run_light_ingest_once(
            runner=runner, channel_manager=channel_manager, dry_run=dry_run,
        )
    )
    return {
        "status": "started",
        "dry_run": dry_run,
        "message": "Light-only ingest started in background",
    }
