"""P3.1 Hermes-style read-only TaskDispatcher — delegate tool.

Main agent calls ``delegate_readonly_tasks(tasks=[...])`` to dispatch
multiple read-only research tasks concurrently to fresh-context workers.
Only structured ``WorkerResult`` summaries are returned; worker tool
outputs never enter the parent context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from lightclaw.agent.tools.delegate_readonly_tasks_worker import (
    WorkerResult,
    WorkerTask,
    run_single_worker,
)
from lightclaw.constant import (
    DELEGATE_MAX_CONCURRENT_CHILDREN,
    DELEGATE_MAX_TASKS,
    DELEGATE_RESULT_MAX_CHARS,
    DELEGATE_TOTAL_TIMEOUT_SECONDS,
    DELEGATE_WORKER_TIMEOUT_SECONDS,
)
from lightclaw.app.runner_context import get_runner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Write-intent detection patterns
# ---------------------------------------------------------------------------

_WRITE_INTENT_PATTERNS: list[str] = [
    "write_file",
    "写入文件",
    "修改代码",
    "delete_file",
    "删除文件",
    "execute_shell",
    "执行命令",
    "执行 shell",
    "运行命令",
    "install ",
    "安装",
    "重启服务",
    "restart",
    "send_message",
    "发送消息",
    "改配置",
    "修改配置",
    "写入",
    "write memory",
    "写 MEMORY",
    "写 memory",
    "browser_control",
    "desktop_screenshot",
    "delegate_readonly",
    "secret",
    "token",
    "password",
    "api_key",
]


def _detect_write_intent(task: dict) -> list[str]:
    """Return a list of detected write/side-effect intents in a task dict."""
    hits: list[str] = []
    combined = json.dumps(task, ensure_ascii=False).lower()
    for pat in _WRITE_INTENT_PATTERNS:
        if pat.lower() in combined:
            hits.append(pat)
    return hits


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_and_normalize(tasks_raw: Any, max_concurrent: Any, per_worker_timeout: Any) -> tuple[list[dict], int, int]:
    """Validate inputs, return (normalized_tasks, max_concurrent, timeout).

    Raises ValueError on invalid input.
    """
    if not isinstance(tasks_raw, list):
        raise ValueError("tasks must be a list")

    if len(tasks_raw) == 0:
        raise ValueError("tasks must not be empty")

    if len(tasks_raw) > DELEGATE_MAX_TASKS:
        raise ValueError(f"tasks count {len(tasks_raw)} exceeds maximum {DELEGATE_MAX_TASKS}")

    # Normalize and validate each task
    normalized: list[dict] = []
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            raise ValueError(f"task[{i}] must be a dict, got {type(t).__name__}")

        goal = (t.get("goal") or "").strip()
        if not goal:
            raise ValueError(f"task[{i}] is missing required field 'goal'")
        if len(goal) > 2000:
            raise ValueError(f"task[{i}] goal exceeds 2000 chars")

        context = (t.get("context") or "")[:6000]  # silently truncate
        expected = (t.get("expected_output") or "")[:2000]

        normalized.append({
            "task_id": t.get("task_id") or f"task_{i}",
            "goal": goal,
            "context": context,
            "expected_output": expected,
            "allowed_tools": t.get("allowed_tools"),  # None → use default whitelist
        })

    # Validate max_concurrent
    try:
        mc = int(max_concurrent) if max_concurrent is not None else DELEGATE_MAX_CONCURRENT_CHILDREN
    except (ValueError, TypeError):
        mc = DELEGATE_MAX_CONCURRENT_CHILDREN
    mc = max(1, min(mc, DELEGATE_MAX_CONCURRENT_CHILDREN))

    # Validate per_worker_timeout
    try:
        pwt = int(per_worker_timeout) if per_worker_timeout is not None else DELEGATE_WORKER_TIMEOUT_SECONDS
    except (ValueError, TypeError):
        pwt = DELEGATE_WORKER_TIMEOUT_SECONDS
    pwt = max(30, min(pwt, DELEGATE_WORKER_TIMEOUT_SECONDS))

    return normalized, mc, pwt


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


async def delegate_readonly_tasks(
    tasks: list[dict],
    max_concurrent: int = DELEGATE_MAX_CONCURRENT_CHILDREN,
    per_worker_timeout: int = DELEGATE_WORKER_TIMEOUT_SECONDS,
    result_format: str = "json",
) -> str:
    """Delegate multiple read-only research tasks to concurrent workers.

    Each worker runs in a fresh, isolated context with readonly tools only.
    Worker tool outputs are NOT returned to the parent — only structured
    summaries.

    Args:
        tasks: List of task dicts, each with:
            - task_id (str, optional): Human-readable ID.
            - goal (str, required): What to accomplish (max 2000 chars).
            - context (str, optional): Files, constraints, context (max 6000 chars).
            - expected_output (str, optional): What kind of answer you want.
            - allowed_tools (list[str], optional): Subset of readonly tools.
        max_concurrent: Maximum concurrent workers (1-4, default 4).
        per_worker_timeout: Timeout per worker in seconds (30-120, default 120).
        result_format: Output format ("json" only for now).

    Returns:
        JSON string with worker results wrapped in <child-agent-results> tags.
    """
    start_time = time.monotonic()

    # -- Validate inputs --------------------------------------------------------
    try:
        normalized, mc, pwt = _validate_and_normalize(tasks, max_concurrent, per_worker_timeout)
    except ValueError as exc:
        return json.dumps({
            "status": "rejected",
            "summary": f"Input validation failed: {exc}",
            "evidence": [],
            "risks": [],
            "recommended_next_action": "Fix input and retry.",
        }, ensure_ascii=False)

    # -- Check for write-intent tasks -------------------------------------------
    rejected: list[dict] = []
    accepted: list[dict] = []
    for t in normalized:
        hits = _detect_write_intent(t)
        if hits:
            rejected.append({
                "task_id": t["task_id"],
                "status": "rejected",
                "summary": f"Task rejected: write/side-effect intent detected — {', '.join(hits[:5])}",
                "recommended_next_action": "Remove write/side-effect operations or handle serially.",
            })
        else:
            accepted.append(t)

    if not accepted and rejected:
        return _format_results(rejected, DELEGATE_RESULT_MAX_CHARS)

    # -- Get runner instance ----------------------------------------------------
    runner = get_runner()
    if runner is None:
        return json.dumps({
            "status": "failed",
            "summary": "Runner not available — service may not be fully initialized.",
            "evidence": [],
            "risks": [],
            "recommended_next_action": "Retry later or use manual investigation.",
        }, ensure_ascii=False)

    # -- Run workers with concurrency control -----------------------------------
    parent_turn_id = f"delegate_{int(start_time)}"
    results: list[WorkerResult] = [WorkerResult(task_id=r["task_id"], status="rejected")
                                     for r in rejected]
    for r in rejected:
        results.append(WorkerResult(**r))

    sem = asyncio.Semaphore(mc)

    async def _run_one(task_dict: dict, idx: int) -> WorkerResult:
        async with sem:
            worker_task = WorkerTask(
                task_id=task_dict["task_id"],
                goal=task_dict["goal"],
                context=task_dict["context"],
                expected_output=task_dict["expected_output"],
                allowed_tools=task_dict.get("allowed_tools"),
            )
            return await run_single_worker(
                runner=runner,
                task=worker_task,
                parent_turn_id=parent_turn_id,
                task_index=idx,
                timeout=pwt,
            )

    try:
        coros = [_run_one(t, i) for i, t in enumerate(accepted)]
        worker_results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=True),
            timeout=DELEGATE_TOTAL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        worker_results = []

    # -- Collect results --------------------------------------------------------
    for i, wr in enumerate(worker_results):
        if isinstance(wr, WorkerResult):
            results.append(wr)
        elif isinstance(wr, BaseException):
            task_id = accepted[i]["task_id"] if i < len(accepted) else f"worker_{i}"
            results.append(WorkerResult(
                task_id=task_id,
                status="failed",
                summary=f"Worker crashed: {type(wr).__name__}: {wr}",
                error=str(wr)[:500],
            ))
        else:
            task_id = accepted[i]["task_id"] if i < len(accepted) else f"worker_{i}"
            results.append(WorkerResult(
                task_id=task_id,
                status="failed",
                summary=f"Unexpected result type: {type(wr).__name__}",
            ))

    total_duration = round(time.monotonic() - start_time, 2)

    return _format_results(
        [r.to_dict() for r in results],
        DELEGATE_RESULT_MAX_CHARS,
        total_duration=total_duration,
        total_workers=len(results),
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_results(
    result_dicts: list[dict],
    max_chars: int,
    *,
    total_duration: float = 0.0,
    total_workers: int = 0,
) -> str:
    """Format worker results as XML-wrapped JSON, enforcing max_chars."""
    wrapper = {
        "total_workers": total_workers or len(result_dicts),
        "total_duration_seconds": total_duration,
        "results": result_dicts,
    }
    json_str = json.dumps(wrapper, ensure_ascii=False, indent=2)

    if len(json_str) > max_chars:
        # Truncate individual summaries to fit
        per_result_max = max(200, (max_chars - 200) // max(len(result_dicts), 1))
        for rd in result_dicts:
            if len(rd.get("summary", "")) > per_result_max:
                rd["summary"] = rd["summary"][:per_result_max - 30] + "...[truncated]"
        wrapper["results"] = result_dicts
        json_str = json.dumps(wrapper, ensure_ascii=False, indent=2)
        # Final hard cut
        if len(json_str) > max_chars:
            json_str = json_str[:max_chars - 50] + "\n...[total output truncated]"

    return f"<child-agent-results>\n{json_str}\n</child-agent-results>"
