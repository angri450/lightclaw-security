"""Single readonly worker runner for P3.1 TaskDispatcher.

Each worker runs in a fresh context with restricted tools and returns
a ``WorkerResult``.  Worker tool outputs never enter the parent context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from lightclaw.constant import (
    DELEGATE_WORKER_MAX_INPUT_TOKENS,
    DELEGATE_WORKER_MAX_ITERS,
    DELEGATE_WORKER_SUMMARY_MAX_CHARS,
    DELEGATE_WORKER_TIMEOUT_SECONDS,
    WORKING_DIR,
)
from lightclaw.domain import TurnInput, TurnRequest, TurnEventType
from lightclaw.domain.identity import build_canonical_user_id
from lightclaw.agent.core.actor_context import (
    ActorContext,
    actor_context_to_dict,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Readonly worker tool whitelist
# ---------------------------------------------------------------------------

_READONLY_WORKER_ALLOWED: tuple[str, ...] = (
    "read_file",
    "grep_search",
    "glob_search",
    "file_preview",
    "get_current_time",
    "read_status_file",
    "read_recent_heartbeat_errors",
    "memory_search",
)

# ---------------------------------------------------------------------------
# WorkerResult
# ---------------------------------------------------------------------------


@dataclass
class WorkerResult:
    """Structured result from a single readonly worker."""

    task_id: str
    status: str  # success | failed | timeout | cancelled | rejected
    summary: str = ""
    evidence: list[dict] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommended_next_action: str = ""
    tokens_used_estimate: int = 0
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "evidence": self.evidence,
            "risks": self.risks,
            "recommended_next_action": self.recommended_next_action,
        }
        if self.error:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# Readonly worker task schema
# ---------------------------------------------------------------------------


@dataclass
class WorkerTask:
    task_id: str
    goal: str
    context: str = ""
    expected_output: str = ""
    allowed_tools: list[str] | None = None


# ---------------------------------------------------------------------------
# Worker runner
# ---------------------------------------------------------------------------


def _build_worker_query(task: WorkerTask) -> str:
    """Build the worker system prompt + user query."""
    allowed_str = ", ".join(task.allowed_tools) if task.allowed_tools else "default readonly set"
    return f"""# Readonly Worker Task

You are a READONLY research worker. Your ONLY job is to complete the goal below.

## Rules (hard constraints)
1. ONLY use readonly tools: {allowed_str}
2. Do NOT write files, modify code, or execute shell commands.
3. Do NOT send messages, change config, or write to memory.
4. Do NOT ask for user confirmation — work independently.
5. Do NOT expand the task beyond the stated goal.
6. Keep your reasoning concise — the parent agent only sees a brief summary.

## Task Goal
{task.goal}

## Context
{task.context or '(none)'}

## Expected Output
{task.expected_output or 'Brief summary of findings with evidence'}

## Output Format
After completing the task, respond with a JSON object using this schema:
```json
{{
  "summary": "<str: core findings, ≤{DELEGATE_WORKER_SUMMARY_MAX_CHARS} chars>",
  "evidence": [
    {{"source": "<file path or data source>", "lines": "<range or key>", "claim": "<what this shows>"}}
  ],
  "risks": ["<any caveats or uncertainties>"],
  "recommended_next_action": "<what the parent agent should do next, or 'none'>"
}}
```

If you cannot complete the task (e.g. insufficient information, access denied), set
"summary" to the reason and include an empty evidence list.

Respond with ONLY the JSON object — no markdown wrapper, no preamble."""


async def run_single_worker(
    *,
    runner: object,
    task: WorkerTask,
    parent_turn_id: str,
    task_index: int,
    timeout: float = DELEGATE_WORKER_TIMEOUT_SECONDS,
) -> WorkerResult:
    """Run a single readonly worker and return its structured result.

    Args:
        runner: AgentRunner instance.
        task: The task to execute.
        parent_turn_id: Parent agent's turn ID (for session isolation).
        task_index: Index of this task (0-based).
        timeout: Maximum time for this worker in seconds.
    """
    start_time = time.monotonic()
    session_id = f"readonly_worker:{parent_turn_id}:{task_index}"

    result = WorkerResult(
        task_id=task.task_id or f"worker_{parent_turn_id}_{task_index}",
        status="failed",
    )

    try:
        # Build worker TurnRequest
        query = _build_worker_query(task)
        actor_ctx = ActorContext.system(
            job_id=f"worker_{parent_turn_id}",
            agent_id="delegate_readonly",
        )
        actor_ctx.session_id = session_id

        allowed_tools = task.allowed_tools if task.allowed_tools else list(_READONLY_WORKER_ALLOWED)

        turn_request = TurnRequest(
            inputs=[
                TurnInput(role="user", content=[{"type": "text", "text": query}]),
            ],
            session_id=session_id,
            user_id="delegate_readonly",
            channel="dashboard",
            ephemeral=True,
            source="delegate_readonly",
            context={
                "actor_context": actor_context_to_dict(actor_ctx),
                "allowed_tools": allowed_tools,
                "tool_policy_source": "delegate_readonly_tasks",
                "runtime_profile": "readonly_worker",
                "max_input_length_override": DELEGATE_WORKER_MAX_INPUT_TOKENS,
                "max_iters_override": DELEGATE_WORKER_MAX_ITERS,
            },
        )

        # Run worker
        accumulated_text: list[str] = []
        completed_text: str | None = None

        async def _run() -> None:
            nonlocal completed_text
            async for event in runner.stream_turns(turn_request):  # type: ignore[union-attr]
                if event.type == TurnEventType.ASSISTANT_DELTA:
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str):
                        accumulated_text.append(delta)
                    elif isinstance(delta, dict) and delta.get("type") == "text":
                        accumulated_text.append(delta.get("text") or "")
                elif event.type == TurnEventType.ASSISTANT_COMPLETED:
                    parts: list[str] = []
                    for part in getattr(event, "content", []) or []:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = (part.get("text") or "").strip()
                            if t:
                                parts.append(t)
                    if parts:
                        completed_text = "".join(parts)

        await asyncio.wait_for(_run(), timeout=timeout)

        duration = time.monotonic() - start_time
        result.duration_seconds = round(duration, 2)

        # Extract the final text — prefer completed, fallback to delta
        raw_output = completed_text or "".join(accumulated_text)
        raw_output = raw_output.strip()

        if not raw_output:
            result.status = "failed"
            result.summary = "Worker produced no output."
            result.error = "empty_output"
            return result

        # Parse JSON from worker output
        parsed = _extract_json(raw_output)

        if parsed:
            result.status = "success"
            result.summary = _truncate(
                parsed.get("summary", ""), DELEGATE_WORKER_SUMMARY_MAX_CHARS
            )
            result.evidence = parsed.get("evidence", [])[:20]
            result.risks = parsed.get("risks", [])[:10]
            result.recommended_next_action = parsed.get("recommended_next_action", "")
        else:
            # Fallback: use raw text as summary
            result.status = "partial"
            result.summary = _truncate(raw_output, DELEGATE_WORKER_SUMMARY_MAX_CHARS)
            result.recommended_next_action = (
                "Worker output was not valid JSON; review summary manually."
            )

        return result

    except asyncio.TimeoutError:
        result.status = "timeout"
        result.error = "timeout"
        result.summary = f"Worker timed out after {timeout}s."
        result.duration_seconds = round(time.monotonic() - start_time, 2)
        return result
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)[:500]
        result.summary = f"Worker failed: {type(exc).__name__}"
        result.duration_seconds = round(time.monotonic() - start_time, 2)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from worker output.

    Handles:
    - Pure JSON
    - JSON inside ```json ... ``` blocks
    - JSON with leading/trailing text
    """
    # Try pure JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 30] + "...[truncated]"
