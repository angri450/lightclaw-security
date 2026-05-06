#!/usr/bin/env python3
"""Integration verification for the proactive outbound pipeline.

Simulates the full pipeline end-to-end using real implementations:
    1. Feed streaming events into AssistantTurnAccumulator.
    2. Run accumulator output through OutboundMessageSanitizer.
    3. Verify the final outbound text is correct.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/root/.lightclaw/venv/lib/python3.12/site-packages")

from lightclaw.app.cron.turn_accumulator import AssistantTurnAccumulator
from lightclaw.app.cron.outbound_sanitizer import OutboundMessageSanitizer


def run_pipeline(
    *,
    deltas: list[str] | None = None,
    thinking: list[str] | None = None,
    completed: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_results: list[dict] | None = None,
) -> str:
    """Accumulate + sanitize, return final outbound text."""
    acc = AssistantTurnAccumulator(turn_id="verify")
    for t in (thinking or []):
        acc.add_thinking(t)
    for d in (deltas or []):
        acc.add_delta(d)
    for tc in (tool_calls or []):
        acc.record_tool_call(tc)
    for tr in (tool_results or []):
        acc.record_tool_result(tr)
    if completed is not None:
        acc.set_completed(completed, source="verify")

    turn = acc.assemble()
    raw = turn.outbound_text
    sanitizer = OutboundMessageSanitizer()
    result = sanitizer.sanitize(raw, context={"source": "integration_test"})
    if result.should_block:
        print(f"  [BLOCKED] {result.block_reason} — returning empty")
        return ""
    return result.text


def simulate_proactive_pipeline() -> int:
    """Run all verification cases. Returns 0 on success, 1 on failure."""
    failures = 0
    sanitizer = OutboundMessageSanitizer()

    def check(desc: str, actual: str, expected_substr: str,
              *, must_not_contain: str = "", max_count: int = -1) -> None:
        nonlocal failures
        ok = expected_substr in actual
        if ok and must_not_contain:
            ok = must_not_contain not in actual
        if ok and max_count >= 0:
            ok = actual.count(expected_substr) <= max_count
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            failures += 1
            print(f"         actual={actual!r}")
            print(f"         expected_substr={expected_substr!r}")
            if must_not_contain:
                print(f"         must_not_contain={must_not_contain!r}")
            if max_count >= 0:
                print(f"         actual.count={actual.count(expected_substr)}, max={max_count}")

    # ------------------------------------------------------------------
    # Case 1: Normal single call
    # ------------------------------------------------------------------
    print("\nCase 1: Normal single call")
    result_1 = run_pipeline(
        deltas=["你好，最近怎么样？"],
        completed="你好，最近怎么样？",
    )
    check("clean msg delivered", result_1, "你好，最近怎么样？")

    # ------------------------------------------------------------------
    # Case 2: Tool call + reply (proactive preamble bug scenario)
    # ------------------------------------------------------------------
    print("\nCase 2: Tool call + reply (proactive preamble)")
    result_2 = run_pipeline(
        deltas=["\n\n\n", "\n\n", "最近农研会那边忙完了吗？"],
        completed="最近农研会那边忙完了吗？",
        tool_calls=[{"name": "memory_search"}],
        tool_results=[{"name": "memory_search"}],
    )
    check("reply appears exactly once", result_2,
          "最近农研会", max_count=1)
    check("no leading triple-newline in outbound", result_2,
          "最近农研会", must_not_contain="\n\n\n")

    # ------------------------------------------------------------------
    # Case 3: Thinking stream exclusion
    # ------------------------------------------------------------------
    print("\nCase 3: Thinking stream exclusion")
    result_3 = run_pipeline(
        thinking=["让我想想..."],
        deltas=["好的，我来帮你查。"],
        completed="好的，我来帮你查。",
    )
    check("thinking excluded from outbound", result_3,
          "好的，我来帮你查。", must_not_contain="让我想想")

    # ------------------------------------------------------------------
    # Case 4: Exact repeat in message (sanitizer collapse)
    # ------------------------------------------------------------------
    print("\nCase 4: Exact repeat collapse in sanitizer")
    r = sanitizer.sanitize("你好世界你好世界", context={"source": "test"})
    if r.text.count("你好世界") == 1 and "collapse_exact_repeat" in r.actions:
        print("  [PASS] folded exact repeat — single occurrence")
    else:
        print(f"  [FAIL] folded exact repeat — count={r.text.count('你好世界')}, actions={r.actions}, text={r.text!r}")
        failures += 1

    # ------------------------------------------------------------------
    # Case 5: Role reversal detection -> blocked
    # ------------------------------------------------------------------
    print("\nCase 5: Role reversal detection")
    bad_msg = "这你问倒我了😂 你是常务副会长，你都不知道的事儿我上哪儿知道去。"
    r = sanitizer.sanitize(bad_msg, context={"source": "test"})
    if r.should_block:
        print(f"  [PASS] role reversal blocked: {r.block_reason}")
    else:
        print(f"  [FAIL] role reversal NOT blocked. warnings={r.warnings}, actions={r.actions}")
        failures += 1

    # ------------------------------------------------------------------
    # Case 6: Meta-discourse detection
    # ------------------------------------------------------------------
    print("\nCase 6: Meta-discourse detection")
    r = sanitizer.sanitize("根据记忆，你之前提到过要测试 Wan2.2。", context={"source": "test"})
    found_meta = any("meta-discourse" in w for w in r.warnings)
    if found_meta:
        print("  [PASS] meta-discourse detected")
    else:
        print(f"  [FAIL] meta-discourse NOT detected. warnings={r.warnings}")
        failures += 1

    # ------------------------------------------------------------------
    # Case 7: Completed authoritative over delta
    # ------------------------------------------------------------------
    print("\nCase 7: Completed authoritative over delta")
    result_7 = run_pipeline(
        deltas=["今天天气不错。"],
        completed="今天天气不错。",
    )
    check("single occurrence", result_7, "今天天气不错。", max_count=1)

    # ------------------------------------------------------------------
    # Case 8: Pre-tool filler text excluded (completed_text wins)
    # ------------------------------------------------------------------
    print("\nCase 8: Pre-tool filler text excluded")
    result_8 = run_pipeline(
        deltas=["让我查一下\n\n", "查询结果：明天晴天。"],
        completed="查询结果：明天晴天。",
        tool_calls=[{"name": "search"}],
        tool_results=[{"name": "search"}],
    )
    check("filler excluded", result_8,
          "查询结果", must_not_contain="让我查一下")

    # ------------------------------------------------------------------
    # Case 9 (new): SKIP marker auto-blocked by sanitizer
    # ------------------------------------------------------------------
    print("\nCase 9: SKIP marker auto-blocked")
    r = sanitizer.sanitize("[SKIP]", context={"source": "test"})
    if r.should_block and "empty_or_skip" in r.block_reason:
        print("  [PASS] [SKIP] auto-blocked")
    else:
        print(f"  [FAIL] [SKIP] NOT blocked. should_block={r.should_block}, reason={r.block_reason!r}")
        failures += 1

    # ------------------------------------------------------------------
    # Case 10 (new): HEARTBEAT_OK auto-blocked by sanitizer
    # ------------------------------------------------------------------
    print("\nCase 10: HEARTBEAT_OK auto-blocked")
    r = sanitizer.sanitize("HEARTBEAT_OK", context={"source": "test"})
    if r.should_block and "empty_or_skip" in r.block_reason:
        print("  [PASS] HEARTBEAT_OK auto-blocked")
    else:
        print(f"  [FAIL] HEARTBEAT_OK NOT blocked. should_block={r.should_block}, reason={r.block_reason!r}")
        failures += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if failures == 0:
        print("ALL VERIFICATIONS PASSED")
        return 0
    else:
        print(f"{failures} VERIFICATION(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(simulate_proactive_pipeline())
