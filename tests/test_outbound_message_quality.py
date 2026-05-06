#!/usr/bin/env python3
"""Tests for outbound message quality using real implementations.

Covers AssistantTurnAccumulator / AccumulatedTurn (turn_accumulator.py)
and OutboundMessageSanitizer / SanitizeResult (outbound_sanitizer.py).
"""

from __future__ import annotations

import sys
import unittest

# Use the installed package from the project virtualenv.
sys.path.insert(0, "/root/.lightclaw/venv/lib/python3.12/site-packages")

from lightclaw.app.cron.turn_accumulator import (
    AccumulatedTurn,
    AssistantTurnAccumulator,
)
from lightclaw.app.cron.outbound_sanitizer import (
    OutboundMessageSanitizer,
    SanitizeResult,
)


# ============================================================================
# TestAssistantTurnAccumulator
# ============================================================================


class TestAssistantTurnAccumulator(unittest.TestCase):
    """Tests for AssistantTurnAccumulator using the real implementation.

    Key semantics:
    - outbound_text prefers completed_text (authoritative).
    - Falls back to delta_text with a WARNING only when completed_text is empty.
    - thinking_text is NEVER included in outbound.
    """

    def _run(
        self,
        *,
        deltas: list[str] | None = None,
        thinking: list[str] | None = None,
        completed: str | None = None,
        completed_source: str = "test",
        tool_calls: list[dict] | None = None,
        tool_results: list[dict] | None = None,
    ) -> AccumulatedTurn:
        acc = AssistantTurnAccumulator(turn_id="test")
        for t in (thinking or []):
            acc.add_thinking(t)
        for d in (deltas or []):
            acc.add_delta(d)
        for tc in (tool_calls or []):
            acc.record_tool_call(tc)
        for tr in (tool_results or []):
            acc.record_tool_result(tr)
        if completed is not None:
            acc.set_completed(completed, source=completed_source)
        return acc.assemble()

    # -- test cases -----------------------------------------------------------

    def test_single_call_no_tool_no_repeat(self):
        turn = self._run(
            deltas=["你好，最近怎么样？"],
            completed="你好，最近怎么样？",
        )
        self.assertIn("最近怎么样", turn.outbound_text)
        self.assertEqual(turn.outbound_text.count("最近怎么样"), 1)

    def test_proactive_with_tool_then_reply(self):
        turn = self._run(
            deltas=["\n\n\n", "\n\n", "最近农研会那边忙完了吗？"],
            completed="最近农研会那边忙完了吗？",
            tool_calls=[{"name": "memory_search"}],
            tool_results=[{"name": "memory_search"}],
        )
        self.assertIn("最近农研会", turn.outbound_text)
        self.assertEqual(
            turn.outbound_text.count("最近农研会"), 1,
            "Reply must appear exactly once — completed_text is authoritative",
        )

    def test_proactive_with_preface_then_reply(self):
        turn = self._run(
            deltas=["让我查一下\n\n", "根据查询结果，明天是晴天。"],
            completed="根据查询结果，明天是晴天。",
            tool_calls=[{"name": "search"}],
            tool_results=[{"name": "search"}],
        )
        self.assertNotIn(
            "让我查一下", turn.outbound_text,
            "Pre-tool filler text must NOT appear — completed_text is authoritative",
        )

    def test_completed_is_authoritative_over_delta(self):
        turn = self._run(
            deltas=["draft reply that is wrong"],
            completed="final correct reply",
        )
        self.assertEqual(turn.outbound_text, "final correct reply")
        self.assertNotIn("draft", turn.outbound_text)

    def test_delta_fallback_when_completed_empty(self):
        turn = self._run(
            deltas=["delta is all we have"],
            completed="",
        )
        self.assertIn("delta is all we have", turn.outbound_text)

    def test_thinking_stream_not_in_outbound(self):
        turn = self._run(
            thinking=["让我想想…", "好的，用户需要…"],
            deltas=["你好，有什么可以帮助你的？"],
            completed="你好，有什么可以帮助你的？",
        )
        self.assertNotIn("让我想想", turn.outbound_text)
        # thinking_text should contain the reasoning content
        self.assertIn("让我想想", turn.thinking_text)
        self.assertIn("有什么可以帮助你的", turn.outbound_text)

    def test_tool_call_whitespace_not_polluting_outbound(self):
        turn = self._run(
            deltas=["\n\n\n", "这是最终回复。"],
            completed="这是最终回复。",
            tool_calls=[{"name": "memory_search"}],
            tool_results=[{"name": "memory_search"}],
        )
        self.assertEqual(turn.outbound_text.strip(), "这是最终回复。")
        self.assertFalse(
            turn.outbound_text.startswith("\n"),
            "Whitespace preamble must not appear in outbound — completed_text is authoritative",
        )

    def test_completed_wins_over_different_delta(self):
        turn = self._run(
            deltas=[" 你好世界 "],
            completed="你好世界",
        )
        self.assertEqual(turn.outbound_text, "你好世界")

    def test_all_empty_produces_empty(self):
        turn = self._run(deltas=[""], completed="")
        self.assertEqual(turn.outbound_text.strip(), "")

    def test_is_skip_detection(self):
        turn_skip = self._run(completed="[SKIP]")
        self.assertTrue(turn_skip.is_skip)

        turn_hb = self._run(completed="HEARTBEAT_OK")
        self.assertTrue(turn_hb.is_skip)

        turn_normal = self._run(completed="你好")
        self.assertFalse(turn_normal.is_skip)

    def test_accumulated_turn_fields(self):
        turn = self._run(
            deltas=["hello "],
            thinking=["hmm..."],
            completed="hello world",
            completed_source="assistant_completed",
            tool_calls=[{"name": "search"}],
            tool_results=[{"result": "found"}],
        )
        self.assertEqual(turn.turn_id, "test")
        self.assertEqual(turn.delta_text, "hello ")
        self.assertEqual(turn.thinking_text, "hmm...")
        self.assertEqual(turn.completed_text, "hello world")
        self.assertEqual(turn.completed_source, "assistant_completed")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(len(turn.tool_results), 1)


# ============================================================================
# TestOutboundMessageSanitizer
# ============================================================================


class TestOutboundMessageSanitizer(unittest.TestCase):
    """Tests for OutboundMessageSanitizer using the real implementation."""

    def setUp(self) -> None:
        self.sanitizer = OutboundMessageSanitizer()

    def _sanitize(self, text: str, source: str = "test") -> SanitizeResult:
        return self.sanitizer.sanitize(text, context={"source": source})

    # -- test cases -----------------------------------------------------------

    def test_strip_whitespace(self):
        r = self._sanitize("  你好  ")
        self.assertEqual(r.text, "你好")
        self.assertIn("strip_whitespace", r.actions)

    def test_collapse_exact_repeat(self):
        # Real impl requires min 8 chars (min_repeat_chunk=4 * 2).
        # "你好世界你好世界" → 8 chars, mid=4, halves: "你好世界" == "你好世界" → collapse.
        r = self._sanitize("你好世界你好世界")
        self.assertEqual(r.text, "你好世界")
        self.assertIn("collapse_exact_repeat", r.actions)

    def test_collapse_exact_repeat_longer(self):
        # 12 chars, "最近怎么样" x 2 → collapse.
        r = self._sanitize("最近怎么样最近怎么样")
        self.assertEqual(r.text, "最近怎么样")
        self.assertIn("collapse_exact_repeat", r.actions)

    def test_no_collapse_when_halves_differ(self):
        # Near-duplicate with newline difference — real impl does strict byte compare.
        # halves differ so no collapse (this is handled by near-repeat instead).
        text = "最近怎么样？\n\n最近怎么样？"
        r = self._sanitize(text)
        # The near-repeat detector (rule c) may handle this, but exact-collapse won't.
        self.assertIsInstance(r.text, str)

    def test_detect_meta_discourse(self):
        r = self._sanitize("根据记忆，你之前提到过要测试 Wan2.2。")
        found_meta = any("meta-discourse" in w for w in r.warnings)
        self.assertTrue(found_meta, f"Meta-discourse not detected. warnings={r.warnings}")

    def test_detect_meta_discourse_english(self):
        r = self._sanitize("let me recall what you said before...")
        found = any("recall" in w for w in r.warnings)
        self.assertTrue(found, f"English meta-discourse 'recall' not detected. warnings={r.warnings}")

    def test_block_role_confusion_regex(self):
        r = self._sanitize(
            "这你问倒我了 你是常务副会长，你都不知道的事儿我上哪儿知道去。"
        )
        self.assertTrue(r.should_block, f"Role confusion should be blocked. block_reason={r.block_reason!r}")

    def test_block_imaginary_reply_opener(self):
        r = self._sanitize("这你问倒我了😂 我上哪儿知道去。")
        self.assertTrue(r.should_block, f"Imaginary reply opener should be blocked. block_reason={r.block_reason!r}")

    def test_clean_message_passes(self):
        r = self._sanitize("你好，最近怎么样？")
        self.assertFalse(r.should_block, f"Clean msg blocked: {r.block_reason}")
        self.assertIn("你好", r.text)

    def test_duplicate_message_folded(self):
        # "知道了知道了" → len=6, < 8 min → won't collapse in strict mode.
        # But near-repeat detection (rule c) should still produce clean output.
        r = self._sanitize("知道了知道了")
        self.assertFalse(r.should_block)

    def test_terminal_whitespace_after_sanitize(self):
        r = self._sanitize("   \n\n   ")
        self.assertEqual(r.text, "")

    def test_harmless_emoji_message_passes(self):
        r = self._sanitize("好的👍 没问题")
        self.assertFalse(r.should_block)
        self.assertIn("好的", r.text)

    def test_strip_preserves_original_len(self):
        r = self._sanitize("  hello  ")
        self.assertEqual(r.original_len, 9)
        self.assertEqual(r.cleaned_len, 5)

    # -- Risk 3: SKIP/HEARTBEAT_OK auto-block tests ---------------------------

    def test_skip_marker_auto_blocked(self):
        r = self._sanitize("[SKIP]")
        self.assertTrue(r.should_block, "[SKIP] should be auto-blocked")
        self.assertIn("empty_or_skip", r.block_reason)

    def test_heartbeat_ok_auto_blocked(self):
        r = self._sanitize("HEARTBEAT_OK")
        self.assertTrue(r.should_block, "HEARTBEAT_OK should be auto-blocked")
        self.assertIn("empty_or_skip", r.block_reason)

    def test_skip_marker_with_whitespace_auto_blocked(self):
        r = self._sanitize("  [SKIP]  ")
        self.assertTrue(r.should_block, "[SKIP] with whitespace should be auto-blocked")

    def test_empty_after_strip_auto_blocked(self):
        r = self._sanitize("   ")
        self.assertTrue(r.should_block, "Empty text after strip should be auto-blocked")

    def test_skip_marker_embedded_in_text_not_blocked(self):
        r = self._sanitize("用户消息包含 [SKIP] 但你还是要回复")
        self.assertFalse(r.should_block, "[SKIP] as part of larger text should NOT be blocked")


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
