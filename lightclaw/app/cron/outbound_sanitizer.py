"""OutboundMessageSanitizer: quality gate for final outbound messages.

Applies cleanup rules before a proactive/heartbeat message is dispatched
to a channel.  Detects duplicates, meta-discourse, and role confusion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phrase lists for meta-discourse detection
# ---------------------------------------------------------------------------

_META_PATTERNS: list[str] = [
    "我先看看",
    "我来检查",
    "根据记忆",
    "根据系统提示",
    "按新规则",
    "recall",
    "tool",
    "heartbeat",
    "proactive",
    "让我",
    "我来",
]

# ---------------------------------------------------------------------------
# Patterns that indicate the assistant is impersonating the user.
# ---------------------------------------------------------------------------

_ROLE_CONFUSION_PATTERNS: list[str] = [
    # Assistant referring to self as "我" while the user is "你",
    # especially when mentioning user identity attributes.
    r"你(是|作为|身为|这个).{0,20}(都|还|也).{0,10}(不知道|不清楚|不明白).{0,5}我.{0,10}(哪儿|哪里|怎么|上哪儿|上哪里)",
    r"我.{0,5}上哪儿.{0,5}知道",
    # Assistant acting like it was asked a question directly.
    r"(这|这个|你这).{0,5}(问倒|难倒).{0,3}我",
    r"你这问题.{0,5}(真|挺|好|难|刁)",
    # Assistant mockingly referencing the user.
    r"你(不是|可|还).{0,5}(常务副会长|会长|群主|管理员|大佬|大神).{0,10}(吗|么|吧)",
    # Assistant dismissing user's identity-based knowledge claim.
    r"你(自己|都).{0,10}(不|没).{0,5}(知道|懂|清楚|明白).{0,10}(还|来).{0,5}(问我|问我来了|来问我)",
]

# Phrases that strongly suggest the assistant is constructing a fake reply
# to an imaginary user question.
_IMAGINARY_REPLY_OPENERS: list[str] = [
    "这你问倒我了",
    "你这个问题",
    "你问我",
    "你问的这个问题",
    "你问得好",
    "好问题",
]


# ---------------------------------------------------------------------------
# SanitizeResult
# ---------------------------------------------------------------------------


@dataclass
class SanitizeResult:
    """Result of applying ``OutboundMessageSanitizer.sanitize()``."""

    text: str  # cleaned text
    original_len: int  # length before cleaning
    cleaned_len: int  # length after cleaning
    actions: list[str] = field(default_factory=list)  # e.g. ["strip_whitespace"]
    warnings: list[str] = field(default_factory=list)  # non-fatal warnings
    should_block: bool = False  # block dispatch entirely
    block_reason: str = ""  # reason if should_block


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Remove <thinking>…</thinking> blocks and leading/trailing whitespace."""
    return re.compile(r"<thinking>.*?</thinking>", re.DOTALL).sub("", text).strip()


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets of strings."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams from *text*."""
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


# ---------------------------------------------------------------------------
# OutboundMessageSanitizer
# ---------------------------------------------------------------------------


class OutboundMessageSanitizer:
    """Apply quality-gate rules to an outbound message before dispatch.

    Usage::

        sanitizer = OutboundMessageSanitizer()
        result = sanitizer.sanitize(text, context={"source": "proactivity"})
        if result.should_block:
            logger.warning("Blocked: %s", result.block_reason)
            return
        final_text = result.text
    """

    # ------------------------------------------------------------------
    # Configuration knobs
    # ------------------------------------------------------------------

    min_repeat_chunk: int = 4
    """Minimum characters in each half for exact-repeat (ABAB) collapse."""

    near_repeat_threshold: float = 0.60
    """Jaccard similarity threshold to consider two chunks near-duplicates."""

    near_repeat_chunk_size: int = 20
    """Sliding-window chunk size (characters) for near-duplicate detection."""

    near_repeat_ngram_n: int = 2
    """n-gram size for near-duplicate comparison."""

    # ------------------------------------------------------------------
    # Rule (a): strip whitespace
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_strip(text: str) -> str:
        stripped = text.strip()
        if stripped != text:
            logger.info("sanitizer: stripped leading/trailing whitespace")
        return stripped

    # ------------------------------------------------------------------
    # Rule (b): collapse exact repeat (ABAB, AAAA, …)
    # ------------------------------------------------------------------

    def _collapse_exact_repeat(self, text: str) -> str:
        """Recursively halve *text* while both halves are identical."""
        iterations = 0
        while True:
            n = len(text)
            if n < self.min_repeat_chunk * 2:
                break
            mid = n // 2
            first = text[:mid]
            second = text[mid : mid * 2]
            # Allow trailing chars (odd-length) to differ by a small margin
            if first and second and first == second:
                text = first
                iterations += 1
                logger.info(
                    "sanitizer: collapsed exact repeat (iter=%d, new_len=%d)",
                    iterations,
                    len(text),
                )
            else:
                break
        return text

    # ------------------------------------------------------------------
    # Rule (c): collapse near-repeat
    # ------------------------------------------------------------------

    def _collapse_near_repeat(self, text: str) -> str:
        """Use sliding-window n-gram similarity to truncate near-duplicate
        paragraphs or blocks."""
        chunk_size = self.near_repeat_chunk_size
        threshold = self.near_repeat_threshold
        ngram_n = self.near_repeat_ngram_n

        if len(text) < chunk_size * 2:
            return text

        # Slide a window and compare each window to the *next* window.
        # If similarity > threshold, truncate before the duplicate.
        prev_ngrams: set[str] | None = None
        truncate_at: int | None = None

        for i in range(0, len(text) - chunk_size, max(chunk_size // 2, 1)):
            window = text[i : i + chunk_size]
            cur_ngrams = _ngrams(window, ngram_n)
            if prev_ngrams is not None:
                sim = _jaccard(prev_ngrams, cur_ngrams)
                if sim >= threshold:
                    truncate_at = i
                    logger.info(
                        "sanitizer: near-repeat at pos=%d, similarity=%.3f",
                        i,
                        sim,
                    )
                    break
            prev_ngrams = cur_ngrams

        if truncate_at is not None:
            return text[:truncate_at].rstrip()
        return text

    # ------------------------------------------------------------------
    # Rule (d): meta-discourse detection (warn only)
    # ------------------------------------------------------------------

    def _detect_meta_discourse(self, text: str) -> list[str]:
        """Scan for debugging / internal-monologue phrases.  Returns
        warning strings for each hit."""
        warnings: list[str] = []
        lower = text.lower()
        for pat in _META_PATTERNS:
            if pat in lower:
                msg = f"meta-discourse phrase detected: {pat!r}"
                warnings.append(msg)
                logger.info("sanitizer: %s", msg)
        return warnings

    # ------------------------------------------------------------------
    # Rule (e): role-confusion detection
    # ------------------------------------------------------------------

    def _detect_role_confusion(self, text: str) -> tuple[bool, str]:
        """Return ``(should_block, reason)`` if the assistant appears to be
        impersonating the user."""
        for pattern in _ROLE_CONFUSION_PATTERNS:
            if re.search(pattern, text):
                reason = f"role-confusion pattern matched: {pattern!r}"
                logger.warning("sanitizer: %s", reason)
                return True, reason

        # Check imaginary-reply openers
        for opener in _IMAGINARY_REPLY_OPENERS:
            if opener in text:
                reason = f"imaginary-reply opener detected: {opener!r}"
                logger.warning("sanitizer: %s", reason)
                return True, reason

        return False, ""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sanitize(self, text: str, context: dict | None = None) -> SanitizeResult:
        """Run all quality-gate rules and return a ``SanitizeResult``.

        Args:
            text: The raw outbound message text.
            context: Optional metadata (e.g. ``{"source": "proactivity"}``).
        """
        original = text
        original_len = len(original)
        actions: list[str] = []
        warnings: list[str] = []
        source = (context or {}).get("source", "unknown")

        # (a) Strip
        text = self._apply_strip(text)
        if len(text) != original_len:
            actions.append("strip_whitespace")

        # (b) Exact repeat collapse
        before = len(text)
        text = self._collapse_exact_repeat(text)
        if len(text) != before:
            actions.append("collapse_exact_repeat")

        # (c) Near-repeat collapse
        before = len(text)
        text = self._collapse_near_repeat(text)
        if len(text) != before:
            actions.append("collapse_near_repeat")

        # (d) Meta-discourse detection (warn only)
        meta_warnings = self._detect_meta_discourse(text)
        warnings.extend(meta_warnings)

        # (e) Role-confusion detection
        should_block, block_reason = self._detect_role_confusion(text)
        if should_block:
            logger.warning(
                "sanitizer: BLOCK outbound message [source=%s]: %s",
                source,
                block_reason,
            )

        # (f) Empty/SKIP/HEARTBEAT_OK after sanitization → auto-block.
        # If all meaningful content has been stripped, collapsed, or was
        # never there to begin with, the message should not be dispatched.
        if not should_block:
            stripped = text.strip()
            if not stripped or stripped in ("[SKIP]", "HEARTBEAT_OK"):
                should_block = True
                block_reason = f"empty_or_skip: text={stripped!r}"
                logger.info(
                    "sanitizer: auto-block empty/skip message [source=%s]: %s",
                    source,
                    block_reason,
                )

        # Summary log
        logger.info(
            "sanitizer: done [source=%s] orig_len=%d cleaned_len=%d "
            "actions=%s warnings=%d block=%s",
            source,
            original_len,
            len(text),
            actions,
            len(warnings),
            should_block,
        )

        return SanitizeResult(
            text=text,
            original_len=original_len,
            cleaned_len=len(text),
            actions=actions,
            warnings=warnings,
            should_block=should_block,
            block_reason=block_reason,
        )
