"""RecallCandidate — structured memory recall candidate with metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class RecallCandidate:
    """A single structured memory recall candidate with full metadata."""

    text: str = ""
    source_type: str = "unknown"
    scope: str = "unknown"
    session_id: str = ""
    session_key: str = ""
    channel: str = ""
    chat_type: str = ""
    chat_id: str = ""
    sender_id: str = ""
    principal_id: str = ""
    trust_level: str = "unknown"
    path: str = ""
    score: float = 0.0
    char_count: int = 0
    token_count: int = 0
    reason: str = ""
    truncated: bool = False
    contains_secret: bool = False
    has_metadata: bool = True
    metadata: dict = field(default_factory=dict)

    def render_for_block(self) -> str:
        truncated_flag = " truncated=true" if self.truncated else ""
        meta_flag = "" if self.has_metadata else " old_metadata=true"
        return (
            f"[source={self.source_type} scope={self.scope}"
            f" session={self.session_id or 'none'}"
            f" principal={self.principal_id or 'unknown'}"
            f" score={self.score:.3f}{truncated_flag}{meta_flag}]\n"
            f"{self.text}"
        )
