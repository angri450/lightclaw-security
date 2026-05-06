"""Regex-based MEMORY.md parser — zero LLM calls.

Parses MEMORY.md into structured MemoryEntry objects, extracting:
  - sections / subsections
  - individual bullet/numbered entries with raw text
  - all date references found
  - entry type (preference/decision/config/fact/todo/unknown)
  - persistence tier (permanent/stable/transient/ephemeral)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

# ═══════════════════════════════════════════════════════════════════
# Pattern library — compiled once at module load
# ═══════════════════════════════════════════════════════════════════

# YAML frontmatter block between opening/closing --- lines
_FRONTMATTER_RE: ClassVar = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Top-level sections (## Header)
_SECTION_RE: ClassVar = re.compile(r"^##\s+(.+)$", re.MULTILINE)

# Sub-sections (### Header)
_SUBSECTION_RE: ClassVar = re.compile(r"^###\s+(.+)$", re.MULTILINE)

# Code fences — capture the whole block so we don't parse its internals
_CODE_FENCE_RE: ClassVar = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# Date extractors — ordered by specificity (most specific first)
_DATE_PATTERNS: ClassVar[list[tuple[re.Pattern, str]]] = [
    (re.compile(r"最后更新\s*[:：]?\s*(\d{4}-\d{2}-\d{2})"), "last_update"),
    (re.compile(r"创建[于在]\s*(\d{4}-\d{2}-\d{2})"), "created_date"),
    (re.compile(r"Distilled\s+on\s+(\d{4}-\d{2}-\d{2})"), "distilled_date"),
    (re.compile(r"distilled\s+on\s+(\d{4}-\d{2}-\d{2})"), "distilled_date"),
    (re.compile(r"(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}"), "datetime"),
    (re.compile(r"(\d{4}-\d{2}-\d{2})"), "date"),
]

# ── Entry type classifiers (regex-only fast path) ──

_TYPE_RULES: ClassVar[list[tuple[str, re.Pattern]]] = [
    ("preference", re.compile(r"(偏好|喜欢|习惯|倾向|prefer|讨厌|不想|不愿意)")),
    ("constraint", re.compile(r"(禁止|不允许|必须|要求|不得|不能|不要|千万别)")),
    ("decision", re.compile(r"(确定|决定|选择|方案\S*采用|采纳)")),
    ("todo", re.compile(
        r"^\[[ x]\]|待办|TODO|"
        r"待办\S*[:：]|TODO\S*[:：]|"
        r"尚未完成|需要去做|接下来要"
    )),
    ("config", re.compile(
        r"(key\s*[:：]|api|token|密钥|密码|地址\s*[:：]|端口|路径\s*[:：]|"
        r"http|命令\s*[:：]|command|配置|安装命令|运行命令|mcporter)"
    )),
    ("fact_version", re.compile(r"(版本\s*[:：]|version|最后更新|v\d+\.\d+)")),
    ("fact_weather", re.compile(r"(天气|°C|℃|温度|湿度|晴|雨|阴|多云|风力)")),
    ("fact_transient", re.compile(r"(余额|额度|file_id|测试文档|test|当前\s*MBTI)")),
]

# ── Persistence tier classification ──

_PERSISTENCE_RULES: ClassVar[list[tuple[str, re.Pattern]]] = [
    ("permanent", re.compile(
        r"(人格|名字|人称|时区|语言|MBTI|学生|身份|角色|"
        r"我叫|我是|我的名字)"
    )),
    ("stable", re.compile(
        r"(偏好|禁止|要求|确定|方案|决策|风格|格式|"
        r"写作|引用|列表|emoji|mermaid|流程图|安装|配置|命令|command|"
        r"skill|安装目录|已安装|skillhub)"
    )),
    ("ephemeral", re.compile(
        r"(天气|°C|℃|温度|湿度|晴|雨|阴|多云|风力|"
        r"版本\s*\d|file_id|测试文档|test\s*文档|"
        r"余额|额度|v\d+\.\d+\.\d+|版本：\d|最后更新\s*\d{4})"
    )),
]
# fallback: if nothing matches above -> "transient"

# ── Richness heuristics ──

# Entries that look like boilerplate / template instructions (section intros)
_BOILERPLATE_RE: ClassVar = re.compile(
    r"(往这里写什么|把它当作|举例|格式参考|Skills\s+告诉你|"
    r"这个文件记的是|一切能让你|"
    r"SSH\s+连接信息|执行\s*Skills\s+时用到|设备名称.*路径.*端口)"
)

# Information density: count of meaningful tokens (Chinese chars + words + numbers)
_DENSITY_TOKEN_RE: ClassVar = re.compile(r"[一-鿿]|\w+|\d+")

# Maximum entries per batch for LLM classification fallback
_MAX_LLM_BATCH = 20


# ═══════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """One parsed entry from MEMORY.md."""

    section: str           # e.g. "工具配置", "Distilled on 2026-05-02"
    subsection: str        # e.g. "文档工具配置", "SiliconFlow 配置"
    raw_text: str          # original text (lines joined, stripped)
    line_start: int        # 1-based line in source file
    line_end: int

    # regex-extracted metadata
    dates: dict[str, str] = field(default_factory=dict)
    # e.g. {"date": "2026-05-02", "last_update": "2026-05-02"}

    entry_type: str = "unknown"
    # one of: preference, constraint, decision, todo, config,
    #         fact_version, fact_weather, fact_transient, meta, unknown

    persistence: str = "transient"
    # one of: permanent, stable, transient, ephemeral

    is_boilerplate: bool = False

    # computed by scorer (not set by parser)
    recency_score: float = 0.0
    frequency_score: float = 0.0
    consolidation_score: float = 0.0
    persistence_score: float = 0.0
    richness_score: float = 0.0
    final_score: float = 0.0

    # cross-reference hits (filled by scorer)
    ref_days: set[str] = field(default_factory=set)
    ref_count: int = 0

    @property
    def newest_date(self) -> str | None:
        """Return the most recent date string, or None."""
        if not self.dates:
            return None
        candidates = []
        for v in self.dates.values():
            # v may be a date-only or datetime string
            candidates.append(v[:10])  # YYYY-MM-DD
        candidates.sort(reverse=True)
        return candidates[0]

    @property
    def days_since_newest(self) -> int | None:
        """Days since the newest date found in this entry, or None."""
        d = self.newest_date
        if not d:
            return None
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - dt).days
        except ValueError:
            return None


@dataclass
class MemoryParseResult:
    """Full parse result for a MEMORY.md file."""

    path: Path
    frontmatter: str = ""
    entries: list[MemoryEntry] = field(default_factory=list)
    total_lines: int = 0
    entry_count: int = 0


# ═══════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════


class MemoryParser:
    """Parse MEMORY.md into structured entries using regex only."""

    def __init__(self, working_dir: str | Path) -> None:
        self._root = Path(working_dir)
        self._memory_path = self._root / "MEMORY.md"

    @property
    def memory_path(self) -> Path:
        return self._memory_path

    def parse(self) -> MemoryParseResult:
        """Parse MEMORY.md and return structured result."""
        if not self._memory_path.exists():
            return MemoryParseResult(path=self._memory_path)

        raw = self._memory_path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        result = MemoryParseResult(
            path=self._memory_path,
            total_lines=len(lines),
        )

        # 1. Extract and strip frontmatter
        fm_match = _FRONTMATTER_RE.match(raw)
        content = raw
        line_offset = 0
        if fm_match:
            result.frontmatter = fm_match.group(1)
            content = raw[fm_match.end():]
            line_offset = fm_match.group(0).count("\n")

        # 2. Remove code fences for parsing (restore them for raw_text capture)
        cleaned = _CODE_FENCE_RE.sub("", content)

        # 3. Split into sections
        sections = self._split_sections(cleaned, line_offset, raw)

        # 4. For each section, split into subsections and extract entries
        current_section = ""
        for sec_name, sec_body, sec_start_line in sections:
            current_section = sec_name
            subsections = self._split_subsections(sec_body, sec_start_line, raw)
            for sub_name, sub_body, sub_start_line in subsections:
                entries = self._extract_entries(
                    current_section, sub_name, sub_body, sub_start_line, raw
                )
                result.entries.extend(entries)

        result.entry_count = len(result.entries)
        return result

    # ── section / subsection splitting ──

    @staticmethod
    def _split_sections(
        content: str, line_offset: int, raw: str,
    ) -> list[tuple[str, str, int]]:
        """Split cleaned content into (section_name, section_body, start_line)."""
        sections: list[tuple[str, str, int]] = []
        parts = _SECTION_RE.split(content)
        # parts = [preamble, name1, body1, name2, body2, ...]
        if len(parts) < 2:
            return sections
        # skip preamble if present (text before first ##)
        start_idx = 1 if parts[0].strip() == "" else 1
        for i in range(start_idx, len(parts), 2):
            if i + 1 >= len(parts):
                break
            name = parts[i].strip()
            body = parts[i + 1]
            # approximate line number
            line_no = line_offset + content[: content.find(f"## {name}")].count("\n") + 1
            sections.append((name, body, line_no))
        return sections

    @staticmethod
    def _split_subsections(
        body: str, parent_line: int, raw: str,
    ) -> list[tuple[str, str, int]]:
        """Split section body into (subsection_name, subsection_body, start_line)."""
        subs: list[tuple[str, str, int]] = []
        parts = _SUBSECTION_RE.split(body)
        if len(parts) < 2:
            # No subsections — treat entire body as one unnamed subsection
            if body.strip():
                subs.append(("", body, parent_line))
            return subs

        if parts[0].strip():
            # Text before first ### — unnamed preamble
            subs.append(("", parts[0], parent_line))

        # Find line of first ### to anchor offsets
        first_header_pos = body.find(f"### {parts[1]}")
        base_line = parent_line + body[:first_header_pos].count("\n") if first_header_pos >= 0 else parent_line

        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
            name = parts[i].strip()
            sub_body = parts[i + 1]
            line_no = base_line
            # Recalculate for subsequent subsections
            if i > 1:
                prev_name = parts[i - 2].strip()
                prev_pos = body.find(f"### {prev_name}")
                cur_pos = body.find(f"### {name}")
                if prev_pos >= 0 and cur_pos >= 0:
                    line_no = parent_line + body[prev_pos:cur_pos].count("\n")
            subs.append((name, sub_body, line_no))
        return subs

    # ── entry extraction ──

    @staticmethod
    def _extract_entries(
        section: str,
        subsection: str,
        body: str,
        base_line: int,
        raw: str,
    ) -> list[MemoryEntry]:
        """Extract individual entries from a subsection body."""
        entries: list[MemoryEntry] = []
        lines = body.splitlines()

        # Collect bullet / numbered entries via regex
        bullet_pattern = re.compile(r"^(\s*[-*]|\d+\.)\s+(.+)", re.MULTILINE)

        for match in bullet_pattern.finditer(body):
            text = match.group(2).strip()
            if not text or len(text) < 3:
                continue

            # Find line number
            prefix = body[: match.start()]
            rel_line = prefix.count("\n")
            line_no = base_line + rel_line

            entry = MemoryParser._build_entry(
                section, subsection, text, line_no, line_no
            )
            entries.append(entry)

        return entries

    @staticmethod
    def _build_entry(
        section: str,
        subsection: str,
        text: str,
        line_start: int,
        line_end: int,
    ) -> MemoryEntry:
        """Build a MemoryEntry with all regex-extracted metadata."""
        entry = MemoryEntry(
            section=section,
            subsection=subsection,
            raw_text=text,
            line_start=line_start,
            line_end=line_end,
        )

        # Extract dates
        for pattern, label in _DATE_PATTERNS:
            m = pattern.search(text)
            if m:
                entry.dates[label] = m.group(1)
                break  # first match wins (most specific pattern)

        # Classify type
        entry.entry_type = MemoryParser._classify_type(text, section, subsection)
        entry.persistence = MemoryParser._classify_persistence(text, entry.entry_type)
        entry.is_boilerplate = bool(_BOILERPLATE_RE.search(text))

        # Exclude section instructions from being treated as real entries
        if entry.is_boilerplate:
            entry.entry_type = "meta"

        return entry

    @staticmethod
    def _classify_type(text: str, section: str, subsection: str) -> str:
        """Classify entry type via regex rules."""
        combined = f"{section} {subsection} {text}".lower()
        for etype, pattern in _TYPE_RULES:
            if pattern.search(combined):
                return etype
        return "unknown"

    @staticmethod
    def _classify_persistence(text: str, entry_type: str) -> str:
        """Map entry to a persistence tier."""
        combined = f"{entry_type} {text}".lower()
        for tier, pattern in _PERSISTENCE_RULES:
            if pattern.search(combined):
                return tier
        # Defaults by type
        if entry_type in ("preference", "constraint"):
            return "stable"
        if entry_type in ("config",):
            return "stable"
        if entry_type in ("fact_version", "fact_weather"):
            return "ephemeral"
        if entry_type in ("fact_transient",):
            return "transient"
        return "transient"


# ═══════════════════════════════════════════════════════════════════
# Utility: compute richness score via regex density measurement
# ═══════════════════════════════════════════════════════════════════

def compute_richness(text: str) -> float:
    """Score 0–1 based on information density of text.

    - Counts meaningful tokens (Chinese chars + words + numbers)
    - Penalizes pure URLs, one-liner versions, weather one-liners
    - Saturates at ~200 meaningful tokens
    """
    tokens = len(_DENSITY_TOKEN_RE.findall(text))
    if tokens == 0:
        return 0.0
    # sigmoid-like saturation
    raw = min(1.0, tokens / 120)
    return round(raw, 3)
