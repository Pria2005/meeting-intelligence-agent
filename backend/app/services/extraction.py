"""Structured extraction over (possibly long) transcripts via map-reduce (PRD §11.1 steps 6-7)."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

from app.core.config import get_settings
from app.schemas.extraction import (ActionItemOut, DecisionOut, MeetingExtraction, RiskOut, SummaryOut,
                                    WindowExtraction, normalize_ref)
from app.services.dtypes import SegmentView
from app.services.llm import LLMService
from app.services.textutil import fmt_time, stems

logger = logging.getLogger(__name__)


def make_windows(segments: Sequence[SegmentView], max_chars: int, overlap_segments: int = 2
                 ) -> List[List[SegmentView]]:
    """Split a transcript into windows that fit the LLM context. Windows overlap by a few segments
    so statements spanning a boundary are not lost (duplicates are removed when merging)."""
    windows: List[List[SegmentView]] = []
    current: List[SegmentView] = []
    size = 0
    for seg in segments:
        length = len(seg.text) + 40
        if current and size + length > max_chars:
            windows.append(current)
            current = list(current[-overlap_segments:]) if overlap_segments else []
            size = sum(len(s.text) + 40 for s in current)
        current.append(seg)
        size += length
    if current:
        windows.append(current)
    return windows


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return a.strip().lower() == b.strip().lower()
    return len(sa & sb) / len(sa | sb) >= threshold


def _merge_refs(a: List[str], b: List[str]) -> List[str]:
    return list(dict.fromkeys([*a, *b]))


def _dedupe(items: list, text_of: Callable) -> list:
    out: list = []
    for it in items:
        for kept in out:
            if _similar(text_of(kept), text_of(it)):
                kept.evidence_segment_ids = _merge_refs(kept.evidence_segment_ids, it.evidence_segment_ids)
                for attr in ("owner", "due_date"):  # keep the more informative copy
                    if hasattr(kept, attr) and not getattr(kept, attr) and getattr(it, attr):
                        setattr(kept, attr, getattr(it, attr))
                break
        else:
            out.append(it)
    return out


def run_extraction(llm: LLMService, meeting_id: str, title: str, segments: Sequence[SegmentView],
                   progress: Optional[Callable[[float], None]] = None) -> MeetingExtraction:
    s = get_settings()
    windows = make_windows(segments, s.llm_max_input_chars)
    parts: List[WindowExtraction] = []
    for i, window in enumerate(windows):
        note = ""
        if len(windows) > 1:
            note = (f"NOTE: this is part {i + 1} of {len(windows)} of a long meeting "
                    f"({fmt_time(window[0].start)} - {fmt_time(window[-1].end)}). Summarise only this part.\n")
        logger.info("Extracting window %d/%d (%d segments)", i + 1, len(windows), len(window))
        parts.append(llm.extract(title, window, note))
        if progress:
            progress((i + 1) / len(windows))

    if len(parts) == 1:
        summary = parts[0].summary
    else:
        summary = llm.merge_summaries(title, [p.summary for p in parts])

    decisions: List[DecisionOut] = _dedupe([d for p in parts for d in p.decisions], lambda d: d.text)
    actions: List[ActionItemOut] = _dedupe([a for p in parts for a in p.action_items], lambda a: a.task)
    risks: List[RiskOut] = _dedupe([r for p in parts for r in p.risks], lambda r: r.text)
    questions: List[RiskOut] = _dedupe([q for p in parts for q in p.open_questions], lambda q: q.text)
    for q in questions:
        q.kind = "open_question"
    return MeetingExtraction(meeting_id=meeting_id, summary=summary, decisions=decisions, action_items=actions,
                             risks=risks, open_questions=questions)


def validate_extraction(extraction: MeetingExtraction, valid_refs: set) -> Tuple[MeetingExtraction, List[str], float]:
    """Schema/evidence sanity pass. Returns (cleaned extraction, issues, fraction of items with valid evidence)."""
    issues: List[str] = []
    total = grounded = 0

    def clean(item) -> None:
        nonlocal total, grounded
        total += 1
        refs = [r for r in (normalize_ref(r) for r in item.evidence_segment_ids) if r in valid_refs]
        if len(refs) != len(item.evidence_segment_ids):
            issues.append("dropped references to non-existent segments")
        item.evidence_segment_ids = list(dict.fromkeys(refs))
        if item.evidence_segment_ids:
            grounded += 1

    for group in (extraction.decisions, extraction.action_items, extraction.risks, extraction.open_questions):
        for item in group:
            clean(item)
    if not extraction.summary.overview.strip():
        issues.append("empty summary overview")
    ratio = grounded / total if total else 1.0
    if total and ratio < 0.5:
        issues.append(f"only {grounded}/{total} extracted items cite valid evidence")
    return extraction, sorted(set(issues)), ratio
