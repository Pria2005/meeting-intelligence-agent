"""Export reviewed notes as Markdown or PDF (PRD FR-11; Appendix B: both formats supported)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import List
from xml.sax.saxutils import escape

from sqlalchemy.orm import Session

from app.db import models as m
from app.schemas import api as schemas
from app.services import repository as repo
from app.services.textutil import fmt_time

STATUS_LABEL = {"verified": "verified", "weak": "needs review (weak evidence)",
                "unsupported": "needs review (unsupported)", "conflicting": "needs review (conflict)",
                "unverified": "unverified"}


def _refs(item) -> str:
    refs = ", ".join(e.ref + f" @{fmt_time(e.start_time)}" for e in item.evidence)
    return refs or "no evidence"


def build_markdown(db: Session, meeting: m.Meeting, include_transcript: bool = True) -> str:
    notes = repo.build_notes(db, meeting)
    lines: List[str] = [f"# {meeting.title}", ""]
    meta = [f"**Meeting ID:** `{meeting.id}`"]
    if meeting.duration:
        meta.append(f"**Duration:** {fmt_time(meeting.duration)}")
    meta.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines += [" · ".join(meta), ""]
    if meeting.needs_review:
        lines += ["> ⚠️ Some AI-generated items are flagged for review.", ""]
    if notes.summary:
        s = notes.summary
        lines += ["## Summary", "", s.overview, ""]
        if s.purpose:
            lines += [f"**Purpose:** {s.purpose}", ""]
        if s.main_topics:
            lines += ["**Main topics:** " + ", ".join(s.main_topics), ""]
        if s.key_points:
            lines += ["### Key points", ""] + [f"- {p}" for p in s.key_points] + [""]
        if s.outcome:
            lines += [f"**Outcome:** {s.outcome}", ""]
    lines += ["## Decisions", ""]
    lines += [f"- {d.description} _[{d.status}; {STATUS_LABEL.get(d.verification_status, d.verification_status)}; {_refs(d)}]_"
              for d in notes.decisions] or ["_None recorded._"]
    lines += ["", "## Action items", "", "| Task | Owner | Due | Status | Evidence |", "|---|---|---|---|---|"]
    for a in notes.action_items:
        lines.append(f"| {a.task.replace('|', '/')} | {a.owner or '—'} | {a.due_date or '—'} | {a.status} | {_refs(a)} |")
    if not notes.action_items:
        lines.append("| _None recorded._ | | | | |")
    lines += ["", "## Risks and blockers", ""]
    lines += [f"- **{r.kind.replace('_', ' ')}**{' (' + r.severity + ')' if r.severity else ''}: {r.description} _[{_refs(r)}]_"
              for r in notes.risks] or ["_None recorded._"]
    lines += ["", "## Open questions", ""]
    lines += [f"- {q.description} _[{_refs(q)}]_" for q in notes.open_questions] or ["_None recorded._"]
    if include_transcript:
        lines += ["", "## Transcript", ""]
        for seg in repo.segment_views(db, meeting.id):
            who = f"**{seg.speaker}:** " if seg.speaker else ""
            lines.append(f"`[{fmt_time(seg.start)}]` {who}{seg.text}  ")
    return "\n".join(lines) + "\n"


def build_pdf(db: Session, meeting: m.Meeting, include_transcript: bool = True) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    notes = repo.build_notes(db, meeting)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=meeting.title, author="AI Meeting Intelligence Agent")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#1e1b4b"), alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#4338ca"), spaceBefore=12)
    body = ParagraphStyle("body", parent=ss["BodyText"], leading=14)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#6b7280"))
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11)

    def P(text: str, style=body) -> Paragraph:
        return Paragraph(escape(text or ""), style)

    story = [P(meeting.title, h1),
             P(f"Meeting {meeting.id}" + (f" · {fmt_time(meeting.duration)}" if meeting.duration else "")
               + f" · exported {datetime.now().strftime('%Y-%m-%d %H:%M')}", small), Spacer(1, 6)]
    if meeting.needs_review:
        story.append(P("Some AI-generated items are flagged for review.", small))
    if notes.summary:
        s = notes.summary
        story += [P("Summary", h2), P(s.overview)]
        if s.purpose:
            story.append(P("Purpose: " + s.purpose))
        if s.main_topics:
            story.append(P("Main topics: " + ", ".join(s.main_topics)))
        for kp in s.key_points:
            story.append(P("• " + kp))
        if s.outcome:
            story.append(P("Outcome: " + s.outcome))
    story.append(P("Decisions", h2))
    for d in notes.decisions or []:
        story.append(P(f"• {d.description}  [{STATUS_LABEL.get(d.verification_status, '')}; {_refs(d)}]"))
    if not notes.decisions:
        story.append(P("None recorded."))
    story.append(P("Action items", h2))
    if notes.action_items:
        data = [[P(h, cell) for h in ("Task", "Owner", "Due", "Status")]]
        for a in notes.action_items:
            data.append([P(a.task, cell), P(a.owner or "—", cell), P(a.due_date or "—", cell), P(a.status, cell)])
        t = Table(data, colWidths=[90 * mm, 30 * mm, 30 * mm, 24 * mm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d2fe")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t)
    else:
        story.append(P("None recorded."))
    story.append(P("Risks and blockers", h2))
    for r in notes.risks:
        story.append(P(f"• [{r.kind.replace('_', ' ')}] {r.description}"))
    if not notes.risks:
        story.append(P("None recorded."))
    story.append(P("Open questions", h2))
    for q in notes.open_questions:
        story.append(P(f"• {q.description}"))
    if not notes.open_questions:
        story.append(P("None recorded."))
    if include_transcript:
        story.append(P("Transcript", h2))
        for seg in repo.segment_views(db, meeting.id):
            who = f"{seg.speaker}: " if seg.speaker else ""
            story.append(P(f"[{fmt_time(seg.start)}] {who}{seg.text}", cell))
    doc.build(story)
    return buf.getvalue()
