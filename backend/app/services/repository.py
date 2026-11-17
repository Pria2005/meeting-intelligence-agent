"""Database helpers shared by routes, the pipeline and the RAG service."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import NotFoundError
from app.db import models as m
from app.schemas import api as schemas
from app.services.dtypes import SegmentView

DEFAULT_USER_USERNAME = "local"


def get_or_create_default_user(db: Session) -> m.User:
    user = db.scalar(select(m.User).where(m.User.username == DEFAULT_USER_USERNAME))
    if user is None:
        user = m.User(username=DEFAULT_USER_USERNAME)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def get_meeting(db: Session, user: m.User, meeting_id: str) -> m.Meeting:
    """Authorisation boundary: foreign meetings are indistinguishable from missing ones (404)."""
    meeting = db.scalar(select(m.Meeting).where(m.Meeting.id == meeting_id, m.Meeting.user_id == user.id))
    if meeting is None:
        raise NotFoundError("Meeting not found.")
    return meeting


def speaker_name(seg: m.TranscriptSegment) -> Optional[str]:
    if seg.speaker is None:
        return None
    return seg.speaker.display_name or seg.speaker.speaker_label


def segment_views(db: Session, meeting_id: str) -> List[SegmentView]:
    segs = db.scalars(select(m.TranscriptSegment).where(m.TranscriptSegment.meeting_id == meeting_id)
                      .options(selectinload(m.TranscriptSegment.speaker)).order_by(m.TranscriptSegment.idx)).all()
    return [SegmentView(ref=s.ref, start=s.start_time, end=s.end_time, speaker=speaker_name(s), text=s.text)
            for s in segs]


def evidence_for(db: Session, meeting_id: str, refs: Sequence[str]) -> List[schemas.EvidenceOut]:
    if not refs:
        return []
    idxs = []
    for r in refs:
        try:
            idxs.append(int(r.split("-")[-1]))
        except ValueError:
            continue
    if not idxs:
        return []
    segs = db.scalars(select(m.TranscriptSegment).where(m.TranscriptSegment.meeting_id == meeting_id,
                                                        m.TranscriptSegment.idx.in_(idxs))
                      .options(selectinload(m.TranscriptSegment.speaker)).order_by(m.TranscriptSegment.idx)).all()
    return [schemas.EvidenceOut(ref=s.ref, start_time=s.start_time, end_time=s.end_time, speaker=speaker_name(s),
                                text=s.text) for s in segs]


def counts_for(db: Session, meeting_ids: Sequence[str]) -> Dict[str, schemas.MeetingCounts]:
    result = {mid: schemas.MeetingCounts() for mid in meeting_ids}
    if not meeting_ids:
        return result
    for model, field in ((m.TranscriptSegment, "segments"), (m.Decision, "decisions"), (m.ActionItem, "action_items"),
                         (m.Risk, "risks"), (m.DocumentChunk, "chunks")):
        rows = db.execute(select(model.meeting_id, func.count()).where(model.meeting_id.in_(meeting_ids))
                          .group_by(model.meeting_id)).all()
        for mid, n in rows:
            setattr(result[mid], field, n)
    return result


def latest_summary(db: Session, meeting_id: str) -> Optional[m.MeetingSummary]:
    return db.scalar(select(m.MeetingSummary).where(m.MeetingSummary.meeting_id == meeting_id)
                     .order_by(m.MeetingSummary.version.desc()).limit(1))


def decision_out(db: Session, d: m.Decision) -> schemas.DecisionOutAPI:
    out = schemas.DecisionOutAPI.model_validate(d)
    out.evidence = evidence_for(db, d.meeting_id, d.evidence_segment_ids or [])
    return out


def action_out(db: Session, a: m.ActionItem) -> schemas.ActionItemOutAPI:
    out = schemas.ActionItemOutAPI.model_validate(a)
    out.evidence = evidence_for(db, a.meeting_id, a.evidence_segment_ids or [])
    return out


def risk_out(db: Session, r: m.Risk) -> schemas.RiskOutAPI:
    out = schemas.RiskOutAPI.model_validate(r)
    out.evidence = evidence_for(db, r.meeting_id, r.evidence_segment_ids or [])
    return out


def build_notes(db: Session, meeting: m.Meeting) -> schemas.MeetingNotes:
    summary = latest_summary(db, meeting.id)
    decisions = db.scalars(select(m.Decision).where(m.Decision.meeting_id == meeting.id).order_by(m.Decision.id)).all()
    actions = db.scalars(select(m.ActionItem).where(m.ActionItem.meeting_id == meeting.id)
                         .order_by(m.ActionItem.id)).all()
    risks = db.scalars(select(m.Risk).where(m.Risk.meeting_id == meeting.id).order_by(m.Risk.id)).all()
    return schemas.MeetingNotes(
        meeting_id=meeting.id, title=meeting.title, status=meeting.status, needs_review=meeting.needs_review,
        summary=schemas.SummaryOutAPI.model_validate(summary) if summary else None,
        decisions=[decision_out(db, d) for d in decisions],
        action_items=[action_out(db, a) for a in actions],
        risks=[risk_out(db, r) for r in risks if r.kind != "open_question"],
        open_questions=[risk_out(db, r) for r in risks if r.kind == "open_question"])


def refresh_review_flag(db: Session, meeting: m.Meeting) -> None:
    flagged = any(
        db.scalar(select(func.count()).where(model.meeting_id == meeting.id, model.needs_review.is_(True))) or 0
        for model in (m.Decision, m.ActionItem, m.Risk))
    meeting.needs_review = bool(flagged)
