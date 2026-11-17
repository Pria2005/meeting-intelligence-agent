"""Relational data model (PRD §9). The relational DB is the source of truth;
ChromaDB only stores derived embeddings."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class MeetingStatus:
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("usr"))
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # "auth reference"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    meetings: Mapped[List["Meeting"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("mtg"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # sanitized original name
    storage_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # never exposed by the API
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload | demo
    status: Mapped[str] = mapped_column(String(16), default=MeetingStatus.UPLOADED, index=True)
    stage: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="meetings")
    segments: Mapped[List["TranscriptSegment"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", order_by="TranscriptSegment.idx")
    speakers: Mapped[List["Speaker"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    summaries: Mapped[List["MeetingSummary"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", order_by="MeetingSummary.version")
    decisions: Mapped[List["Decision"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    action_items: Mapped[List["ActionItem"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    risks: Mapped[List["Risk"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    chunks: Mapped[List["DocumentChunk"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    events: Mapped[List["ProcessingEvent"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", order_by="ProcessingEvent.id")


class Speaker(Base):
    __tablename__ = "speakers"
    __table_args__ = (UniqueConstraint("meeting_id", "speaker_label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    speaker_label: Mapped[str] = mapped_column(String(64))  # e.g. SPEAKER_00 / Unknown
    display_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)  # user-confirmed name

    meeting: Mapped[Meeting] = relationship(back_populates="speakers")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (UniqueConstraint("meeting_id", "idx"), Index("ix_segment_meeting_time", "meeting_id", "start_time"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)  # 1-based order; ref = seg-0001
    speaker_id: Mapped[Optional[int]] = mapped_column(ForeignKey("speakers.id", ondelete="SET NULL"), nullable=True)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edited: Mapped[bool] = mapped_column(Boolean, default=False)

    meeting: Mapped[Meeting] = relationship(back_populates="segments")
    speaker: Mapped[Optional[Speaker]] = relationship()

    @property
    def ref(self) -> str:
        return segment_ref(self.idx)


def segment_ref(idx: int) -> str:
    return f"seg-{idx:04d}"


class MeetingSummary(Base):
    __tablename__ = "meeting_summaries"
    __table_args__ = (UniqueConstraint("meeting_id", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    overview: Mapped[str] = mapped_column(Text, default="")
    purpose: Mapped[str] = mapped_column(Text, default="")
    main_topics: Mapped[list] = mapped_column(JSON, default=list)
    key_points: Mapped[list] = mapped_column(JSON, default=list)
    outcome: Mapped[str] = mapped_column(Text, default="")
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    generated_by: Mapped[str] = mapped_column(String(80), default="")  # provider:model
    prompt_version: Mapped[str] = mapped_column(String(16), default="")
    edited_by_user: Mapped[bool] = mapped_column(Boolean, default=False)

    meeting: Mapped[Meeting] = relationship(back_populates="summaries")


class _Extracted:
    """Shared columns for LLM-extracted, evidence-linked items."""

    evidence_segment_ids: Mapped[list] = mapped_column(JSON, default=list)
    verification_status: Mapped[str] = mapped_column(String(16), default="unverified")  # verified|weak|unsupported|unverified
    verification_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verification_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    user_edited: Mapped[bool] = mapped_column(Boolean, default=False)


class Decision(Base, _Extracted):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="proposed")  # proposed|confirmed|rejected

    meeting: Mapped[Meeting] = relationship(back_populates="decisions")


class ActionItem(Base, _Extracted):
    __tablename__ = "action_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    task: Mapped[str] = mapped_column(Text)
    owner: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)  # blank when unknown
    due_date: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # ISO date or phrase; blank when unknown
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|in_progress|done|dropped

    meeting: Mapped[Meeting] = relationship(back_populates="action_items")


class Risk(Base, _Extracted):
    """Risks, blockers and unresolved questions (kept apart from decisions, PRD §11.2)."""

    __tablename__ = "risks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="risk")  # risk|blocker|open_question
    description: Mapped[str] = mapped_column(Text)
    severity: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # low|medium|high
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|resolved

    meeting: Mapped[Meeting] = relationship(back_populates="risks")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("meeting_id", "chunk_index"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # deterministic: {meeting_id}:c{index:04d}
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_text: Mapped[str] = mapped_column(Text)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    meeting: Mapped[Meeting] = relationship(back_populates="chunks")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    meeting_id: Mapped[Optional[str]] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), nullable=True,
                                                      index=True)
    item_type: Mapped[str] = mapped_column(String(32))  # summary|decision|action_item|risk|answer|transcript
    item_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    rating: Mapped[int] = mapped_column(Integer)  # -1 / +1, or 1..5
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ProcessingEvent(Base):
    """Observability: per-stage latency, retries, provider/model, token usage (PRD §13, §16.2)."""

    __tablename__ = "processing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # completed|failed|skipped
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    provider: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # short, no transcript content

    meeting: Mapped[Meeting] = relationship(back_populates="events")
