"""Request / response models for the HTTP API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------- auth
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(min_length=8, max_length=128)
    email: Optional[str] = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(ORM):
    id: str
    username: str
    email: Optional[str] = None
    created_at: datetime


class AuthConfig(BaseModel):
    auth_enabled: bool
    allow_registration: bool


# --------------------------------------------------------------- meetings
class MeetingOut(ORM):
    id: str
    title: str
    filename: Optional[str] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    file_size: Optional[int] = None
    source: str = "upload"
    status: str
    stage: Optional[str] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    needs_review: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class MeetingCounts(BaseModel):
    segments: int = 0
    decisions: int = 0
    action_items: int = 0
    risks: int = 0
    chunks: int = 0


class MeetingListItem(MeetingOut):
    counts: MeetingCounts = MeetingCounts()
    overview: Optional[str] = None


class MeetingDetail(MeetingOut):
    counts: MeetingCounts = MeetingCounts()
    has_audio: bool = False
    events: List["ProcessingEventOut"] = []


class MeetingUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class UploadResponse(BaseModel):
    meeting_id: str
    status: str
    message: str
    meeting: MeetingOut


class JobResponse(BaseModel):
    meeting_id: str
    status: str
    message: str


class ProcessingEventOut(ORM):
    stage: str
    status: str
    started_at: datetime
    duration_ms: int
    retries: int = 0
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    detail: Optional[str] = None


# ------------------------------------------------------------- transcript
class SpeakerOut(ORM):
    id: int
    speaker_label: str
    display_name: Optional[str] = None


class SpeakerUpdate(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=120)


class SegmentOut(BaseModel):
    id: int
    ref: str
    idx: int
    start_time: float
    end_time: float
    text: str
    speaker_id: Optional[int] = None
    speaker_label: Optional[str] = None
    speaker_name: Optional[str] = None  # display_name or label
    edited: bool = False


class SegmentUpdate(BaseModel):
    text: Optional[str] = Field(default=None, min_length=1)
    speaker_id: Optional[int] = None
    clear_speaker: bool = False


class TranscriptResponse(BaseModel):
    meeting_id: str
    status: str
    duration: Optional[float] = None
    language: Optional[str] = None
    speakers: List[SpeakerOut] = []
    segments: List[SegmentOut] = []


# ----------------------------------------------------------------- summary
class SummaryOutAPI(ORM):
    version: int
    overview: str
    purpose: str = ""
    main_topics: List[str] = []
    key_points: List[str] = []
    outcome: str = ""
    generated_at: datetime
    generated_by: str = ""
    prompt_version: str = ""
    edited_by_user: bool = False


class SummaryUpdate(BaseModel):
    overview: Optional[str] = None
    purpose: Optional[str] = None
    main_topics: Optional[List[str]] = None
    key_points: Optional[List[str]] = None
    outcome: Optional[str] = None


class EvidenceOut(BaseModel):
    ref: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    text: str


class _ExtractedOut(ORM):
    id: int
    evidence_segment_ids: List[str] = []
    verification_status: str = "unverified"
    verification_score: Optional[float] = None
    verification_note: Optional[str] = None
    needs_review: bool = False
    user_edited: bool = False


class DecisionOutAPI(_ExtractedOut):
    description: str
    status: str = "proposed"
    evidence: List[EvidenceOut] = []


class DecisionUpdate(BaseModel):
    description: Optional[str] = Field(default=None, min_length=1)
    status: Optional[str] = Field(default=None, pattern="^(proposed|confirmed|rejected)$")
    needs_review: Optional[bool] = None


class ActionItemOutAPI(_ExtractedOut):
    meeting_id: str
    task: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "open"
    evidence: List[EvidenceOut] = []


class ActionItemUpdate(BaseModel):
    task: Optional[str] = Field(default=None, min_length=1)
    owner: Optional[str] = None
    due_date: Optional[str] = None
    status: Optional[str] = Field(default=None, pattern="^(open|in_progress|done|dropped)$")
    needs_review: Optional[bool] = None


class RiskOutAPI(_ExtractedOut):
    kind: str
    description: str
    severity: Optional[str] = None
    status: str = "open"
    evidence: List[EvidenceOut] = []


class RiskUpdate(BaseModel):
    description: Optional[str] = Field(default=None, min_length=1)
    status: Optional[str] = Field(default=None, pattern="^(open|resolved)$")
    severity: Optional[str] = Field(default=None, pattern="^(low|medium|high)$")
    needs_review: Optional[bool] = None


class MeetingNotes(BaseModel):
    """Appendix A style structured output, plus review metadata."""

    meeting_id: str
    title: str
    status: str
    needs_review: bool = False
    summary: Optional[SummaryOutAPI] = None
    decisions: List[DecisionOutAPI] = []
    action_items: List[ActionItemOutAPI] = []
    risks: List[RiskOutAPI] = []
    open_questions: List[RiskOutAPI] = []


# -------------------------------------------------------------------- chat
class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    meeting_ids: Optional[List[str]] = Field(default=None, description="Restrict to these meetings; null = all authorised")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    history: List[ChatMessage] = []


class Citation(BaseModel):
    label: str
    meeting_id: str
    meeting_title: str
    segment_refs: List[str] = []
    start_time: float
    end_time: float
    speakers: List[str] = []
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    found: bool
    answer_type: str  # direct | synthesized | not_found
    citations: List[Citation] = []
    scope: List[str] = []  # meeting ids actually searched
    model: Optional[str] = None
    latency_ms: int = 0


# ---------------------------------------------------------------- feedback
class FeedbackCreate(BaseModel):
    meeting_id: Optional[str] = None
    item_type: str = Field(pattern="^(summary|decision|action_item|risk|answer|transcript|other)$")
    item_id: Optional[str] = None
    rating: int = Field(ge=-1, le=5)
    comment: Optional[str] = Field(default=None, max_length=2000)


class FeedbackOut(ORM):
    id: int
    item_type: str
    rating: int
    comment: Optional[str] = None
    created_at: datetime


# ------------------------------------------------------------------- stats
class StatsResponse(BaseModel):
    total_meetings: int
    by_status: dict
    total_duration_seconds: float
    open_action_items: int
    total_decisions: int
    open_risks: int
    avg_processing_seconds: Optional[float] = None
    needs_review: int = 0
    llm_provider: str
    llm_model: str
    embedding_backend: str
    transcription_available: bool
    diarization_enabled: bool


class AppInfo(BaseModel):
    name: str
    version: str
    environment: str
    docs_url: str
    auth_enabled: bool
    llm_provider: str
    llm_model: str
    embedding_backend: str
    whisper_model: str
    diarization_enabled: bool
    max_upload_mb: int
    allowed_extensions: List[str]


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_store: str
    time: datetime
    details: dict[str, Any] = {}


UserOut.model_rebuild()
TokenResponse.model_rebuild()
MeetingDetail.model_rebuild()
