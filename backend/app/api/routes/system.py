from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db import models as m
from app.db.database import get_db
from app.schemas import api as schemas
from app.services.embeddings import get_embedder
from app.services.llm import describe_llm
from app.services.transcription import transcription_available
from app.services.vectorstore import get_vector_store

router = APIRouter(tags=["system"])


@router.get("/", response_model=schemas.AppInfo, summary="Service information")
def root() -> schemas.AppInfo:
    s = get_settings()
    llm = describe_llm()
    return schemas.AppInfo(
        name=s.app_name, version=s.app_version, environment=s.environment, docs_url="/docs",
        auth_enabled=s.auth_enabled, llm_provider=llm["provider"], llm_model=llm["model"],
        embedding_backend=get_embedder().name, whisper_model=s.whisper_model_size,
        diarization_enabled=s.diarization_enabled, max_upload_mb=s.max_upload_mb,
        allowed_extensions=s.allowed_extensions)


@router.get("/health", response_model=schemas.HealthResponse, summary="Health check")
def health(db: Session = Depends(get_db)) -> schemas.HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        db_state = "ok"
    except Exception:  # noqa: BLE001
        db_state = "error"
    try:
        vs_state = "ok" if get_vector_store().ping() else "error"
    except Exception:  # noqa: BLE001
        vs_state = "error"
    overall = "ok" if db_state == "ok" and vs_state == "ok" else "degraded"
    return schemas.HealthResponse(status=overall, database=db_state, vector_store=vs_state,
                                  time=datetime.now(timezone.utc),
                                  details={"transcription_available": transcription_available()})


@router.get("/stats", response_model=schemas.StatsResponse, summary="Dashboard statistics")
def stats(db: Session = Depends(get_db), user: m.User = Depends(get_current_user)) -> schemas.StatsResponse:
    s = get_settings()
    rows = db.execute(select(m.Meeting.status, func.count()).where(m.Meeting.user_id == user.id)
                      .group_by(m.Meeting.status)).all()
    by_status = {status: n for status, n in rows}
    mids = select(m.Meeting.id).where(m.Meeting.user_id == user.id)
    total_dur = db.scalar(select(func.coalesce(func.sum(m.Meeting.duration), 0.0)).where(m.Meeting.user_id == user.id))
    open_actions = db.scalar(select(func.count()).where(m.ActionItem.meeting_id.in_(mids),
                                                        m.ActionItem.status.in_(["open", "in_progress"])))
    decisions = db.scalar(select(func.count()).where(m.Decision.meeting_id.in_(mids)))
    open_risks = db.scalar(select(func.count()).where(m.Risk.meeting_id.in_(mids), m.Risk.status == "open"))
    review = db.scalar(select(func.count()).where(m.Meeting.user_id == user.id, m.Meeting.needs_review.is_(True)))
    done = db.execute(select(m.Meeting.created_at, m.Meeting.completed_at)
                      .where(m.Meeting.user_id == user.id, m.Meeting.completed_at.is_not(None))).all()
    avg = None
    if done:
        secs = []
        for created, completed in done:
            if created and completed:
                c = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
                d = completed if completed.tzinfo else completed.replace(tzinfo=timezone.utc)
                secs.append((d - c).total_seconds())
        avg = round(sum(secs) / len(secs), 1) if secs else None
    llm = describe_llm()
    return schemas.StatsResponse(
        total_meetings=sum(by_status.values()), by_status=by_status, total_duration_seconds=float(total_dur or 0),
        open_action_items=open_actions or 0, total_decisions=decisions or 0, open_risks=open_risks or 0,
        avg_processing_seconds=avg, needs_review=review or 0, llm_provider=llm["provider"], llm_model=llm["model"],
        embedding_backend=get_embedder().name, transcription_available=transcription_available(),
        diarization_enabled=s.diarization_enabled)
