from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_current_user
from app.core.errors import ConflictError, NotFoundError, ValidationFailure
from app.db import models as m
from app.db.database import get_db
from app.schemas import api as schemas
from app.services import export as export_service
from app.services import jobs
from app.services import repository as repo
from app.services import storage
from app.services.demo import create_demo_meetings
from app.services.indexing import index_meeting
from app.services.vectorstore import get_vector_store

router = APIRouter(prefix="/meetings", tags=["meetings"])
logger = logging.getLogger(__name__)


def _title_from(filename: Optional[str]) -> str:
    stem = Path(filename or "Untitled meeting").stem.replace("_", " ").replace("-", " ").strip()
    return (stem[:1].upper() + stem[1:]) if stem else "Untitled meeting"


def _detail(db: Session, meeting: m.Meeting) -> schemas.MeetingDetail:
    base = schemas.MeetingOut.model_validate(meeting).model_dump()
    events = db.scalars(select(m.ProcessingEvent).where(m.ProcessingEvent.meeting_id == meeting.id)
                        .order_by(m.ProcessingEvent.id)).all()
    return schemas.MeetingDetail(
        **base, counts=repo.counts_for(db, [meeting.id])[meeting.id],
        has_audio=bool(meeting.storage_path and Path(meeting.storage_path).is_file()),
        events=[schemas.ProcessingEventOut.model_validate(e) for e in events])


# ---------------------------------------------------------------- upload
@router.post("/upload", response_model=schemas.UploadResponse, status_code=201, summary="Upload and validate audio")
async def upload_meeting(
    file: UploadFile = File(..., description="MP3, WAV or M4A recording"),
    title: Optional[str] = Form(default=None, max_length=255),
    auto_process: bool = Form(default=True, description="Start transcription and analysis immediately"),
    db: Session = Depends(get_db), user: m.User = Depends(get_current_user),
) -> schemas.UploadResponse:
    meeting = m.Meeting(user_id=user.id, title="pending", status=m.MeetingStatus.UPLOADED)
    db.add(meeting)
    db.flush()  # allocates the unique meeting id
    try:
        stored = await storage.save_upload(file, meeting.id)
    except Exception:
        db.rollback()
        raise
    meeting.title = (title or "").strip() or _title_from(stored.display_name)
    meeting.filename = stored.display_name
    meeting.storage_path = str(stored.path)
    meeting.file_size = stored.size
    meeting.content_type = stored.content_type
    meeting.duration = stored.duration
    db.commit()
    db.refresh(meeting)
    message = "File uploaded."
    if auto_process:
        jobs.submit(meeting.id)
        db.refresh(meeting)
        message = "File uploaded. Processing has started."
    return schemas.UploadResponse(meeting_id=meeting.id, status=meeting.status, message=message,
                                  meeting=schemas.MeetingOut.model_validate(meeting))


@router.post("/demo", response_model=List[schemas.MeetingOut], status_code=201,
             summary="Create sample meetings (no audio needed)")
def create_demo(db: Session = Depends(get_db), user: m.User = Depends(get_current_user)) -> List[schemas.MeetingOut]:
    meetings = create_demo_meetings(db, user)
    for mt in meetings:
        if mt.status == m.MeetingStatus.UPLOADED and not jobs.is_active(mt.id):
            jobs.submit(mt.id)
    for mt in meetings:
        db.refresh(mt)
    return [schemas.MeetingOut.model_validate(mt) for mt in meetings]


# ------------------------------------------------------------------ list
@router.get("", response_model=List[schemas.MeetingListItem], summary="List meetings")
def list_meetings(
    status: Optional[str] = Query(default=None, pattern="^(uploaded|queued|processing|completed|failed)$"),
    q: Optional[str] = Query(default=None, max_length=100, description="Title search"),
    limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db), user: m.User = Depends(get_current_user),
) -> List[schemas.MeetingListItem]:
    stmt = select(m.Meeting).where(m.Meeting.user_id == user.id)
    if status:
        stmt = stmt.where(m.Meeting.status == status)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(m.Meeting.title.ilike(like), m.Meeting.filename.ilike(like)))
    meetings = db.scalars(stmt.order_by(m.Meeting.created_at.desc()).limit(limit).offset(offset)).all()
    counts = repo.counts_for(db, [x.id for x in meetings])
    out = []
    for mt in meetings:
        summary = repo.latest_summary(db, mt.id)
        item = schemas.MeetingListItem(**schemas.MeetingOut.model_validate(mt).model_dump(), counts=counts[mt.id],
                                       overview=summary.overview if summary else None)
        out.append(item)
    return out


@router.get("/{meeting_id}", response_model=schemas.MeetingDetail, summary="Meeting details and processing state")
def get_meeting(meeting_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    return _detail(db, repo.get_meeting(db, user, meeting_id))


@router.patch("/{meeting_id}", response_model=schemas.MeetingOut, summary="Rename a meeting")
def rename_meeting(meeting_id: str, body: schemas.MeetingUpdate, db: Session = Depends(get_db),
                   user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    mt.title = body.title.strip()
    db.commit()
    return schemas.MeetingOut.model_validate(mt)


# --------------------------------------------------------------- processing
@router.post("/{meeting_id}/transcribe", response_model=schemas.JobResponse, status_code=202,
             summary="Start (or restart) transcription + analysis job")
def transcribe(meeting_id: str, force: bool = Query(default=False, description="Re-transcribe even if a transcript exists"),
               db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    if jobs.is_active(mt.id):
        raise ConflictError("This meeting is already being processed.")
    jobs.submit(mt.id, force_transcribe=force)
    db.refresh(mt)
    return schemas.JobResponse(meeting_id=mt.id, status=mt.status, message="Processing job queued.")


@router.get("/{meeting_id}/events", response_model=List[schemas.ProcessingEventOut], summary="Pipeline stage metrics")
def events(meeting_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    rows = db.scalars(select(m.ProcessingEvent).where(m.ProcessingEvent.meeting_id == mt.id)
                      .order_by(m.ProcessingEvent.id)).all()
    return [schemas.ProcessingEventOut.model_validate(r) for r in rows]


# -------------------------------------------------------------- transcript
@router.get("/{meeting_id}/transcript", response_model=schemas.TranscriptResponse,
            summary="Retrieve timestamped transcript")
def get_transcript(meeting_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    segs = db.scalars(select(m.TranscriptSegment).where(m.TranscriptSegment.meeting_id == mt.id)
                      .options(selectinload(m.TranscriptSegment.speaker)).order_by(m.TranscriptSegment.idx)).all()
    speakers = db.scalars(select(m.Speaker).where(m.Speaker.meeting_id == mt.id).order_by(m.Speaker.id)).all()
    return schemas.TranscriptResponse(
        meeting_id=mt.id, status=mt.status, duration=mt.duration, language=mt.language,
        speakers=[schemas.SpeakerOut.model_validate(s) for s in speakers],
        segments=[_seg_out(s) for s in segs])


def _seg_out(s: m.TranscriptSegment) -> schemas.SegmentOut:
    return schemas.SegmentOut(
        id=s.id, ref=s.ref, idx=s.idx, start_time=s.start_time, end_time=s.end_time, text=s.text,
        speaker_id=s.speaker_id, speaker_label=s.speaker.speaker_label if s.speaker else None,
        speaker_name=repo.speaker_name(s), edited=s.edited)


@router.patch("/{meeting_id}/segments/{segment_id}", response_model=schemas.SegmentOut,
              summary="Correct a transcript segment (text and/or speaker)")
def update_segment(meeting_id: str, segment_id: int, body: schemas.SegmentUpdate, db: Session = Depends(get_db),
                   user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    seg = db.scalar(select(m.TranscriptSegment).where(m.TranscriptSegment.id == segment_id,
                                                      m.TranscriptSegment.meeting_id == mt.id))
    if seg is None:
        raise NotFoundError("Segment not found.")
    if body.text is not None:
        seg.text, seg.edited = " ".join(body.text.split()), True
    if body.clear_speaker:
        seg.speaker_id = None
    elif body.speaker_id is not None:
        sp = db.scalar(select(m.Speaker).where(m.Speaker.id == body.speaker_id, m.Speaker.meeting_id == mt.id))
        if sp is None:
            raise ValidationFailure("Unknown speaker for this meeting.")
        seg.speaker_id, seg.edited = sp.id, True
    db.commit()
    if mt.status == m.MeetingStatus.COMPLETED:
        _reindex(db, mt)
    db.refresh(seg)
    return _seg_out(seg)


@router.patch("/{meeting_id}/speakers/{speaker_id}", response_model=schemas.SpeakerOut,
              summary="Name a speaker (user-confirmed identity)")
def rename_speaker(meeting_id: str, speaker_id: int, body: schemas.SpeakerUpdate, db: Session = Depends(get_db),
                   user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    sp = db.scalar(select(m.Speaker).where(m.Speaker.id == speaker_id, m.Speaker.meeting_id == mt.id))
    if sp is None:
        raise NotFoundError("Speaker not found.")
    name = (body.display_name or "").strip()
    sp.display_name = name or None
    db.commit()
    if mt.status == m.MeetingStatus.COMPLETED:
        _reindex(db, mt)  # chunks embed speaker names
    return schemas.SpeakerOut.model_validate(sp)


def _reindex(db: Session, mt: m.Meeting) -> None:
    try:
        index_meeting(db, mt)
        db.commit()
    except Exception:  # noqa: BLE001 - the edit itself succeeded; surface the problem in logs only
        db.rollback()
        logger.exception("re-index after edit failed")


@router.post("/{meeting_id}/reindex", response_model=schemas.JobResponse, summary="Rebuild the search index")
def reindex(meeting_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    if not db.scalar(select(m.TranscriptSegment.id).where(m.TranscriptSegment.meeting_id == mt.id).limit(1)):
        raise ConflictError("This meeting has no transcript to index yet.")
    index_meeting(db, mt)
    db.commit()
    return schemas.JobResponse(meeting_id=mt.id, status=mt.status, message="Index rebuilt.")


# ----------------------------------------------------------------- summary
@router.get("/{meeting_id}/summary", response_model=schemas.MeetingNotes,
            summary="Summary, decisions, action items, risks (with evidence)")
def get_summary(meeting_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    return repo.build_notes(db, repo.get_meeting(db, user, meeting_id))


@router.patch("/{meeting_id}/summary", response_model=schemas.SummaryOutAPI,
              summary="Edit the summary (stored as a new version)")
def edit_summary(meeting_id: str, body: schemas.SummaryUpdate, db: Session = Depends(get_db),
                 user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    cur = repo.latest_summary(db, mt.id)
    if cur is None:
        raise NotFoundError("No summary has been generated yet.")
    data = body.model_dump(exclude_unset=True)
    new = m.MeetingSummary(
        meeting_id=mt.id, version=cur.version + 1, overview=data.get("overview", cur.overview),
        purpose=data.get("purpose", cur.purpose), main_topics=data.get("main_topics", cur.main_topics),
        key_points=data.get("key_points", cur.key_points), outcome=data.get("outcome", cur.outcome),
        generated_by=cur.generated_by, prompt_version=cur.prompt_version, edited_by_user=True)
    db.add(new)
    db.commit()
    db.refresh(new)
    return schemas.SummaryOutAPI.model_validate(new)


# ------------------------------------------------------------ audio / export
@router.get("/{meeting_id}/audio", summary="Stream the original recording")
def get_audio(meeting_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    if not mt.storage_path or not Path(mt.storage_path).is_file():
        raise NotFoundError("No audio file is stored for this meeting.")
    return FileResponse(mt.storage_path, media_type=mt.content_type or "application/octet-stream",
                        filename=mt.filename or "recording")


@router.get("/{meeting_id}/export", summary="Export reviewed notes as Markdown or PDF")
def export_meeting(meeting_id: str, format: str = Query(default="md", pattern="^(md|pdf)$"),
                   include_transcript: bool = Query(default=True), db: Session = Depends(get_db),
                   user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in mt.title).strip().replace(" ", "_")[:60] or "meeting"
    if format == "pdf":
        data = export_service.build_pdf(db, mt, include_transcript)
        return Response(data, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{safe}.pdf"'})
    md = export_service.build_markdown(db, mt, include_transcript)
    return Response(md, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.md"'})


# ------------------------------------------------------------------ delete
@router.delete("/{meeting_id}", status_code=200, summary="Delete a meeting and all derived data")
def delete_meeting(meeting_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    mt = repo.get_meeting(db, user, meeting_id)
    if jobs.is_active(mt.id):
        raise ConflictError("This meeting is being processed. Try again when it finishes.")
    path = mt.storage_path
    try:
        get_vector_store().delete_meeting(mt.id)
    except Exception:  # noqa: BLE001
        logger.exception("could not delete vectors")
    db.delete(mt)  # cascades to every child table
    db.commit()
    storage.delete_audio(path)
    return {"deleted": True, "meeting_id": meeting_id}
