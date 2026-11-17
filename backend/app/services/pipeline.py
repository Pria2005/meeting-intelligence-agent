"""LangGraph orchestration of the meeting-processing workflow (PRD §11).

    ingest → transcribe → diarize (optional) → chunk → extract → validate_schema → verify_evidence → persist → index

Conditional edges: failed transcription ends the run; extraction failures still index the transcript so Q&A works;
schema/evidence problems loop back to ``extract`` (bounded) and finally fall through to human review.
Every stage records latency / provider / token usage in ``processing_events`` (PRD §16.2)."""
from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import delete, func, select

from app.core.config import get_settings
from app.core.errors import AppError
from app.db import models as m
from app.db.database import session_scope
from app.schemas.extraction import MeetingExtraction
from app.services import repository as repo
from app.services.chunking import chunk_transcript
from app.services.diarization import DiarizationUnavailable, assign_speakers, get_diarizer
from app.services.dtypes import SegmentView
from app.services.extraction import run_extraction, validate_extraction
from app.services.indexing import index_meeting
from app.services.llm import LLMService, get_llm
from app.services.transcription import RawSegment, TranscriptionUnavailable, get_transcriber
from app.services.verification import verify_extraction

logger = logging.getLogger(__name__)

MAX_EXTRACTION_ATTEMPTS = 2
_asr_lock = threading.Lock()  # one ASR / diarization job at a time: keeps RAM and GPU use predictable


class PipelineState(TypedDict, total=False):
    meeting_id: str
    force_transcribe: bool
    needs_transcription: bool
    segments: List[SegmentView]
    extraction: Optional[MeetingExtraction]
    verdicts: Dict[str, Any]
    issues: List[str]
    attempts: int
    needs_review: bool
    failed: bool
    error: Optional[str]
    chunk_count: int


class TransientPipelineError(RuntimeError):
    """Signals the job runner that retrying the whole job may help."""


# ----------------------------------------------------------------- helpers
def _progress(meeting_id: str, stage: str, fraction: float, status: Optional[str] = None) -> None:
    with session_scope() as db:
        mt = db.get(m.Meeting, meeting_id)
        if mt is None:
            return
        mt.stage = stage
        mt.progress = max(mt.progress or 0.0, min(1.0, fraction))  # progress never moves backwards
        if status:
            mt.status = status


@contextmanager
def _stage(meeting_id: str, name: str, llm: Optional[LLMService] = None):
    """Record a ProcessingEvent for the wrapped block."""
    info: Dict[str, Any] = {"status": "completed", "detail": None, "retries": 0}
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    usage0 = (llm.usage.input_tokens, llm.usage.output_tokens, llm.usage.retries) if llm else (0, 0, 0)
    try:
        yield info
    except Exception as exc:  # noqa: BLE001
        info["status"], info["detail"] = "failed", type(exc).__name__
        raise
    finally:
        try:
            with session_scope() as db:
                ev = m.ProcessingEvent(meeting_id=meeting_id, stage=name, status=info["status"], started_at=started_at,
                                       duration_ms=int((time.perf_counter() - t0) * 1000), detail=info["detail"],
                                       retries=info.get("retries", 0))
                if llm is not None:
                    ev.provider, ev.model = llm.provider, llm.model
                    ev.input_tokens = llm.usage.input_tokens - usage0[0]
                    ev.output_tokens = llm.usage.output_tokens - usage0[1]
                    ev.retries = (llm.usage.retries - usage0[2]) + info.get("retries", 0)
                db.add(ev)
        except Exception:  # noqa: BLE001 - observability must never break the job
            logger.warning("could not record processing event for stage %s", name)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ------------------------------------------------------------------- graph
def build_graph(llm: LLMService):
    def ingest(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        with _stage(mid, "ingest") as ev:
            with session_scope() as db:
                mt = db.get(m.Meeting, mid)
                if mt is None:
                    raise AppError("Meeting disappeared before processing started.")
                n_segments = db.scalar(select(func.count()).where(m.TranscriptSegment.meeting_id == mid)) or 0
                has_audio = bool(mt.storage_path and Path(mt.storage_path).is_file())
                needs = bool(state.get("force_transcribe")) or n_segments == 0
                if needs and not has_audio:
                    raise AppError("The stored audio file is missing, so the meeting cannot be transcribed.")
                mt.status, mt.stage, mt.error_message = m.MeetingStatus.PROCESSING, "ingest", None
                mt.progress = 0.02
                ev["detail"] = "reprocessing existing transcript" if not needs else "audio ready"
        return {"needs_transcription": needs, "attempts": 0, "issues": [], "failed": False, "error": None,
                "needs_review": False}

    def transcribe(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        if not state.get("needs_transcription"):
            with _stage(mid, "transcribe") as ev:
                ev["status"], ev["detail"] = "skipped", "existing transcript reused"
            return {}
        _progress(mid, "transcribe", 0.03)
        with _stage(mid, "transcribe") as ev:
            with session_scope() as db:
                path = db.get(m.Meeting, mid).storage_path
            transcriber = get_transcriber()
            last = [0.0]

            def on_progress(frac: float) -> None:
                if frac - last[0] >= 0.02:
                    last[0] = frac
                    _progress(mid, "transcribe", 0.03 + 0.52 * frac)

            try:
                with _asr_lock:
                    result = transcriber.transcribe(path, on_progress)
            except TranscriptionUnavailable as exc:
                raise AppError(str(exc), code="transcription_unavailable") from exc
            segments = [s for s in result.segments if _clean_text(s.text)]
            if not segments:
                raise AppError("No speech was detected in the recording.", code="no_speech")
            with session_scope() as db:
                mt = db.get(m.Meeting, mid)
                _replace_transcript(db, mt, segments)
                mt.language = result.language or mt.language
                if result.duration and not mt.duration:
                    mt.duration = result.duration
            ev["detail"] = f"{len(segments)} segments via {getattr(transcriber, 'name', 'asr')}"
        return {}

    def diarize(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        s = get_settings()
        if not (s.diarization_enabled and state.get("needs_transcription")):
            return {}
        _progress(mid, "diarize", 0.58)
        with _stage(mid, "diarize") as ev:
            try:
                with session_scope() as db:
                    path = db.get(m.Meeting, mid).storage_path
                with _asr_lock:
                    turns = get_diarizer().diarize(path)
                with session_scope() as db:
                    mt = db.get(m.Meeting, mid)
                    segs = db.scalars(select(m.TranscriptSegment).where(m.TranscriptSegment.meeting_id == mid)
                                      .order_by(m.TranscriptSegment.idx)).all()
                    raws = [RawSegment(s_.start_time, s_.end_time, s_.text) for s_ in segs]
                    assign_speakers(raws, turns)
                    _apply_speakers(db, mt, segs, [r.speaker for r in raws])
                    ev["detail"] = f"{len({r.speaker for r in raws if r.speaker})} speaker labels"
            except (DiarizationUnavailable, ImportError) as exc:
                ev["status"], ev["detail"] = "skipped", str(exc)[:200]
                logger.warning("Diarization skipped: %s", exc)
            except Exception as exc:  # noqa: BLE001 - diarization is optional; never fail the meeting for it
                ev["status"], ev["detail"] = "failed", type(exc).__name__
                logger.warning("Diarization failed (%s); continuing without speaker labels", type(exc).__name__)
        return {}

    def chunk(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        _progress(mid, "chunk", 0.62)
        with _stage(mid, "chunk") as ev:
            with session_scope() as db:
                views = repo.segment_views(db, mid)
            chunks = chunk_transcript(views)
            ev["detail"] = f"{len(views)} segments → {len(chunks)} chunks"
        return {"segments": views, "chunk_count": len(chunks)}

    def extract(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        attempts = state.get("attempts", 0) + 1
        _progress(mid, "extract", 0.65)
        with _stage(mid, "extract", llm) as ev:
            try:
                with session_scope() as db:
                    title = db.get(m.Meeting, mid).title
                extraction = run_extraction(
                    llm, mid, title, state["segments"],
                    progress=lambda f: _progress(mid, "extract", 0.65 + 0.2 * f))
                ev["detail"] = f"attempt {attempts}"
                return {"extraction": extraction, "attempts": attempts}
            except AppError as exc:
                ev["status"], ev["detail"] = "failed", exc.code
                return {"extraction": None, "attempts": attempts, "error": f"AI analysis failed: {exc.message}"}

    def validate_schema(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        _progress(mid, "validate", 0.86)
        with _stage(mid, "validate") as ev:
            valid = {s.ref for s in state["segments"]}
            extraction, issues, ratio = validate_extraction(state["extraction"], valid)
            ev["detail"] = "; ".join(issues) if issues else "schema ok"
        return {"extraction": extraction, "issues": issues, "needs_review": bool(issues) or state.get("needs_review", False),
                "verdicts": {"_ratio": ratio}}

    def verify(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        _progress(mid, "verify", 0.9)
        with _stage(mid, "verify", llm) as ev:
            verdicts = verify_extraction(state["extraction"], state["segments"], llm)
            flagged = sum(1 for v in verdicts.values() if v.needs_review)
            ev["detail"] = f"{len(verdicts)} claims checked, {flagged} flagged"
        return {"verdicts": verdicts}

    def persist(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        _progress(mid, "persist", 0.94)
        with _stage(mid, "persist") as ev:
            with session_scope() as db:
                mt = db.get(m.Meeting, mid)
                _persist_extraction(db, mt, state["extraction"], state["verdicts"], llm)
                repo.refresh_review_flag(db, mt)
                mt.needs_review = mt.needs_review or bool(state.get("needs_review"))
            ev["detail"] = "notes saved"
        return {}

    def index(state: PipelineState) -> PipelineState:
        mid = state["meeting_id"]
        _progress(mid, "index", 0.97)
        error = state.get("error")
        with _stage(mid, "index") as ev:
            with session_scope() as db:
                mt = db.get(m.Meeting, mid)
                n = index_meeting(db, mt)
                ev["detail"] = f"{n} chunks indexed"
                mt.stage, mt.progress = "done" if not error else "index", 1.0
                if error:
                    mt.status, mt.error_message = m.MeetingStatus.FAILED, error
                else:
                    mt.status, mt.error_message = m.MeetingStatus.COMPLETED, None
                    mt.completed_at = datetime.now(timezone.utc)
        return {}

    # ---- routing
    def after_transcribe(state: PipelineState) -> str:
        return "diarize"

    def after_extract(state: PipelineState) -> str:
        return "index" if state.get("extraction") is None else "validate_schema"

    def after_validate(state: PipelineState) -> str:
        ratio = (state.get("verdicts") or {}).get("_ratio", 1.0)
        if ratio < 0.5 and state.get("attempts", 0) < MAX_EXTRACTION_ATTEMPTS:
            logger.info("Low evidence coverage (%.2f); retrying extraction", ratio)
            return "extract"
        return "verify_evidence"

    g = StateGraph(PipelineState)
    for name, fn in (("ingest", ingest), ("transcribe", transcribe), ("diarize", diarize), ("chunk", chunk),
                     ("extract", extract), ("validate_schema", validate_schema), ("verify_evidence", verify),
                     ("persist", persist), ("index", index)):
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "transcribe")
    g.add_conditional_edges("transcribe", after_transcribe, {"diarize": "diarize"})
    g.add_edge("diarize", "chunk")
    g.add_edge("chunk", "extract")
    g.add_conditional_edges("extract", after_extract, {"index": "index", "validate_schema": "validate_schema"})
    g.add_conditional_edges("validate_schema", after_validate, {"extract": "extract", "verify_evidence": "verify_evidence"})
    g.add_edge("verify_evidence", "persist")
    g.add_edge("persist", "index")
    g.add_edge("index", END)
    return g.compile()


# ---------------------------------------------------------- persistence
def _replace_transcript(db, meeting: m.Meeting, segments: List[RawSegment]) -> None:
    db.execute(delete(m.TranscriptSegment).where(m.TranscriptSegment.meeting_id == meeting.id))
    db.execute(delete(m.Speaker).where(m.Speaker.meeting_id == meeting.id))
    db.flush()
    speakers: Dict[str, m.Speaker] = {}
    rows = []
    for i, seg in enumerate(segments, 1):
        label = seg.speaker
        sp = None
        if label:
            sp = speakers.get(label)
            if sp is None:
                sp = m.Speaker(meeting_id=meeting.id, speaker_label=label)
                db.add(sp)
                db.flush()
                speakers[label] = sp
        rows.append(m.TranscriptSegment(meeting_id=meeting.id, idx=i, speaker_id=sp.id if sp else None,
                                        start_time=round(seg.start, 2), end_time=round(max(seg.end, seg.start), 2),
                                        text=_clean_text(seg.text), confidence=seg.confidence))
    db.add_all(rows)
    db.flush()


def _apply_speakers(db, meeting: m.Meeting, segs: List[m.TranscriptSegment], labels: List[Optional[str]]) -> None:
    existing = {sp.speaker_label: sp for sp in db.scalars(select(m.Speaker).where(m.Speaker.meeting_id == meeting.id))}
    for seg, label in zip(segs, labels):
        if not label:
            seg.speaker_id = None
            continue
        sp = existing.get(label)
        if sp is None:
            sp = m.Speaker(meeting_id=meeting.id, speaker_label=label)
            db.add(sp)
            db.flush()
            existing[label] = sp
        seg.speaker_id = sp.id


def _persist_extraction(db, meeting: m.Meeting, ex: MeetingExtraction, verdicts: Dict[str, Any], llm: LLMService) -> None:
    from app.prompts import PROMPT_VERSION

    for model in (m.Decision, m.ActionItem, m.Risk):
        db.execute(delete(model).where(model.meeting_id == meeting.id))
    last = repo.latest_summary(db, meeting.id)
    db.add(m.MeetingSummary(
        meeting_id=meeting.id, version=(last.version + 1) if last else 1, overview=ex.summary.overview,
        purpose=ex.summary.purpose, main_topics=ex.summary.main_topics, key_points=ex.summary.key_points,
        outcome=ex.summary.outcome, generated_by=f"{llm.provider}:{llm.model}", prompt_version=PROMPT_VERSION))

    def vfields(key: str) -> dict:
        v = verdicts.get(key)
        if v is None:
            return {}
        return {"verification_status": v.status, "verification_score": v.score,
                "verification_note": v.note or None, "needs_review": v.needs_review}

    for i, d in enumerate(ex.decisions):
        db.add(m.Decision(meeting_id=meeting.id, description=d.text, evidence_segment_ids=d.evidence_segment_ids,
                          **vfields(f"decision:{i}")))
    for i, a in enumerate(ex.action_items):
        db.add(m.ActionItem(meeting_id=meeting.id, task=a.task, owner=a.owner, due_date=a.due_date,
                            evidence_segment_ids=a.evidence_segment_ids, **vfields(f"action:{i}")))
    for i, r in enumerate(ex.risks):
        db.add(m.Risk(meeting_id=meeting.id, kind=r.kind, description=r.text, severity=r.severity,
                      evidence_segment_ids=r.evidence_segment_ids, **vfields(f"risk:{i}")))
    for i, q in enumerate(ex.open_questions):
        db.add(m.Risk(meeting_id=meeting.id, kind="open_question", description=q.text,
                      evidence_segment_ids=q.evidence_segment_ids, **vfields(f"question:{i}")))
    db.flush()


# ------------------------------------------------------------------ entry
def process_meeting(meeting_id: str, force_transcribe: bool = False) -> PipelineState:
    """Run the full workflow once. Raises on unrecoverable errors (the job runner marks the meeting failed)."""
    llm = get_llm()
    graph = build_graph(llm)
    return graph.invoke({"meeting_id": meeting_id, "force_transcribe": force_transcribe})
