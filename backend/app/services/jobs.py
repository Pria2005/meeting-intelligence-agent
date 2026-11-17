"""Background job runner (PRD §8, §13). Processing never blocks an HTTP request.

An in-process thread pool keeps the local MVP dependency-free. The public surface (``submit``, status in the DB)
is deliberately small so it can be swapped for Celery/RQ/ARQ workers when scaling out (PRD §16.1)."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

from sqlalchemy import select

from app.core.config import get_settings
from app.core.errors import AppError, ConflictError, ProviderError
from app.db import models as m
from app.db.database import session_scope
from app.services import pipeline

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_executor: Optional[ThreadPoolExecutor] = None
_futures: Dict[str, Future] = {}


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max(1, get_settings().job_workers), thread_name_prefix="job")
    return _executor


def is_active(meeting_id: str) -> bool:
    with _lock:
        fut = _futures.get(meeting_id)
        return bool(fut and not fut.done())


def submit(meeting_id: str, force_transcribe: bool = False) -> None:
    """Queue processing for a meeting. Raises ConflictError if a job is already running for it."""
    with _lock:
        fut = _futures.get(meeting_id)
        if fut and not fut.done():
            raise ConflictError("This meeting is already being processed.")
        with session_scope() as db:
            mt = db.get(m.Meeting, meeting_id)
            if mt is None:
                raise AppError("Meeting not found.", status_code=404, code="not_found")
            mt.status, mt.stage, mt.progress, mt.error_message = m.MeetingStatus.QUEUED, "queued", 0.0, None
        _futures[meeting_id] = _get_executor().submit(_run, meeting_id, force_transcribe)


def wait_for(meeting_id: str, timeout: float = 60.0) -> bool:
    """Block until the job for ``meeting_id`` finishes (used by tests and scripts)."""
    with _lock:
        fut = _futures.get(meeting_id)
    if fut is None:
        return True
    try:
        fut.result(timeout=timeout)
    except Exception:  # noqa: BLE001 - failures are recorded on the meeting
        pass
    return fut.done()


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (ProviderError, pipeline.TransientPipelineError, ConnectionError, TimeoutError))


def _run(meeting_id: str, force_transcribe: bool) -> None:
    s = get_settings()
    attempt = 0
    while True:
        try:
            logger.info("Job start meeting=%s attempt=%d", meeting_id, attempt + 1)
            t0 = time.perf_counter()
            final = pipeline.process_meeting(meeting_id, force_transcribe)
            logger.info("Job finished meeting=%s in %.1fs", meeting_id, time.perf_counter() - t0,
                        extra={"ctx": {"meeting_id": meeting_id, "failed": bool(final.get("error"))}})
            return
        except Exception as exc:  # noqa: BLE001
            retryable = _is_retryable(exc)
            if retryable and attempt < s.job_max_retries:
                attempt += 1
                delay = min(30, 2 ** attempt * 2)
                logger.warning("Job %s failed transiently (%s); retry %d/%d in %ds", meeting_id, type(exc).__name__,
                               attempt, s.job_max_retries, delay)
                with session_scope() as db:
                    mt = db.get(m.Meeting, meeting_id)
                    if mt:
                        mt.retry_count = (mt.retry_count or 0) + 1
                        mt.stage = "retrying"
                time.sleep(delay)
                continue
            _mark_failed(meeting_id, exc)
            return


def _mark_failed(meeting_id: str, exc: BaseException) -> None:
    message = exc.message if isinstance(exc, AppError) else "Processing failed unexpectedly. Check the server logs."
    if not isinstance(exc, AppError):
        logger.error("Job %s crashed: %s", meeting_id, type(exc).__name__, exc_info=exc)
    else:
        logger.warning("Job %s failed: %s (%s)", meeting_id, exc.code, message)
    try:
        with session_scope() as db:
            mt = db.get(m.Meeting, meeting_id)
            if mt:
                mt.status, mt.error_message = m.MeetingStatus.FAILED, message
                mt.stage = mt.stage or "failed"
                db.add(m.ProcessingEvent(meeting_id=meeting_id, stage="job", status="failed", duration_ms=0,
                                         detail=type(exc).__name__))
    except Exception:  # noqa: BLE001
        logger.exception("could not record failure state")


def recover_stale_jobs() -> int:
    """After a restart, jobs left queued/processing can never finish: mark them failed so users can retry."""
    with session_scope() as db:
        rows = db.scalars(select(m.Meeting).where(m.Meeting.status.in_(
            [m.MeetingStatus.QUEUED, m.MeetingStatus.PROCESSING]))).all()
        for mt in rows:
            mt.status = m.MeetingStatus.FAILED
            mt.error_message = "Processing was interrupted by a server restart. Press “Process again” to retry."
        return len(rows)


def shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    with _lock:
        _futures.clear()
