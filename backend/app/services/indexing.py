"""Transcript chunking → relational DocumentChunk rows + ChromaDB vectors (idempotent, PRD §12.1)."""
from __future__ import annotations

import logging

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db import models as m
from app.services import repository as repo
from app.services.chunking import chunk_transcript
from app.services.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


def index_meeting(db: Session, meeting: m.Meeting) -> int:
    """Rebuild chunks for ``meeting`` from the stored transcript. Safe to call repeatedly (retries, edits)."""
    views = repo.segment_views(db, meeting.id)
    chunks = chunk_transcript(views)
    db.execute(delete(m.DocumentChunk).where(m.DocumentChunk.meeting_id == meeting.id))
    rows, vec_chunks = [], []
    for c in chunks:
        cid = c.chunk_id(meeting.id)
        meta = {"segment_refs": ",".join(c.segment_refs), "speakers": ", ".join(c.speakers),
                "start_time": round(c.start, 2), "end_time": round(c.end, 2), "title": meeting.title,
                "chunk_index": c.index}
        rows.append(m.DocumentChunk(id=cid, meeting_id=meeting.id, chunk_index=c.index, chunk_text=c.text,
                                    start_time=c.start, end_time=c.end, chunk_metadata=meta))
        vec_chunks.append({"id": cid, "text": c.text, "metadata": meta})
    db.add_all(rows)
    db.flush()
    n = get_vector_store().index_meeting(meeting.id, meeting.user_id, vec_chunks)
    logger.info("Indexed %d chunks for %s", n, meeting.id)
    return n
