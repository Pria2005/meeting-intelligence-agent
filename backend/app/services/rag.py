"""Evidence-grounded question answering over one or many authorised meetings (PRD FR-08, FR-09, §12.2)."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import NotFoundError
from app.db import models as m
from app.schemas import api as schemas
from app.services.dtypes import ContextChunk
from app.services.indexing import index_meeting
from app.services.llm import get_llm
from app.services.textutil import coverage, truncate
from app.services.vectorstore import Hit, get_vector_store

logger = logging.getLogger(__name__)

NOT_FOUND_TEXT = ("I couldn't find that in the selected meetings. The transcripts don't contain enough evidence "
                  "to answer reliably.")


def authorised_meeting_ids(db: Session, user: m.User, requested: Optional[List[str]]) -> List[str]:
    """Scope = indexed meetings the user owns (those with chunks), optionally narrowed to ``requested``."""
    own = {row[0] for row in db.execute(
        select(m.Meeting.id).where(m.Meeting.user_id == user.id,
                                   m.Meeting.id.in_(select(m.DocumentChunk.meeting_id).distinct())))}
    if requested:
        owned_any = {row[0] for row in db.execute(select(m.Meeting.id).where(m.Meeting.user_id == user.id,
                                                                            m.Meeting.id.in_(requested)))}
        if set(requested) - owned_any:
            raise NotFoundError("One or more requested meetings were not found.")
        return [mid for mid in requested if mid in own]
    return sorted(own)


def _ensure_indexed(db: Session, scope: List[str]) -> None:
    """After an embedding-model change a meeting has no vectors in the active collection: rebuild lazily."""
    store = get_vector_store()
    for mid in scope:
        if store.count(mid) == 0:
            meeting = db.get(m.Meeting, mid)
            if meeting is not None:
                logger.info("Re-indexing %s for the active embedding model", mid)
                index_meeting(db, meeting)
                db.commit()


def _rank(question: str, hits: List[Hit], top_k: int, min_score: float) -> List[Hit]:
    """Hybrid ranking: semantic similarity blended with keyword coverage; drops weak matches."""
    scored = []
    for h in hits:
        lex = coverage(question, h.text)
        blended = 0.75 * h.score + 0.25 * lex
        if h.score >= min_score or lex >= 0.6:
            scored.append((blended, h))
    scored.sort(key=lambda t: -t[0])
    out = []
    for blended, h in scored[:top_k]:
        h.score = round(blended, 4)
        out.append(h)
    return out


def answer_question(db: Session, user: m.User, req: schemas.ChatRequest) -> schemas.ChatResponse:
    s = get_settings()
    started = time.perf_counter()
    scope = authorised_meeting_ids(db, user, req.meeting_ids)
    top_k = req.top_k or s.rag_top_k
    if not scope:
        return schemas.ChatResponse(answer="There are no processed meetings to search yet.", found=False,
                                    answer_type="not_found", scope=[],
                                    latency_ms=int((time.perf_counter() - started) * 1000))

    _ensure_indexed(db, scope)
    hits = get_vector_store().query(req.question, scope, top_k=max(top_k * 3, 12))
    ranked = _rank(req.question, hits, top_k, s.rag_min_score or 0.0)
    if not ranked:
        return schemas.ChatResponse(answer=NOT_FOUND_TEXT, found=False, answer_type="not_found", scope=scope,
                                    latency_ms=int((time.perf_counter() - started) * 1000))

    titles: Dict[str, str] = {row[0]: row[1] for row in db.execute(
        select(m.Meeting.id, m.Meeting.title).where(m.Meeting.id.in_({h.meeting_id for h in ranked})))}
    contexts: List[ContextChunk] = []
    by_label: Dict[str, Hit] = {}
    for i, h in enumerate(ranked, 1):
        label = f"C{i}"
        speakers = [x.strip() for x in str(h.metadata.get("speakers", "")).split(",") if x.strip()]
        contexts.append(ContextChunk(label=label, meeting_id=h.meeting_id,
                                     meeting_title=titles.get(h.meeting_id, "Untitled meeting"),
                                     start=float(h.metadata.get("start_time", 0)),
                                     end=float(h.metadata.get("end_time", 0)), text=h.text, speakers=speakers))
        by_label[label] = h

    llm = get_llm()
    history = [(msg.role, msg.content) for msg in req.history]
    result = llm.answer(req.question, contexts, history)

    if not result.found or not result.answer.strip():
        return schemas.ChatResponse(answer=NOT_FOUND_TEXT, found=False, answer_type="not_found", scope=scope,
                                    model=f"{llm.provider}:{llm.model}",
                                    latency_ms=int((time.perf_counter() - started) * 1000))

    cited_labels = [l for l in (x.strip().upper() for x in result.source_ids) if l in by_label] or [contexts[0].label]
    ctx_by_label = {c.label: c for c in contexts}
    citations: List[schemas.Citation] = []
    for label in dict.fromkeys(cited_labels):
        h, c = by_label[label], ctx_by_label[label]
        citations.append(schemas.Citation(
            label=label, meeting_id=c.meeting_id, meeting_title=c.meeting_title,
            segment_refs=[r for r in str(h.metadata.get("segment_refs", "")).split(",") if r],
            start_time=c.start, end_time=c.end, speakers=c.speakers, snippet=truncate(c.text, 420),
            score=h.score))
    meetings_cited = {c.meeting_id for c in citations}
    answer_type = result.answer_type
    if len(meetings_cited) > 1 and answer_type == "direct" and len(scope) > 1:
        answer_type = "synthesized" if "synth" in result.answer.lower() else answer_type
    logger.info("RAG answered question in scope=%d meetings, citations=%d, provider=%s", len(scope), len(citations),
                llm.provider)
    return schemas.ChatResponse(answer=result.answer.strip(), found=True, answer_type=answer_type, citations=citations,
                                scope=scope, model=f"{llm.provider}:{llm.model}",
                                latency_ms=int((time.perf_counter() - started) * 1000))