"""Evidence verification of extracted claims (PRD FR-10, §11.1 step 8).

Each decision / action item / risk is compared against the transcript lines it cites:
  * lexical + embedding similarity gives a score,
  * borderline claims are additionally judged by the LLM (supported / unsupported / conflicting),
  * owners and due dates that are not in the evidence are removed instead of guessed."""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.core.config import get_settings
from app.schemas.extraction import MeetingExtraction
from app.services.dtypes import SegmentView
from app.services.embeddings import get_embedder
from app.services.llm import LLMService
from app.services.textutil import coverage, tokenize

logger = logging.getLogger(__name__)

MAX_LLM_JUDGEMENTS = 30
VERIFIED_THRESHOLD = 0.5


@dataclass
class Verdict:
    status: str  # verified | weak | unsupported | conflicting
    score: float
    note: str = ""

    @property
    def needs_review(self) -> bool:
        return self.status != "verified"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _evidence(refs: Sequence[str], by_ref: Dict[str, SegmentView]) -> List[SegmentView]:
    return [by_ref[r] for r in refs if r in by_ref]


def _mentioned(value: str, evidence: Sequence[SegmentView], include_speakers: bool) -> bool:
    """Is ``value`` (an owner name or a due-date phrase) actually present in the cited lines?"""
    haystack = " ".join(s.text for s in evidence).lower()
    if include_speakers:
        haystack += " " + " ".join((s.speaker or "") for s in evidence).lower()
    toks = [t for t in tokenize(value) if len(t) > 1]
    if not toks:
        return False
    if value.lower() in haystack:
        return True
    hay_tokens = set(tokenize(haystack))
    return all(t in hay_tokens or t.rstrip("s") in hay_tokens for t in toks if not re.fullmatch(r"\d{4}", t))


def verify_extraction(extraction: MeetingExtraction, segments: Sequence[SegmentView], llm: LLMService
                      ) -> Dict[str, Verdict]:
    """Annotate verification results. Returns verdicts keyed by f"{kind}:{index}".

    Mutates the extraction in place only to remove unsupported owners/due dates."""
    s = get_settings()
    by_ref = {seg.ref: seg for seg in segments}
    embedder = get_embedder()
    verdicts: Dict[str, Verdict] = {}
    judgements = 0

    def check(key: str, kind: str, claim: str, refs: List[str]) -> Verdict:
        nonlocal judgements
        ev = _evidence(refs, by_ref)
        if not ev:
            return Verdict("unsupported", 0.0, "No valid transcript evidence is cited.")
        ev_text = " ".join(e.text for e in ev)
        lex = coverage(claim, ev_text)
        try:
            vecs = embedder.embed([claim, ev_text])
            sem = max(0.0, _cosine(vecs[0], vecs[1]))
        except Exception:  # noqa: BLE001 - verification must not break processing
            sem = 0.0
        score = round(max(lex, sem), 3)
        status = "verified" if score >= VERIFIED_THRESHOLD else ("weak" if score >= s.verify_min_score else "unsupported")
        note = ""
        # Second opinion for anything not clearly supported, and a polarity check for everything cheap enough.
        if (status != "verified" or llm.provider == "heuristic") and judgements < MAX_LLM_JUDGEMENTS:
            try:
                judgements += 1
                j = llm.judge_claim(kind, claim, ev)
                if j.verdict == "conflicting":
                    status, note = "conflicting", j.reason or "The cited lines appear to contradict the claim."
                elif j.verdict == "supported" and status in {"weak", "unsupported"}:
                    status, note = ("verified" if lex >= 0.35 or llm.provider != "heuristic" else "weak",
                                    j.reason or "Confirmed by a second check.")
                elif j.verdict == "unsupported" and status == "weak":
                    status, note = "unsupported", j.reason or "The cited lines do not state this."
                elif j.verdict == "unsupported" and status == "verified" and llm.provider != "heuristic":
                    status, note = "weak", j.reason
            except Exception as exc:  # noqa: BLE001
                logger.warning("claim judgement failed: %s", type(exc).__name__)
        if status == "unsupported" and not note:
            note = "The cited lines do not clearly state this claim."
        elif status == "weak" and not note:
            note = "Only partially supported by the cited lines."
        return Verdict(status, score, note)

    for i, d in enumerate(extraction.decisions):
        verdicts[f"decision:{i}"] = check(f"decision:{i}", "decision", d.text, d.evidence_segment_ids)

    for i, a in enumerate(extraction.action_items):
        v = check(f"action:{i}", "action item", a.task, a.evidence_segment_ids)
        ev = _evidence(a.evidence_segment_ids, by_ref)
        notes = [v.note] if v.note else []
        if a.owner and not _mentioned(a.owner, ev, include_speakers=True):
            notes.append(f"Owner '{a.owner}' removed: not stated in the cited lines.")
            a.owner = None
        if a.due_date and not _mentioned(a.due_date, ev, include_speakers=False):
            notes.append(f"Due date '{a.due_date}' removed: not stated in the cited lines.")
            a.due_date = None
        verdicts[f"action:{i}"] = Verdict(v.status, v.score, " ".join(notes))

    for i, r in enumerate(extraction.risks):
        verdicts[f"risk:{i}"] = check(f"risk:{i}", r.kind.replace("_", " "), r.text, r.evidence_segment_ids)
    for i, q in enumerate(extraction.open_questions):
        verdicts[f"question:{i}"] = check(f"question:{i}", "open question", q.text, q.evidence_segment_ids)
    return verdicts
