"""Schemas used to validate LLM output (PRD §11.2, Appendix A). Versioned together with prompts."""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "v1"

_SEG_RE = re.compile(r"(\d+)")


def normalize_ref(value: object) -> Optional[str]:
    """Accept 'seg-4', 'seg-0004', 'S4', 4 ... and normalise to 'seg-0004'."""
    m = _SEG_RE.search(str(value))
    if not m:
        return None
    return f"seg-{int(m.group(1)):04d}"


def _clean_refs(values: object) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    out: List[str] = []
    for v in values:
        ref = normalize_ref(v)
        if ref and ref not in out:
            out.append(ref)
    return out


def _blank_to_none(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none", "unknown", "n/a", "na", "unassigned", "tbd", "not specified"}:
        return None
    return text


class _Evidenced(BaseModel):
    evidence_segment_ids: List[str] = Field(default_factory=list)

    @field_validator("evidence_segment_ids", mode="before")
    @classmethod
    def _refs(cls, v):  # noqa: N805
        return _clean_refs(v)


class DecisionOut(_Evidenced):
    text: str = Field(min_length=3)


class ActionItemOut(_Evidenced):
    task: str = Field(min_length=3)
    owner: Optional[str] = None
    due_date: Optional[str] = None

    @field_validator("owner", "due_date", mode="before")
    @classmethod
    def _unknown_is_none(cls, v):  # noqa: N805
        return _blank_to_none(v)


class RiskOut(_Evidenced):
    text: str = Field(min_length=3)
    kind: str = "risk"  # risk | blocker | open_question
    severity: Optional[str] = None

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v):  # noqa: N805
        v = str(v or "risk").lower().replace(" ", "_").replace("-", "_")
        return v if v in {"risk", "blocker", "open_question"} else "risk"

    @field_validator("severity", mode="before")
    @classmethod
    def _severity(cls, v):  # noqa: N805
        v = _blank_to_none(v)
        return v.lower() if v and v.lower() in {"low", "medium", "high"} else None


class SummaryOut(BaseModel):
    overview: str = Field(min_length=3)
    purpose: str = ""
    main_topics: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    outcome: str = ""

    @field_validator("main_topics", "key_points", mode="before")
    @classmethod
    def _list(cls, v):  # noqa: N805
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x).strip() for x in v if str(x).strip()]


class WindowExtraction(BaseModel):
    """What the LLM returns for one transcript window."""

    summary: SummaryOut
    decisions: List[DecisionOut] = Field(default_factory=list)
    action_items: List[ActionItemOut] = Field(default_factory=list)
    risks: List[RiskOut] = Field(default_factory=list)
    open_questions: List[RiskOut] = Field(default_factory=list)


class MeetingExtraction(BaseModel):
    """Merged result for the whole meeting."""

    meeting_id: str = ""
    summary: SummaryOut
    decisions: List[DecisionOut] = Field(default_factory=list)
    action_items: List[ActionItemOut] = Field(default_factory=list)
    risks: List[RiskOut] = Field(default_factory=list)
    open_questions: List[RiskOut] = Field(default_factory=list)


class AnswerOut(BaseModel):
    found: bool = True
    answer: str = ""
    answer_type: str = "direct"  # direct | synthesized
    source_ids: List[str] = Field(default_factory=list)  # chunk labels like C1, C2

    @field_validator("answer_type", mode="before")
    @classmethod
    def _type(cls, v):  # noqa: N805
        return "synthesized" if str(v).lower().startswith("synth") else "direct"


class ClaimJudgement(BaseModel):
    verdict: str = "supported"  # supported | unsupported | conflicting
    reason: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):  # noqa: N805
        v = str(v).lower()
        for key in ("conflict", "unsupport", "support"):
            if key in v:
                return {"conflict": "conflicting", "unsupport": "unsupported", "support": "supported"}[key]
        return "unsupported"
