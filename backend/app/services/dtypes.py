"""Plain data containers shared between services (avoids circular imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SegmentView:
    ref: str
    start: float
    end: float
    speaker: Optional[str]
    text: str


@dataclass
class ContextChunk:
    label: str  # C1, C2 ...
    meeting_id: str
    meeting_title: str
    start: float
    end: float
    text: str
    speakers: List[str] = field(default_factory=list)


@dataclass
class Usage:
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.retries += other.retries
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
