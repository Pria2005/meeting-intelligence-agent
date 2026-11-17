"""Timestamp-preserving transcript chunking (PRD §11.1 step 5, §12.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.dtypes import SegmentView


@dataclass
class Chunk:
    index: int
    text: str
    start: float
    end: float
    segment_refs: List[str] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)

    def chunk_id(self, meeting_id: str) -> str:
        return f"{meeting_id}:c{self.index:04d}"

    def to_document(self, meeting_id: str, title: str) -> Document:
        return Document(page_content=self.text, metadata={
            "meeting_id": meeting_id, "title": title, "chunk_id": self.chunk_id(meeting_id),
            "start_time": self.start, "end_time": self.end})


def _line(seg: SegmentView) -> str:
    return f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text


def _split_long(seg: SegmentView, limit: int) -> List[SegmentView]:
    """A single very long segment is split on sentence/word boundaries; pieces keep the segment's timing."""
    if len(_line(seg)) <= limit:
        return [seg]
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=limit, chunk_overlap=0,
                                              separators=["\n", ". ", "? ", "! ", "; ", ", ", " "])
    parts = splitter.split_text(seg.text)
    dur = max(seg.end - seg.start, 0.0)
    n = max(len(parts), 1)
    out = []
    for i, p in enumerate(parts):
        out.append(SegmentView(ref=seg.ref, start=seg.start + dur * i / n, end=seg.start + dur * (i + 1) / n,
                               speaker=seg.speaker, text=p))
    return out


def chunk_transcript(segments: Sequence[SegmentView], max_chars: int | None = None,
                     overlap_chars: int | None = None) -> List[Chunk]:
    """Greedy packing of whole segments up to ``max_chars``; the tail of each chunk (≈ overlap_chars)
    is repeated at the head of the next one so answers that straddle a boundary stay retrievable."""
    s = get_settings()
    max_chars = max_chars or s.chunk_chars
    overlap_chars = s.chunk_overlap_chars if overlap_chars is None else overlap_chars
    pieces: List[SegmentView] = []
    for seg in segments:
        if seg.text.strip():
            pieces.extend(_split_long(seg, max_chars))

    chunks: List[Chunk] = []
    current: List[SegmentView] = []
    size = 0

    def flush() -> None:
        if not current:
            return
        refs: List[str] = []
        for p in current:
            if p.ref not in refs:
                refs.append(p.ref)
        speakers: List[str] = []
        for p in current:
            if p.speaker and p.speaker not in speakers:
                speakers.append(p.speaker)
        chunks.append(Chunk(index=len(chunks), text="\n".join(_line(p) for p in current),
                            start=current[0].start, end=current[-1].end, segment_refs=refs, speakers=speakers))

    for piece in pieces:
        length = len(_line(piece)) + 1
        if current and size + length > max_chars:
            flush()
            # build overlap from the tail of the previous chunk
            tail: List[SegmentView] = []
            acc = 0
            for p in reversed(current):
                if acc + len(_line(p)) > overlap_chars and tail:
                    break
                tail.insert(0, p)
                acc += len(_line(p)) + 1
                if acc >= overlap_chars:
                    break
            current = tail if overlap_chars > 0 else []
            size = sum(len(_line(p)) + 1 for p in current)
        current.append(piece)
        size += length
    flush()  # the last piece always postdates the previous flush, so this chunk is never redundant
    return chunks
