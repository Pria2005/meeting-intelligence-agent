"""Optional speaker diarization with pyannote.audio (PRD FR-03).

Uncertain turns stay generic ("Unknown") instead of guessing (PRD §11.1 step 4).
Mapping labels to real names is always a manual user action."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Protocol

from app.core.config import get_settings
from app.services.transcription import RawSegment

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"
MIN_OVERLAP_RATIO = 0.4


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


class Diarizer(Protocol):
    def diarize(self, audio_path: str) -> List[Turn]: ...


class DiarizationUnavailable(RuntimeError):
    pass


class PyannoteDiarizer:
    _pipeline = None
    _lock = threading.Lock()

    @classmethod
    def _load(cls):
        s = get_settings()
        if not s.hf_token:
            raise DiarizationUnavailable("HF_TOKEN is required to download the pyannote diarization model.")
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise DiarizationUnavailable("pyannote.audio is not installed (see requirements-optional.txt).") from exc
        with cls._lock:
            if cls._pipeline is None:
                try:
                    cls._pipeline = Pipeline.from_pretrained(s.diarization_model, token=s.hf_token)
                except TypeError:  # older pyannote releases
                    cls._pipeline = Pipeline.from_pretrained(s.diarization_model, use_auth_token=s.hf_token)
                if cls._pipeline is None:
                    raise DiarizationUnavailable(
                        "Could not load the diarization model. Accept the model terms on Hugging Face.")
                try:
                    import torch  # type: ignore

                    if torch.cuda.is_available():
                        cls._pipeline.to(torch.device("cuda"))
                except Exception:  # noqa: BLE001
                    pass
        return cls._pipeline

    def diarize(self, audio_path: str) -> List[Turn]:
        pipeline = self._load()
        audio = _decode_waveform(audio_path)
        result = pipeline(audio)
        annotation = getattr(result, "speaker_diarization", result)  # pyannote 4 returns an output object
        turns = [Turn(float(t.start), float(t.end), str(label))
                 for t, _, label in annotation.itertracks(yield_label=True)]
        return sorted(turns, key=lambda t: t.start)


def _decode_waveform(path: str) -> dict:
    """Decode with PyAV so M4A/MP3 work without torchaudio backends or the ffmpeg CLI."""
    import av  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore

    target_sr = 16000
    chunks = []
    with av.open(path) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
    data = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
    return {"waveform": torch.from_numpy(data.astype("float32")).unsqueeze(0), "sample_rate": target_sr}


def assign_speakers(segments: List[RawSegment], turns: List[Turn]) -> List[RawSegment]:
    """Assign each transcript segment the speaker with the largest time overlap.

    Speakers are renamed to stable labels in order of first appearance (Speaker 1, 2, ...).
    Segments without enough overlap evidence are labelled Unknown."""
    order: dict = {}
    for seg in segments:
        dur = max(seg.end - seg.start, 1e-6)
        overlap_by_speaker: dict = {}
        for t in turns:
            if t.end <= seg.start:
                continue
            if t.start >= seg.end:
                break
            ov = min(seg.end, t.end) - max(seg.start, t.start)
            if ov > 0:
                overlap_by_speaker[t.speaker] = overlap_by_speaker.get(t.speaker, 0.0) + ov
        if not overlap_by_speaker:
            seg.speaker = UNKNOWN
            continue
        best, best_ov = max(overlap_by_speaker.items(), key=lambda kv: kv[1])
        total_ov = sum(overlap_by_speaker.values())
        ambiguous = len(overlap_by_speaker) > 1 and best_ov / total_ov < 0.6  # overlapping speech
        if best_ov / dur < MIN_OVERLAP_RATIO or ambiguous:
            seg.speaker = UNKNOWN
            continue
        if best not in order:
            order[best] = f"Speaker {len(order) + 1}"
        seg.speaker = order[best]
    return segments


_diarizer_override: Optional[Diarizer] = None


def set_diarizer(d: Optional[Diarizer]) -> None:
    global _diarizer_override
    _diarizer_override = d


def get_diarizer() -> Diarizer:
    return _diarizer_override or PyannoteDiarizer()
