"""Speech-to-text with Faster-Whisper (PRD FR-02)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None  # filled by diarization


@dataclass
class TranscriptionResult:
    segments: List[RawSegment]
    language: Optional[str]
    duration: Optional[float]


ProgressCb = Callable[[float], None]


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio_path: str, progress: Optional[ProgressCb] = None) -> TranscriptionResult: ...


class TranscriptionUnavailable(RuntimeError):
    """Raised when faster-whisper is not installed."""


class FasterWhisperTranscriber:
    """Lazy, process-wide cached Faster-Whisper model."""

    _model = None
    _model_key: Optional[tuple] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        s = get_settings()
        self.name = f"faster-whisper:{s.whisper_model_size}"

    @classmethod
    def _load(cls):
        s = get_settings()
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TranscriptionUnavailable(
                "faster-whisper is not installed. Run: pip install faster-whisper") from exc

        device = s.whisper_device
        compute = s.whisper_compute_type
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        key = (s.whisper_model_size, device, compute)
        with cls._lock:
            if cls._model is None or cls._model_key != key:
                logger.info("Loading Whisper model size=%s device=%s compute=%s", *key)
                try:
                    cls._model = WhisperModel(s.whisper_model_size, device=device, compute_type=compute)
                except Exception as exc:  # noqa: BLE001 - usually a first-run download problem
                    logger.error("Whisper model load failed: %s", type(exc).__name__)
                    raise TranscriptionUnavailable(
                        f"Could not load the Whisper '{s.whisper_model_size}' model. The first run downloads it from "
                        "Hugging Face, so check your internet connection (or pick another WHISPER_MODEL_SIZE).") from exc
                cls._model_key = key
        return cls._model

    def transcribe(self, audio_path: str, progress: Optional[ProgressCb] = None) -> TranscriptionResult:
        s = get_settings()
        model = self._load()
        seg_iter, info = model.transcribe(
            audio_path,
            language=s.whisper_language,
            beam_size=s.whisper_beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        total = float(getattr(info, "duration", 0) or 0)
        out: List[RawSegment] = []
        for seg in seg_iter:  # generator: decoding happens lazily here
            text = (seg.text or "").strip()
            if not text:
                continue
            conf = None
            avg_logprob = getattr(seg, "avg_logprob", None)
            if avg_logprob is not None:
                conf = round(max(0.0, min(1.0, 2.718281828 ** avg_logprob)), 3)
            out.append(RawSegment(start=float(seg.start), end=float(seg.end), text=text, confidence=conf))
            if progress and total:
                progress(min(1.0, float(seg.end) / total))
        return TranscriptionResult(segments=out, language=getattr(info, "language", None), duration=total or None)


def _cuda_available() -> bool:
    try:
        import ctranslate2  # type: ignore

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def transcription_available() -> bool:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


_transcriber_override: Optional[Transcriber] = None


def set_transcriber(t: Optional[Transcriber]) -> None:
    """Dependency-injection hook (tests, alternative ASR providers)."""
    global _transcriber_override
    _transcriber_override = t


def get_transcriber() -> Transcriber:
    return _transcriber_override or FasterWhisperTranscriber()
