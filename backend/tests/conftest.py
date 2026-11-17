"""Shared fixtures: an isolated data dir, offline LLM/embeddings, and a fake transcriber (no model downloads)."""
from __future__ import annotations

import io
import math
import os
import struct
import wave
from typing import Callable, Optional

import pytest

os.environ.setdefault("LLM_PROVIDER", "heuristic")
os.environ.setdefault("EMBEDDING_BACKEND", "hash")


def make_wav(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(2000 * math.sin(2 * math.pi * 220 * i / rate)))
                          for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return buf.getvalue()


class FakeTranscriber:
    name = "fake-asr"

    def __init__(self, lines=None):
        from app.services.demo import DEMO_MEETINGS, _timed_segments

        self._segments = _timed_segments(lines or DEMO_MEETINGS[0][1])

    def transcribe(self, audio_path: str, progress: Optional[Callable[[float], None]] = None):
        from app.services.transcription import RawSegment, TranscriptionResult

        segs = [RawSegment(s.start, s.end, s.text, s.confidence) for s in self._segments]  # no speaker labels
        if progress:
            progress(1.0)
        return TranscriptionResult(segments=segs, language="en", duration=segs[-1].end)


def _boot(tmp_path, monkeypatch, auth: bool):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AUTH_ENABLED", "true" if auth else "false")
    monkeypatch.setenv("LLM_PROVIDER", "heuristic")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("DIARIZATION_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("JOB_MAX_RETRIES", "0")
    from app.core.config import reset_settings_cache
    from app.db.database import dispose_engine
    from app.services import embeddings, jobs, transcription, vectorstore

    reset_settings_cache()
    dispose_engine()
    vectorstore.reset_vector_store()
    embeddings.set_embedder(None)
    transcription.set_transcriber(FakeTranscriber())
    jobs.shutdown()
    from fastapi.testclient import TestClient
    from app.main import create_app

    return TestClient(create_app())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    with _boot(tmp_path, monkeypatch, auth=False) as c:
        yield c
    from app.services import jobs, transcription

    jobs.shutdown()
    transcription.set_transcriber(None)


@pytest.fixture()
def auth_client(tmp_path, monkeypatch):
    with _boot(tmp_path, monkeypatch, auth=True) as c:
        yield c
    from app.services import jobs, transcription

    jobs.shutdown()
    transcription.set_transcriber(None)


@pytest.fixture()
def wav_bytes() -> bytes:
    return make_wav()


def upload_and_wait(client, wav: bytes, title: str = "Weekly sync", filename: str = "sync.wav") -> str:
    from app.services import jobs

    r = client.post("/meetings/upload", files={"file": (filename, wav, "audio/wav")}, data={"title": title})
    assert r.status_code == 201, r.text
    mid = r.json()["meeting_id"]
    assert jobs.wait_for(mid, timeout=120)
    return mid
