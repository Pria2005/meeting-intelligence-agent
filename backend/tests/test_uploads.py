import pytest

from tests.conftest import make_wav


def up(client, name, data, ctype="audio/wav"):
    return client.post("/meetings/upload", files={"file": (name, data, ctype)}, data={"auto_process": "false"})


def test_valid_upload_returns_unique_ids(client, wav_bytes):
    a, b = up(client, "a.wav", wav_bytes), up(client, "b.wav", wav_bytes)
    assert a.status_code == b.status_code == 201
    assert a.json()["meeting_id"] != b.json()["meeting_id"]
    assert a.json()["status"] == "uploaded"
    assert a.json()["meeting"]["duration"] == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("name,data,code", [
    ("notes.txt", b"hello world, not audio", 415),
    ("empty.wav", b"", 422),
    ("fake.wav", b"this is definitely not a wav file at all", 415),
    ("fake.mp3", make_wav(), 415),  # content/extension mismatch
    ("../../etc/passwd.wav", b"junk-junk-junk-junk", 415),
])
def test_invalid_uploads_rejected(client, name, data, code):
    r = up(client, name, data)
    assert r.status_code == code, r.text
    assert r.json()["error"]["message"]
    assert client.get("/meetings").json() == []  # nothing left behind


def test_corrupt_wav_rejected(client):
    r = up(client, "bad.wav", b"RIFF\x00\x00\x00\x00WAVEgarbagegarbagegarbage")
    assert r.status_code in (415, 422)


def test_oversize_rejected(client, wav_bytes, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    r = up(client, "big.wav", wav_bytes)
    assert r.status_code == 413


def test_filename_is_sanitised_and_path_hidden(client, wav_bytes):
    r = up(client, "../../weird name?.wav", wav_bytes).json()
    assert "/" not in r["meeting"]["filename"] and ".." not in r["meeting"]["filename"]
    assert "storage_path" not in r["meeting"]


def test_process_failure_does_not_crash(client, wav_bytes):
    from app.services import jobs, transcription

    class Boom:
        name = "boom"

        def transcribe(self, *a, **k):
            raise RuntimeError("secret internal detail /var/x")

    transcription.set_transcriber(Boom())
    r = client.post("/meetings/upload", files={"file": ("a.wav", wav_bytes, "audio/wav")})
    mid = r.json()["meeting_id"]
    jobs.wait_for(mid, 30)
    d = client.get(f"/meetings/{mid}").json()
    assert d["status"] == "failed"
    assert "secret" not in (d["error_message"] or "")
    assert client.get("/health").status_code == 200  # app still alive


def test_no_speech_fails_cleanly(client, wav_bytes):
    from app.services import jobs, transcription
    from app.services.transcription import TranscriptionResult

    class Silent:
        name = "silent"

        def transcribe(self, *a, **k):
            return TranscriptionResult([], "en", 2.0)

    transcription.set_transcriber(Silent())
    mid = client.post("/meetings/upload", files={"file": ("a.wav", wav_bytes, "audio/wav")}).json()["meeting_id"]
    jobs.wait_for(mid, 30)
    d = client.get(f"/meetings/{mid}").json()
    assert d["status"] == "failed" and "speech" in d["error_message"].lower()
