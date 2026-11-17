"""Audio upload validation and safe storage (PRD FR-01, §14)."""
from __future__ import annotations

import logging
import re
import unicodedata
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.errors import PayloadTooLargeError, UnsupportedMediaError, ValidationFailure

logger = logging.getLogger(__name__)

CONTENT_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
_CHUNK = 1024 * 1024


@dataclass
class StoredAudio:
    meeting_id: str
    path: Path
    display_name: str
    size: int
    content_type: str
    duration: Optional[float]


def sanitize_filename(name: Optional[str]) -> str:
    """Return a safe display name (never used as a filesystem path)."""
    name = Path(name or "recording").name  # strips any directory components
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return (name or "recording")[:120]


def _sniff_format(head: bytes) -> Optional[str]:
    """Identify the container by magic bytes, independent of the claimed extension."""
    if len(head) < 12:
        return None
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    if head[4:8] == b"ftyp":
        return ".m4a"
    return None


def probe_duration(path: Path) -> Optional[float]:
    """Best-effort duration in seconds. Returns None if it cannot be determined."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                frames, rate = w.getnframes(), w.getframerate()
                if rate:
                    return round(frames / float(rate), 2)
        except (wave.Error, EOFError):
            pass
    try:
        import av  # type: ignore  (installed with faster-whisper)

        with av.open(str(path)) as container:
            if container.duration:
                return round(container.duration / 1_000_000, 2)
            streams = [s for s in container.streams if s.type == "audio"]
            if streams and streams[0].duration and streams[0].time_base:
                return round(float(streams[0].duration * streams[0].time_base), 2)
    except Exception as exc:  # noqa: BLE001 - probing must never crash an upload
        logger.debug("duration probe failed: %s", type(exc).__name__)
    return None


def _decodable(path: Path) -> bool:
    """Confirm the file really is audio when PyAV is available."""
    try:
        import av  # type: ignore
    except ImportError:
        return True
    try:
        with av.open(str(path)) as container:
            return any(s.type == "audio" for s in container.streams)
    except Exception:  # noqa: BLE001
        return False


async def save_upload(upload: UploadFile, meeting_id: str) -> StoredAudio:
    """Validate and persist an upload. Raises AppError subclasses on invalid input."""
    s = get_settings()
    display_name = sanitize_filename(upload.filename)
    ext = Path(display_name).suffix.lower()
    if ext not in s.allowed_extensions:
        raise UnsupportedMediaError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(s.allowed_extensions)}.")

    dest_dir = Path(s.upload_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Stored name is generated server-side: no user-controlled path components.
    dest = dest_dir / f"{meeting_id}_{uuid.uuid4().hex[:8]}{ext}"

    size = 0
    head = b""
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await upload.read(_CHUNK)
                if not chunk:
                    break
                if len(head) < 16:
                    head += chunk[: 16 - len(head)]
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise PayloadTooLargeError(f"File exceeds the {s.max_upload_mb} MB upload limit.")
                out.write(chunk)

        if size == 0:
            raise ValidationFailure("The uploaded file is empty.")
        sniffed = _sniff_format(head)
        if sniffed is None:
            raise UnsupportedMediaError("The file content is not a recognised MP3, WAV or M4A recording.")
        if sniffed != ext:
            raise UnsupportedMediaError(
                f"File content looks like {sniffed} but the extension is {ext}. Please rename or re-export it.")
        if not _decodable(dest):
            raise ValidationFailure("The audio file is corrupt or cannot be decoded.")
        duration = probe_duration(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    return StoredAudio(meeting_id=meeting_id, path=dest, display_name=display_name, size=size,
                       content_type=CONTENT_TYPES.get(ext, "application/octet-stream"), duration=duration)


def delete_audio(storage_path: Optional[str]) -> None:
    if not storage_path:
        return
    try:
        p = Path(storage_path).resolve()
        root = Path(get_settings().upload_dir).resolve()
        if root in p.parents:  # path-safety: only ever delete inside the upload dir
            p.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not delete stored audio file")
