"""Application configuration.

All settings come from environment variables (or a local ``.env`` file) so that
secrets are never hard-coded (PRD §14, §19).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "AI Meeting Intelligence Agent"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"])

    # ------------------------------------------------------------- storage
    data_dir: Path = PROJECT_ROOT / "data"
    database_url: Optional[str] = None  # defaults to SQLite inside data_dir
    upload_dir: Optional[Path] = None  # defaults to data_dir/uploads (outside any web root)
    chroma_dir: Optional[Path] = None  # defaults to data_dir/chroma
    max_upload_mb: int = 200
    allowed_extensions: List[str] = Field(default_factory=lambda: [".mp3", ".wav", ".m4a"])

    # ---------------------------------------------------------------- auth
    # PRD Appendix B: local single-user by default; flip AUTH_ENABLED for multi-user.
    auth_enabled: bool = False
    jwt_secret: str = "change-me-in-production-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 12
    allow_registration: bool = True

    # --------------------------------------------------------- transcription
    whisper_model_size: str = "base"  # tiny | base | small | medium | large-v3 ...
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute_type: str = "auto"  # auto -> int8 on CPU, float16 on CUDA
    whisper_language: Optional[str] = None  # None = auto-detect
    whisper_beam_size: int = 5

    # ----------------------------------------------------------- diarization
    diarization_enabled: bool = False
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: Optional[str] = None

    # ------------------------------------------------------------------ LLM
    # provider: "groq" | "openai" | "openai_compatible" | "heuristic"
    # "heuristic" is an offline rule-based fallback that needs no API key.
    llm_provider: str = "auto"  # auto -> groq if key present, else heuristic
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.1
    llm_timeout_s: int = 120
    llm_max_retries: int = 3
    llm_max_input_chars: int = 28000  # transcript window size for map-reduce extraction
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_api_key: Optional[str] = None  # for generic OpenAI-compatible endpoints
    # Privacy gate (PRD §14): external providers only receive content if enabled.
    allow_external_llm: bool = True

    # ------------------------------------------------------------- embeddings
    # backend: "sentence-transformers" | "hash" (offline, dependency-free fallback)
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ------------------------------------------------------------------ RAG
    chunk_chars: int = 900
    chunk_overlap_chars: int = 150
    rag_top_k: int = 6
    rag_min_score: Optional[float] = None  # derived from embedding backend if unset
    verify_min_score: float = 0.28

    # ---------------------------------------------------------------- jobs
    job_workers: int = 2
    job_max_retries: int = 2

    # -------------------------------------------------------------- frontend
    api_url: str = "http://127.0.0.1:8000"

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"
        if self.database_url is None:
            self.database_url = f"sqlite:///{(self.data_dir / 'meetings.db').as_posix()}"
        if self.rag_min_score is None:
            self.rag_min_score = 0.30 if self.embedding_backend.startswith("sentence") else 0.12
        self.allowed_extensions = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in self.allowed_extensions]
        return self

    # ------------------------------------------------------------- helpers
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_llm_provider(self) -> str:
        provider = (self.llm_provider or "auto").lower()
        if provider == "auto":
            if self.groq_api_key:
                return "groq"
            if self.openai_api_key:
                return "openai"
            if self.llm_api_key and self.llm_base_url:
                return "openai_compatible"
            return "heuristic"
        return provider

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.chroma_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    get_settings.cache_clear()
