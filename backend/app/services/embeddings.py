"""Text embeddings (PRD §12.1): Sentence Transformers with a dependency-free offline fallback."""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from typing import List, Optional, Protocol, Sequence

from app.core.config import get_settings
from app.services.textutil import content_tokens, stem

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


class SentenceTransformerEmbedder:
    _model = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self._load()

    def _load(self):
        cls = SentenceTransformerEmbedder
        with cls._lock:
            if cls._model is None or getattr(cls._model, "_mi_name", None) != self.model_name:
                from sentence_transformers import SentenceTransformer  # type: ignore

                logger.info("Loading embedding model %s", self.model_name)
                model = SentenceTransformer(self.model_name)
                model._mi_name = self.model_name  # type: ignore[attr-defined]
                cls._model = model
        return cls._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._load().encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class HashingEmbedder:
    """Deterministic bag-of-stems hashing embedder (no model download, works offline).

    Captures lexical overlap only - good enough for demos/tests, weaker than a neural model."""

    DIM = 768

    def __init__(self) -> None:
        self.name = "hash:768"

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.DIM
        toks = [stem(t) for t in content_tokens(text)]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            idx = h % self.DIM
            sign = 1.0 if (h >> 40) & 1 else -1.0
            v[idx] += sign * (1.0 if "_" not in f else 0.6)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


_embedder: Optional[Embedder] = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            s = get_settings()
            if s.embedding_backend.lower().startswith("sentence"):
                try:
                    _embedder = SentenceTransformerEmbedder(s.embedding_model)
                except ImportError:
                    logger.warning("sentence-transformers is not installed; falling back to the hashing embedder. "
                                   "Install requirements-ml.txt for semantic search quality.")
                    _embedder = HashingEmbedder()
            else:
                _embedder = HashingEmbedder()
        return _embedder


def set_embedder(e: Optional[Embedder]) -> None:
    global _embedder
    with _lock:
        _embedder = e


def embedding_backend_name() -> str:
    return get_embedder().name
