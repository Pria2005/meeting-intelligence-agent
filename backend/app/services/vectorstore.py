"""ChromaDB wrapper (PRD §12). Indexing is idempotent: chunk ids are deterministic and a meeting
is always re-indexed by delete-then-upsert."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import get_settings
from app.services.embeddings import get_embedder

logger = logging.getLogger(__name__)
COLLECTION_PREFIX = "meeting_chunks"


def collection_name() -> str:
    """One collection per embedding model: vectors from different models/dimensions are never mixed,
    so switching EMBEDDING_BACKEND / EMBEDDING_MODEL is always safe (meetings are re-indexed lazily)."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", get_embedder().name).strip("_").lower()[:40]
    return f"{COLLECTION_PREFIX}_{slug}"


@dataclass
class Hit:
    chunk_id: str
    meeting_id: str
    text: str
    score: float  # cosine similarity 0..1
    metadata: Dict[str, Any]


class VectorStore:
    def __init__(self) -> None:
        import chromadb  # lazy

        s = get_settings()
        self._client = chromadb.PersistentClient(path=str(s.chroma_dir))
        self._lock = threading.Lock()
        self._collections: dict = {}

    @property
    def _collection(self):
        name = collection_name()
        col = self._collections.get(name)
        if col is None:
            col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
            self._collections[name] = col
        return col

    # ----------------------------------------------------------- write path
    def index_meeting(self, meeting_id: str, user_id: str, chunks: Sequence[Dict[str, Any]]) -> int:
        """chunks: dicts with id, text, metadata. Replaces any existing vectors for the meeting."""
        embedder = get_embedder()
        with self._lock:
            self._collection.delete(where={"meeting_id": meeting_id})
            if not chunks:
                return 0
            vectors = embedder.embed([c["text"] for c in chunks])
            metas = []
            for c in chunks:
                m = dict(c["metadata"])
                m.update({"meeting_id": meeting_id, "user_id": user_id, "chunk_id": c["id"],
                          "embedding_model": embedder.name})
                metas.append({k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))})
            self._collection.upsert(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                                    embeddings=vectors, metadatas=metas)
        return len(chunks)

    def delete_meeting(self, meeting_id: str) -> None:
        """Remove the meeting from every embedding collection (current and legacy)."""
        with self._lock:
            for col in self._client.list_collections():
                name = getattr(col, "name", col)
                if str(name).startswith(COLLECTION_PREFIX):
                    self._client.get_collection(name).delete(where={"meeting_id": meeting_id})

    def count(self, meeting_id: Optional[str] = None) -> int:
        with self._lock:
            if meeting_id is None:
                return self._collection.count()
            got = self._collection.get(where={"meeting_id": meeting_id}, include=[])
            return len(got["ids"])

    # ------------------------------------------------------------ read path
    def query(self, question: str, meeting_ids: Sequence[str], top_k: int) -> List[Hit]:
        """Semantic search limited to ``meeting_ids`` (authorisation is decided by the caller)."""
        if not meeting_ids:
            return []
        vec = get_embedder().embed([question])[0]
        where = {"meeting_id": meeting_ids[0]} if len(meeting_ids) == 1 else {"meeting_id": {"$in": list(meeting_ids)}}
        with self._lock:
            total = self._collection.count()
            if total == 0:
                return []
            res = self._collection.query(query_embeddings=[vec], n_results=min(top_k, total), where=where,
                                         include=["documents", "metadatas", "distances"])
        hits: List[Hit] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(chunk_id=cid, meeting_id=meta.get("meeting_id", ""), text=doc,
                            score=max(0.0, min(1.0, 1.0 - float(dist))), metadata=meta))
        return hits

    def ping(self) -> bool:
        try:
            with self._lock:
                self._collection.count()
            return True
        except Exception:  # noqa: BLE001
            return False


_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore()
        return _store


def reset_vector_store() -> None:
    global _store
    with _store_lock:
        _store = None