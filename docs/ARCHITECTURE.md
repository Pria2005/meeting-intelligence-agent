# Architecture

## Components
| Layer | Module | Notes |
|---|---|---|
| UI | `frontend/` (Streamlit) | Talks to the API only through `api_client.py`; custom CSS design system in `theme.py`; all dynamic HTML is escaped |
| API | `backend/app/api` | FastAPI routers, consistent error envelope, auth dependency |
| Orchestration | `services/pipeline.py` (LangGraph) | Stateful graph; each node records a `ProcessingEvent` |
| Jobs | `services/jobs.py` | Thread pool; status in DB; restart recovery; retry with backoff |
| AI providers | `services/llm.py`, `transcription.py`, `diarization.py`, `embeddings.py` | Protocol-based; replaceable; test doubles injected via `set_*` |
| Data | `db/models.py`, `services/vectorstore.py` | SQL = source of truth; Chroma = derived, rebuildable (`/reindex`) |

## Pipeline graph
```
START → ingest → transcribe → diarize → chunk → extract ─┬─(LLM failed)────────────────► index → END
                                                         └► validate_schema ─┬─(<50% grounded, attempts<2)─► extract
                                                                             └► verify_evidence → persist → index → END
```
* Re-running skips transcription when a transcript exists (`?force=true` to redo ASR).
* Long transcripts are windowed (`LLM_MAX_INPUT_CHARS`), extracted per window, deduplicated, and summaries merged (map-reduce).
* Evidence ids are `seg-0001`-style, scoped to a meeting; validation drops references to non-existent segments.

## Grounding & hallucination controls
1. Prompts forbid inventing owners/dates; schema converts "unknown/N/A" to `null`.
2. Verification removes owners/due dates not present in the cited lines.
3. Each claim gets `verified | weak | unsupported | conflicting` and `needs_review`; humans clear the flag by editing/saving.
4. RAG answers only from retrieved chunks; below-threshold retrieval or `found=false` → explicit "not found".

## Security model
* `AUTH_ENABLED=false`: one implicit local user. `true`: JWT bearer; all queries filter by `user_id`; foreign IDs → 404.
* Uploads: extension allow-list, magic-byte sniffing, decode check, size cap while streaming, generated filenames, stored under `DATA_DIR/uploads` (never served statically); delete only within the upload dir.
* Logs never include transcript text; API never returns filesystem paths.

## Scaling path
SQLite → PostgreSQL (`DATABASE_URL`), thread pool → Celery/RQ workers, local disk → object storage in `services/storage.py`, Chroma persistent client → Chroma server / pgvector behind `VectorStore`.
