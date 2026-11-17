"""System status, configuration summary and architecture."""
from __future__ import annotations

import streamlit as st

import api_client as api
import theme
from common import guard

theme.hero("System", "Live status of the services behind the app and how your data flows through them.", "Operations")

health = guard(api.health, {}) or {}
info = guard(api.info, {}) or {}
stats = guard(api.stats, {}) or {}

c = st.columns(4)
ok = lambda v: "✅ OK" if v == "ok" else "⚠️ " + str(v)  # noqa: E731
c[0].markdown(theme.metric_card("🧩", "API", ok(health.get("status")), f'v{info.get("version", "?")}', "var(--emerald)"), unsafe_allow_html=True)
c[1].markdown(theme.metric_card("🗄️", "Database", ok(health.get("database")), "SQLite / PostgreSQL", "var(--cyan)"), unsafe_allow_html=True)
c[2].markdown(theme.metric_card("🧭", "Vector store", ok(health.get("vector_store")), "ChromaDB", "var(--violet)"), unsafe_allow_html=True)
asr = (health.get("details") or {}).get("transcription_available")
c[3].markdown(theme.metric_card("🎙️", "Speech-to-text", "✅ Ready" if asr else "⚠️ Missing",
                                f'whisper {info.get("whisper_model", "?")}', "var(--amber)"), unsafe_allow_html=True)
st.write("")

left, right = st.columns(2)
with left:
    st.markdown(f'''<div class="mi-card"><h4>Configuration</h4><p>
    <b>LLM:</b> {theme.esc(info.get("llm_provider"))} · {theme.esc(info.get("llm_model"))}<br/>
    <b>Embeddings:</b> {theme.esc(info.get("embedding_backend"))}<br/>
    <b>Diarization:</b> {"enabled" if info.get("diarization_enabled") else "disabled"}<br/>
    <b>Authentication:</b> {"multi-user" if info.get("auth_enabled") else "single local user"}<br/>
    <b>Upload limit:</b> {info.get("max_upload_mb", "?")} MB · {theme.esc(", ".join(info.get("allowed_extensions", [])))}<br/>
    <b>Environment:</b> {theme.esc(info.get("environment"))}</p></div>''', unsafe_allow_html=True)
    if info.get("llm_provider") == "heuristic":
        st.info("Running with the offline rule-based analyser. For richer summaries and answers, set `GROQ_API_KEY` "
                "(or another provider) in `.env` and restart the backend.")
    if not asr:
        st.warning("`faster-whisper` is not installed, so audio cannot be transcribed. Run "
                   "`pip install -r backend/requirements-ml.txt`. Sample meetings still work.")
with right:
    avg = stats.get("avg_processing_seconds")
    st.markdown(f'''<div class="mi-card"><h4>Your workspace</h4><p>
    <b>Meetings:</b> {stats.get("total_meetings", 0)} · <b>Recorded time:</b> {theme.fmt_duration(stats.get("total_duration_seconds"))}<br/>
    <b>Average time to ready:</b> {f"{avg:.0f}s" if avg else "—"}<br/>
    <b>Open risks:</b> {stats.get("open_risks", 0)} · <b>Items needing review:</b> {stats.get("needs_review", 0)}</p></div>''',
                unsafe_allow_html=True)
    st.markdown('<div class="mi-card"><h4>API reference</h4><p>Interactive docs are served by the backend at '
                f'<a href="{api.API_URL}/docs" target="_blank">{api.API_URL}/docs</a>.</p></div>', unsafe_allow_html=True)

st.markdown("### How a recording becomes knowledge")
steps = [("1", "Ingest", "Validate type, size and content; store safely"), ("2", "Transcribe", "Faster-Whisper, timestamped"),
         ("3", "Speakers", "Optional diarization; uncertain = Unknown"), ("4", "Analyse", "LLM summary, decisions, actions, risks"),
         ("5", "Verify", "Every claim checked against cited lines"), ("6", "Index", "Chunks → embeddings → ChromaDB")]
cols = st.columns(len(steps))
for col, (n, t, d) in zip(cols, steps):
    col.markdown(f'<div class="mi-card"><span class="mi-chip">{n}</span><h4 style="margin-top:.5rem">{t}</h4>'
                 f'<div class="mi-sub">{d}</div></div>', unsafe_allow_html=True)
