"""Upload a recording and watch it being processed."""
from __future__ import annotations

import mimetypes

import streamlit as st

import api_client as api
import theme
from common import go_to_meeting, guard, info_cached, uses_external_llm

theme.hero("Add a meeting", "Drop an MP3, WAV or M4A recording. We transcribe it, extract the key outcomes and index it "
                            "for questions — you can watch each step happen.", "New meeting")

info = info_cached()
max_mb = info.get("max_upload_mb", 200)
exts = [e.lstrip(".") for e in info.get("allowed_extensions", [".mp3", ".wav", ".m4a"])]

left, right = st.columns([1.7, 1])
with left:
    file = st.file_uploader("Audio recording", type=exts, accept_multiple_files=False,
                            help=f"Up to {max_mb} MB · {', '.join(e.upper() for e in exts)}")
    title = st.text_input("Meeting title (optional)", placeholder="e.g. Sprint planning — 14 Nov")
    auto = st.toggle("Start processing right away", value=True)
    consent = True
    if uses_external_llm(info):
        consent = st.checkbox(
            f"I understand the transcript text will be sent to the configured AI provider "
            f"({info.get('llm_provider')} · {info.get('llm_model')}) to generate notes and answers.",
            help="Meeting content is only sent to an external provider with your approval.")
    if file is not None:
        st.audio(file)
    go = st.button("🚀 Upload & analyse" if auto else "⬆️ Upload", type="primary", width="stretch",
                   disabled=file is None or not consent)
    if go and file is not None:
        with st.spinner("Uploading and validating…"):
            mime = file.type or mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            res = guard(lambda: api.upload(file.name, file.getvalue(), mime, title.strip(), auto))
        if res:
            st.session_state["last_upload"] = res["meeting_id"]
            st.toast("Uploaded ✔", icon="✅")

with right:
    st.markdown(
        '<div class="mi-card"><h4>💡 For best results</h4>'
        '<p>• Record close to the speakers and reduce background noise.<br/>'
        '• One meeting per file; mono or stereo both work.<br/>'
        f'• Max {max_mb} MB per file.<br/>'
        '• Speaker names can be corrected after processing.</p></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mi-card"><h4>🔒 Privacy</h4>'
        '<p>Recordings are stored on the server outside any public folder and can be deleted at any time from the '
        'meeting page — deletion removes the audio, transcript, notes and search index.</p></div>',
        unsafe_allow_html=True)
    st.markdown(f'<div class="mi-card"><h4>🧠 Engine</h4><p>Speech-to-text: <b>Faster-Whisper ({theme.esc(info.get("whisper_model", "base"))})</b><br/>'
                f'Notes &amp; Q&amp;A: <b>{theme.esc(info.get("llm_provider", "?"))} · {theme.esc(info.get("llm_model", "?"))}</b><br/>'
                f'Embeddings: <b>{theme.esc(info.get("embedding_backend", "?"))}</b></p></div>', unsafe_allow_html=True)
    if not file and st.button("✨ Try a sample meeting instead", width="stretch"):
        meetings = guard(api.create_demo)
        if meetings:
            st.session_state["last_upload"] = meetings[0]["id"]
            st.rerun()

# ------------------------------------------------------------ live progress
mid = st.session_state.get("last_upload")
if mid:
    st.divider()
    st.subheader("Processing status")

    current = guard(lambda: api.get_meeting(mid), None, show=False)
    running = bool(current and current["status"] in ("queued", "processing"))

    def render_status() -> None:
        mt = guard(lambda: api.get_meeting(mid), None)
        if not mt:
            return
        if running and mt["status"] not in ("queued", "processing"):
            st.rerun()  # finished: re-render once without polling
        st.markdown(f'<div class="mi-card"><div class="mi-spread"><div><h4>{theme.esc(mt["title"])}</h4>'
                    f'<div class="mi-sub">{theme.esc(mid)} · {theme.fmt_duration(mt.get("duration"))}</div></div>'
                    f'{theme.badge(mt["status"])}</div>'
                    f'{theme.progress_html(mt["progress"], mt["stage"]) if mt["status"] in ("queued", "processing") else ""}'
                    f'</div>', unsafe_allow_html=True)
        if mt["status"] == "completed":
            st.success("Done! Your transcript, notes and search index are ready.")
            if st.button("Open meeting →", type="primary", key="open_done"):
                go_to_meeting(mid)
        elif mt["status"] == "failed":
            st.error(mt.get("error_message") or "Processing failed.")
            if st.button("Try again", key="retry_upload"):
                guard(lambda: api.start_processing(mid))
                st.rerun()
        elif mt["status"] == "uploaded":
            if st.button("Start processing", type="primary", key="start_uploaded"):
                guard(lambda: api.start_processing(mid))
                st.rerun()

    st.fragment(run_every=2 if running else None)(render_status)()
