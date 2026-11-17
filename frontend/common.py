"""Helpers shared by several pages: error display, live-refresh fragments and the chat component."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import streamlit as st

import api_client as api
import theme


def guard(fn: Callable[[], Any], default: Any = None, show: bool = True) -> Any:
    """Run an API call and turn failures into friendly messages instead of tracebacks."""
    try:
        return fn()
    except api.BackendDown as exc:
        if show:
            st.error(f"🔌 {exc.message}")
    except api.ApiError as exc:
        if show:
            st.error(f"⚠️ {exc.message}")
    return default


def go_to_meeting(meeting_id: str) -> None:
    st.session_state["selected_meeting"] = meeting_id
    st.switch_page("pages/meeting.py")


def any_active(meetings: List[Dict[str, Any]]) -> bool:
    return any(m.get("status") in ("queued", "processing") for m in meetings)


def info_cached() -> Dict[str, Any]:
    if "_info" not in st.session_state:
        st.session_state["_info"] = guard(api.info, {}, show=False) or {}
    return st.session_state["_info"]


def uses_external_llm(info: Optional[Dict[str, Any]] = None) -> bool:
    info = info or info_cached()
    return info.get("llm_provider") not in (None, "heuristic")


# ------------------------------------------------------------------- chat
SUGGESTIONS = ["What decisions were made?", "What action items are assigned and who owns them?",
               "What are the main risks or blockers?", "What questions are still open?"]


def render_assistant(msg: Dict[str, Any]) -> None:
    kind = msg.get("answer_type", "direct")
    if not msg.get("found", True):
        tag = theme.badge("unsupported", theme.VERIFY_META, "Not found in meetings")
    elif kind == "synthesized":
        tag = '<span class="mi-badge violet"><span class="mi-dot"></span>Synthesized across sources</span>'
    else:
        tag = theme.badge("verified", theme.VERIFY_META, "Direct statement")
    st.markdown(tag, unsafe_allow_html=True)
    st.write(msg["content"])
    cites = msg.get("citations") or []
    if cites:
        with st.expander(f"📎 {len(cites)} evidence source{'s' if len(cites) != 1 else ''}", expanded=False):
            for c in cites:
                refs = ", ".join(c.get("segment_refs", [])[:6])
                st.markdown(
                    f'<div class="mi-quote"><b>{theme.esc(c["meeting_title"])}</b> '
                    f'<span class="t">{theme.fmt_time(c["start_time"])}–{theme.fmt_time(c["end_time"])}</span>'
                    f'{theme.chip(c["label"])}<br/>{theme.esc(c["snippet"])}'
                    f'<div class="mi-note">{theme.esc(refs)} · relevance {c["score"]:.2f}</div></div>',
                    unsafe_allow_html=True)
    if msg.get("model"):
        st.caption(f"{msg['model']} · {msg.get('latency_ms', 0)} ms")


def chat_ui(scope_ids: Optional[List[str]], key: str, placeholder: str = "Ask anything about your meetings…") -> None:
    history: List[Dict[str, Any]] = st.session_state.setdefault(f"chat_{key}", [])
    for msg in history:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🎙️"):
            if msg["role"] == "assistant":
                render_assistant(msg)
            else:
                st.write(msg["content"])

    question = st.chat_input(placeholder, key=f"input_{key}")
    if not history:
        cols = st.columns(len(SUGGESTIONS))
        for col, s in zip(cols, SUGGESTIONS):
            if col.button(s, key=f"sugg_{key}_{s}", width="stretch"):
                question = s
    if history:
        if st.button("🧹 Clear conversation", key=f"clear_{key}"):
            st.session_state[f"chat_{key}"] = []
            st.rerun()
    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(question)
    with st.chat_message("assistant", avatar="🎙️"):
        with st.spinner("Searching transcripts and checking the evidence…"):
            prior = [{"role": m["role"], "content": m["content"]} for m in history[:-1]
                     if m["role"] == "user" or m.get("found", True)][-6:]
            resp = guard(lambda: api.chat(question, scope_ids, prior))
        if resp is None:
            history.pop()
            return
        msg = {"role": "assistant", "content": resp["answer"], "found": resp["found"],
               "answer_type": resp["answer_type"], "citations": resp["citations"], "model": resp.get("model"),
               "latency_ms": resp.get("latency_ms", 0)}
        render_assistant(msg)
    history.append(msg)
