"""Dashboard: KPIs, quick actions and live-updating meeting list (PRD FR-12)."""
from __future__ import annotations

import streamlit as st

import api_client as api
import theme
from common import any_active, go_to_meeting, guard

theme.hero("Every meeting, turned into knowledge",
           "Upload a recording and get a timestamped transcript, decisions, action items and risks — all searchable "
           "and backed by evidence from the conversation.")

stats = guard(api.stats, None)
if stats:
    hours = stats["total_duration_seconds"] / 3600
    cols = st.columns(5)
    cards = [
        ("🎧", "Meetings", stats["total_meetings"], f'{stats["by_status"].get("completed", 0)} ready', "var(--indigo)"),
        ("⏱️", "Hours captured", f"{hours:.1f}", "total recorded time", "var(--cyan)"),
        ("✅", "Decisions", stats["total_decisions"], "extracted with evidence", "var(--emerald)"),
        ("📌", "Open actions", stats["open_action_items"], "waiting on someone", "var(--amber)"),
        ("🔍", "Needs review", stats["needs_review"], "flagged by verification", "var(--rose)"),
    ]
    for col, (ic, lbl, val, hint, accent) in zip(cols, cards):
        col.markdown(theme.metric_card(ic, lbl, val, hint, accent), unsafe_allow_html=True)
    st.write("")

c1, c2, c3 = st.columns([1, 1, 2.2])
if c1.button("⬆️ New meeting", type="primary", width="stretch"):
    st.switch_page("pages/upload.py")
if c2.button("✨ Load samples", width="stretch", help="Adds two ready-made demo meetings (no audio needed)"):
    with st.spinner("Creating sample meetings…"):
        if guard(api.create_demo):
            st.toast("Sample meetings added — processing…", icon="✨")
            st.rerun()

f1, f2 = st.columns([2, 1])
query = f1.text_input("Search meetings", placeholder="🔎  Search by title…", label_visibility="collapsed")
status_filter = f2.selectbox("Status", ["All", "completed", "processing", "queued", "failed"],
                             label_visibility="collapsed")

meetings = guard(lambda: api.list_meetings(None if status_filter == "All" else status_filter, query or None), []) or []
active = any_active(meetings)


def render_list() -> None:
    items = guard(lambda: api.list_meetings(None if status_filter == "All" else status_filter, query or None), []) or []
    if active and not any_active(items):
        st.rerun()  # processing just finished: refresh KPIs and stop polling
    if not items:
        theme.empty_state("🎙️", "No meetings yet",
                          "Upload your first recording, or load the sample meetings to explore the product.")
        return
    cols = st.columns(2)
    for i, mt in enumerate(items):
        with cols[i % 2]:
            st.markdown(theme.meeting_card(mt), unsafe_allow_html=True)
            b1, b2, _ = st.columns([1, 1, 2])
            if b1.button("Open", key=f"open_{mt['id']}", width="stretch",
                         disabled=mt["status"] in ("queued", "processing")):
                go_to_meeting(mt["id"])
            if mt["status"] in ("failed", "uploaded") and b2.button("Process", key=f"proc_{mt['id']}",
                                                                      width="stretch", type="primary"):
                if guard(lambda: api.start_processing(mt["id"])):
                    st.rerun()


# Poll only while something is running, so idle dashboards stay quiet.
st.fragment(run_every=2.5 if active else None)(render_list)()
