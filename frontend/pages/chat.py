"""Cross-meeting question answering (PRD FR-08, FR-09)."""
from __future__ import annotations

import streamlit as st

import api_client as api
import theme
from common import chat_ui, guard

theme.hero("Ask your meetings", "Questions are answered only from transcript evidence. Search one meeting or all of them — "
                                "every answer links back to who said what, and when.", "Cross-meeting memory")

meetings = [m for m in (guard(lambda: api.list_meetings("completed"), []) or [])]
if not meetings:
    theme.empty_state("💬", "No indexed meetings yet",
                      "Process a meeting first — once it is ready you can ask questions across everything you've recorded.")
    if st.button("Load sample meetings", type="primary"):
        if guard(api.create_demo):
            st.rerun()
    st.stop()

titles = {m["id"]: m["title"] for m in meetings}
c1, c2 = st.columns([3, 1])
scope = c1.multiselect("Search scope", list(titles), format_func=lambda i: titles[i], placeholder="All meetings",
                       help="Leave empty to search every meeting you can access.")
c2.markdown(f'<div class="mi-card" style="margin-top:1.65rem"><div class="mi-sub">Searching</div>'
            f'<b>{len(scope) or len(meetings)}</b> meeting{"s" if (len(scope) or len(meetings)) != 1 else ""}</div>',
            unsafe_allow_html=True)
st.markdown('<div class="mi-note">🟢 <b>Direct statement</b> — said in the meeting &nbsp; · &nbsp; 🟣 <b>Synthesized</b> — '
            'combined from several excerpts / meetings &nbsp; · &nbsp; 🔴 <b>Not found</b> — the evidence is insufficient</div>',
            unsafe_allow_html=True)
st.write("")
chat_ui(scope or None, key="global")
