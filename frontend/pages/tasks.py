"""Cross-meeting action board."""
from __future__ import annotations

import streamlit as st

import api_client as api
import theme
from common import go_to_meeting, guard

theme.hero("Action board", "Every follow-up from every meeting, in one place. Move tasks forward as work gets done.",
           "Follow-through")

items = guard(lambda: api.action_items(), []) or []
titles = {m["id"]: m["title"] for m in (guard(lambda: api.list_meetings(), []) or [])}
if not items:
    theme.empty_state("📌", "No action items yet", "Action items appear here once a meeting has been processed.")
    st.stop()

owners = sorted({i["owner"] for i in items if i["owner"]})
f1, f2, f3 = st.columns([1.2, 1.5, 1])
view = f1.segmented_control("View", ["Board", "Table"], default="Board", label_visibility="collapsed") or "Board"
owner = f2.selectbox("Owner", ["Everyone", "Unassigned", *owners], label_visibility="collapsed")
hide_dropped = f3.toggle("Hide dropped", value=True)

filtered = [i for i in items if not (hide_dropped and i["status"] == "dropped")]
if owner == "Unassigned":
    filtered = [i for i in filtered if not i["owner"]]
elif owner != "Everyone":
    filtered = [i for i in filtered if i["owner"] == owner]

total, done = len(filtered), sum(1 for i in filtered if i["status"] == "done")
st.markdown(f'<div class="mi-progress"><i style="width:{int(100 * done / total) if total else 0}%"></i></div>'
            f'<div class="mi-note">{done} of {total} done</div>', unsafe_allow_html=True)
st.write("")

if view == "Table":
    st.dataframe([{"Task": i["task"], "Owner": i["owner"] or "—", "Due": i["due_date"] or "—", "Status": i["status"],
                   "Meeting": titles.get(i["meeting_id"], i["meeting_id"]),
                   "Verification": i["verification_status"]} for i in filtered],
                 width="stretch", hide_index=True)
    st.stop()

NEXT = {"open": ("in_progress", "▶ Start"), "in_progress": ("done", "✔ Done"), "done": ("open", "↺ Reopen")}
columns = st.columns(3)
for col, (status, label, accent) in zip(columns, [("open", "To do", "var(--indigo)"),
                                                  ("in_progress", "In progress", "var(--amber)"),
                                                  ("done", "Done", "var(--emerald)")]):
    group = [i for i in filtered if i["status"] == status]
    with col:
        st.markdown(f'<div class="mi-row" style="margin-bottom:.6rem"><b style="color:{accent}">●</b> <b>{label}</b> '
                    f'<span class="mi-badge muted">{len(group)}</span></div>', unsafe_allow_html=True)
        if not group:
            st.markdown('<div class="mi-note">Nothing here.</div>', unsafe_allow_html=True)
        for it in group:
            owner_txt = theme.esc(it["owner"]) if it["owner"] else "Unassigned"
            due_txt = theme.esc(it["due_date"]) if it["due_date"] else "no due date"
            flag = '<span class="mi-badge warn">⚑ Review</span>' if it["needs_review"] else ""
            st.markdown(
                f'<div class="mi-card"><div class="mi-spread"><div class="mi-sub">{theme.esc(titles.get(it["meeting_id"], ""))}</div>{flag}</div>'
                f'<p style="color:#e9edfb;margin:.3rem 0 .5rem">{theme.esc(it["task"])}</p>'
                f'<div class="mi-sub">👤 {owner_txt} · 📅 {due_txt}</div></div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            nxt, nlabel = NEXT[status]
            if b1.button(nlabel, key=f"mv_{it['id']}", width="stretch"):
                if guard(lambda: api.update_action_item(it["id"], status=nxt)):
                    st.rerun()
            if b2.button("Open", key=f"op_{it['id']}", width="stretch"):
                go_to_meeting(it["meeting_id"])
