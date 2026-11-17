"""Meeting review: summary, decisions, actions, risks, transcript, Q&A, export, pipeline metrics."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import streamlit as st

import api_client as api
import theme
from common import chat_ui, guard

# ------------------------------------------------------------ pick a meeting
meetings = guard(lambda: api.list_meetings(), []) or []
if not meetings:
    theme.hero("Meeting review", "Inspect transcripts and AI-generated notes, correct them and export.", "Review")
    theme.empty_state("📝", "Nothing to review yet", "Upload a recording or load the sample meetings from the dashboard.")
    if st.button("Go to dashboard", type="primary"):
        st.switch_page("pages/dashboard.py")
    st.stop()

ids = [m["id"] for m in meetings]
if st.session_state.get("selected_meeting") not in ids:
    st.session_state["selected_meeting"] = ids[0]
labels = {m["id"]: f'{m["title"]}  ·  {m["created_at"][:10]}' for m in meetings}
chosen = st.selectbox("Meeting", ids, index=ids.index(st.session_state["selected_meeting"]),
                      format_func=lambda i: labels[i], label_visibility="collapsed")
st.session_state["selected_meeting"] = chosen
mid = chosen

mt = guard(lambda: api.get_meeting(mid))
if not mt:
    st.stop()

# ------------------------------------------------------------------- header
flag = theme.badge("weak", theme.VERIFY_META, "Needs review") if mt["needs_review"] else ""
src = '<span class="mi-badge violet">Sample</span>' if mt["source"] == "demo" else ""
counts = mt["counts"]
st.markdown(
    f'<div class="mi-hero"><div class="mi-row">{theme.badge(mt["status"])}{flag}{src}</div>'
    f'<h1>{theme.esc(mt["title"])}</h1>'
    f'<p>🕒 {theme.fmt_duration(mt.get("duration"))} · 💬 {counts["segments"]} segments · '
    f'✅ {counts["decisions"]} decisions · 📌 {counts["action_items"]} actions · ⚠️ {counts["risks"]} risks'
    f'{" · 🌐 " + theme.esc(mt["language"]) if mt.get("language") else ""}</p></div>', unsafe_allow_html=True)

if mt["status"] in ("queued", "processing"):
    def _live() -> None:
        cur = guard(lambda: api.get_meeting(mid))
        if cur and cur["status"] in ("queued", "processing"):
            st.markdown(f'<div class="mi-card">{theme.progress_html(cur["progress"], cur["stage"])}</div>',
                        unsafe_allow_html=True)
        else:
            st.rerun()
    st.fragment(run_every=2)(_live)()
    st.info("This page updates automatically when processing finishes.")
    st.stop()

if mt["status"] == "failed":
    st.error(mt.get("error_message") or "Processing failed.")
elif mt["status"] == "uploaded":
    st.warning("This recording has been uploaded but not processed yet.")

a1, a2, a3, a4 = st.columns([1.2, 1.2, 1, 3])
if a1.button("🔄 Process again", width="stretch", help="Re-run analysis on the existing transcript"):
    if guard(lambda: api.start_processing(mid)):
        st.rerun()
if mt.get("has_audio") and a2.button("🎙️ Re-transcribe", width="stretch", help="Run speech-to-text again"):
    if guard(lambda: api.start_processing(mid, force=True)):
        st.rerun()


@st.dialog("Delete this meeting?")
def _confirm_delete() -> None:
    st.write("This permanently removes the recording, transcript, notes and search index.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Delete", type="primary", width="stretch"):
        if guard(lambda: api.delete_meeting(mid)):
            st.session_state.pop("selected_meeting", None)
            st.switch_page("pages/dashboard.py")


if a3.button("🗑️ Delete", width="stretch"):
    _confirm_delete()
with a4.popover("✏️ Rename"):
    new_title = st.text_input("Title", value=mt["title"], key=f"title_{mid}")
    if st.button("Save title", key=f"save_title_{mid}") and new_title.strip():
        if guard(lambda: api.rename_meeting(mid, new_title.strip())):
            st.rerun()

if mt["status"] != "completed" and not counts["segments"]:
    st.stop()

# ------------------------------------------------------------------- audio
if mt.get("has_audio"):
    with st.expander("🎧 Recording", expanded="audio_start" in st.session_state):
        key = f"audio_{mid}"
        if st.button("Load audio player", key=f"load_audio_{mid}") or key in st.session_state:
            if key not in st.session_state:
                st.session_state[key] = guard(lambda: api.audio_bytes(mid))
            if st.session_state.get(key):
                st.audio(st.session_state[key], start_time=int(st.session_state.get("audio_start", 0)))


def play_button(seconds: float, key: str) -> None:
    if mt.get("has_audio") and st.button(f"▶ {theme.fmt_time(seconds)}", key=key, help="Play the recording from here"):
        st.session_state["audio_start"] = int(seconds)
        st.rerun()


notes: Dict[str, Any] = guard(lambda: api.notes(mid), {}) or {}
tr: Dict[str, Any] = guard(lambda: api.transcript(mid), {}) or {}

tabs = st.tabs(["📋 Overview", "✅ Decisions", "📌 Action items", "⚠️ Risks & questions", "📜 Transcript",
                "💬 Ask", "📤 Export", "🛠️ Pipeline"])
t_over, t_dec, t_act, t_risk, t_tr, t_ask, t_exp, t_pipe = tabs


# --------------------------------------------------------------- helpers
def review_controls(kind: str, item: Dict[str, Any], updater, extra_fields=None) -> None:
    """Shared edit form for decisions/actions/risks."""
    with st.expander("Evidence & edit"):
        st.markdown(theme.evidence_html(item.get("evidence", [])), unsafe_allow_html=True)
        if item.get("verification_note"):
            st.caption(f"Verification: {item['verification_note']}")
        if item.get("verification_score") is not None:
            st.caption(f"Evidence match score {item['verification_score']:.2f}")
        with st.form(f"form_{kind}_{item['id']}"):
            fields = extra_fields(item) if extra_fields else {}
            c1, c2 = st.columns(2)
            ok = c1.form_submit_button("💾 Save & mark reviewed", type="primary", width="stretch")
            flag_it = c2.form_submit_button("🚩 Flag for review", width="stretch")
            if ok or flag_it:
                payload = {**fields, **({"needs_review": True} if flag_it else {})}
                if guard(lambda: updater(item["id"], **payload)):
                    st.toast("Saved", icon="✅")
                    st.rerun()
        r1, r2, _ = st.columns([1, 1, 4])
        if r1.button("👍", key=f"up_{kind}_{item['id']}", help="This is accurate"):
            guard(lambda: api.feedback(kind, 1, mid, str(item["id"])))
            st.toast("Thanks for the feedback", icon="👍")
        if r2.button("👎", key=f"down_{kind}_{item['id']}", help="This is wrong or missing context"):
            guard(lambda: api.feedback(kind, -1, mid, str(item["id"])))
            st.toast("Feedback recorded", icon="👎")


def item_header(title_html: str, item: Dict[str, Any], extra: str = "") -> str:
    edited = '<span class="mi-badge violet">Edited</span>' if item.get("user_edited") else ""
    flag = '<span class="mi-badge warn">⚑ Review</span>' if item.get("needs_review") else ""
    return (f'<div class="mi-card"><div class="mi-spread"><div style="flex:1;min-width:220px"><h4>{title_html}</h4>{extra}</div>'
            f'<div class="mi-row">{flag}{edited}{theme.verify_badge(item.get("verification_status", "unverified"))}</div></div></div>')


def evidence_players(kind: str, item: Dict[str, Any]) -> None:
    evs = item.get("evidence", [])[:4]
    if evs and mt.get("has_audio"):
        cols = st.columns(len(evs) + 2)
        for col, e in zip(cols, evs):
            with col:
                play_button(e["start_time"], f"play_{kind}_{item['id']}_{e['ref']}")


# ---------------------------------------------------------------- overview
with t_over:
    s = notes.get("summary")
    if notes.get("needs_review"):
        st.warning("Some AI-generated items could not be fully verified against the transcript. "
                   "Look for the ⚑ Review badges and check the cited evidence.")
    if not s:
        theme.empty_state("🧠", "No summary yet", "Process the meeting to generate notes.")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f'<div class="mi-card"><h4>Overview</h4><p>{theme.esc(s["overview"])}</p>'
                        + (f'<div class="mi-note"><b>Purpose:</b> {theme.esc(s["purpose"])}</div>' if s.get("purpose") else "")
                        + '</div>', unsafe_allow_html=True)
            if s["key_points"]:
                pts = "".join(f"<li>{theme.esc(p)}</li>" for p in s["key_points"])
                st.markdown(f'<div class="mi-card"><h4>Key points</h4><ul style="color:#cdd5ee;line-height:1.7">{pts}</ul></div>',
                            unsafe_allow_html=True)
            if s.get("outcome"):
                st.markdown(f'<div class="mi-card"><h4>Outcome</h4><p>{theme.esc(s["outcome"])}</p></div>',
                            unsafe_allow_html=True)
        with right:
            if s["main_topics"]:
                st.markdown('<div class="mi-card"><h4>Main topics</h4>' + "".join(theme.pill(t) for t in s["main_topics"])
                            + '</div>', unsafe_allow_html=True)
            speakers = tr.get("speakers", [])
            if speakers:
                st.markdown('<div class="mi-card"><h4>Speakers</h4>'
                            + "".join(theme.pill(sp["display_name"] or sp["speaker_label"]) for sp in speakers)
                            + '</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="mi-card"><h4>Generated by</h4><div class="mi-note">{theme.esc(s["generated_by"])}<br/>'
                f'prompts {theme.esc(s["prompt_version"])} · version {s["version"]}'
                f'{" · edited by you" if s["edited_by_user"] else ""}</div></div>', unsafe_allow_html=True)
        with st.expander("✏️ Edit summary"):
            with st.form(f"edit_summary_{mid}"):
                ov = st.text_area("Overview", s["overview"], height=110)
                pu = st.text_input("Purpose", s.get("purpose", ""))
                tp = st.text_input("Main topics (comma separated)", ", ".join(s["main_topics"]))
                kp = st.text_area("Key points (one per line)", "\n".join(s["key_points"]), height=140)
                oc = st.text_area("Outcome", s.get("outcome", ""), height=70)
                if st.form_submit_button("Save new version", type="primary"):
                    if guard(lambda: api.edit_summary(
                            mid, overview=ov, purpose=pu, outcome=oc,
                            main_topics=[t.strip() for t in tp.split(",") if t.strip()],
                            key_points=[k.strip() for k in kp.splitlines() if k.strip()])):
                        st.toast("Summary updated", icon="✅")
                        st.rerun()

# --------------------------------------------------------------- decisions
with t_dec:
    decisions = notes.get("decisions", [])
    if not decisions:
        theme.empty_state("🤝", "No explicit decisions found", "Only clearly stated decisions are extracted — "
                                                              "ideas and proposals are not counted.")
    for d in decisions:
        st.markdown(item_header(theme.esc(d["description"]), d,
                                f'<div class="mi-sub">Status: {theme.esc(d["status"])}</div>'), unsafe_allow_html=True)
        evidence_players('decision', d)

        def _dec_fields(item=d) -> Dict[str, Any]:
            text = st.text_area("Decision", item["description"], key=f"dtext_{item['id']}", height=80)
            status = st.selectbox("Status", ["proposed", "confirmed", "rejected"],
                                  index=["proposed", "confirmed", "rejected"].index(item["status"]),
                                  key=f"dstat_{item['id']}")
            return {"description": text, "status": status}
        review_controls("decision", d, api.update_decision, _dec_fields)

# ------------------------------------------------------------------ actions
with t_act:
    actions = notes.get("action_items", [])
    if not actions:
        theme.empty_state("📌", "No action items found", "Tasks are extracted only when someone clearly commits to them.")
    STATUS = ["open", "in_progress", "done", "dropped"]
    for a in actions:
        owner = theme.esc(a["owner"]) if a["owner"] else '<span style="color:var(--faint)">Unassigned</span>'
        due = theme.esc(a["due_date"]) if a["due_date"] else '<span style="color:var(--faint)">No due date</span>'
        meta = (f'<div class="mi-sub">👤 {owner} &nbsp;·&nbsp; 📅 {due} &nbsp;·&nbsp; '
                f'{theme.badge(a["status"], {"open": ("Open", "info"), "in_progress": ("In progress", "warn"), "done": ("Done", "ok"), "dropped": ("Dropped", "muted")})}</div>')
        st.markdown(item_header(theme.esc(a["task"]), a, meta), unsafe_allow_html=True)
        evidence_players('action_item', a)

        def _act_fields(item=a) -> Dict[str, Any]:
            task = st.text_area("Task", item["task"], key=f"atask_{item['id']}", height=80)
            c1, c2, c3 = st.columns(3)
            owner_v = c1.text_input("Owner", item["owner"] or "", key=f"aowner_{item['id']}")
            due_v = c2.text_input("Due", item["due_date"] or "", key=f"adue_{item['id']}")
            status_v = c3.selectbox("Status", STATUS, index=STATUS.index(item["status"]), key=f"astat_{item['id']}")
            return {"task": task, "owner": owner_v, "due_date": due_v, "status": status_v}
        review_controls("action_item", a, api.update_action_item, _act_fields)

# -------------------------------------------------------------------- risks
with t_risk:
    risks, questions = notes.get("risks", []), notes.get("open_questions", [])
    if not risks and not questions:
        theme.empty_state("🛡️", "No risks or open questions", "Nothing was flagged as a blocker, risk or unresolved question.")
    kind_meta = {"risk": ("Risk", "warn"), "blocker": ("Blocker", "bad"), "open_question": ("Open question", "info")}
    for r in risks + questions:
        sev = f' · severity {theme.esc(r["severity"])}' if r.get("severity") else ""
        meta = (f'<div class="mi-sub">{theme.badge(r["kind"], kind_meta)}{sev} · {theme.esc(r["status"])}</div>')
        st.markdown(item_header(theme.esc(r["description"]), r, meta), unsafe_allow_html=True)
        evidence_players('risk', r)

        def _risk_fields(item=r) -> Dict[str, Any]:
            text = st.text_area("Description", item["description"], key=f"rtext_{item['id']}", height=80)
            status = st.selectbox("Status", ["open", "resolved"], index=0 if item["status"] == "open" else 1,
                                  key=f"rstat_{item['id']}")
            return {"description": text, "status": status}
        review_controls("risk", r, api.update_risk, _risk_fields)

# --------------------------------------------------------------- transcript
with t_tr:
    segs: List[Dict[str, Any]] = tr.get("segments", [])
    speakers = tr.get("speakers", [])
    if not segs:
        theme.empty_state("📜", "No transcript yet", "Process the meeting to generate a transcript.")
    else:
        f1, f2 = st.columns([2, 1.3])
        q = f1.text_input("Search transcript", placeholder="🔎 Find a word or phrase…", key=f"tq_{mid}",
                          label_visibility="collapsed")
        names = sorted({s["speaker_name"] for s in segs if s.get("speaker_name")})
        picked = f2.multiselect("Speakers", names, placeholder="All speakers", key=f"tsp_{mid}",
                                label_visibility="collapsed") if names else []
        highlight = {r for r in re.findall(r"\w+", q.lower())} if q else set()
        palette = ["#a5b4fc", "#67e8f9", "#6ee7b7", "#fcd34d", "#fda4af", "#c4b5fd", "#f9a8d4"]
        colour = {n: palette[i % len(palette)] for i, n in enumerate(names)}
        rows = []
        shown = 0
        for sg in segs:
            if picked and sg.get("speaker_name") not in picked:
                continue
            if q and q.lower() not in sg["text"].lower():
                continue
            shown += 1
            text = theme.esc(sg["text"])
            if q:
                text = re.sub(f"({re.escape(theme.esc(q))})", r"<mark style='background:#fbbf2466;color:#fff;border-radius:4px'>\1</mark>",
                              text, flags=re.I)
            who = (f'<span class="who" style="color:{colour.get(sg["speaker_name"], "#a5b4fc")}">{theme.esc(sg["speaker_name"])}</span>'
                   if sg.get("speaker_name") else "")
            edited = " ✎" if sg.get("edited") else ""
            rows.append(f'<div class="mi-seg"><div class="t">{theme.fmt_time(sg["start_time"])}</div>'
                        f'<div class="txt">{who}{text}{edited}<span style="display:none">{theme.esc(sg["ref"])}</span></div></div>')
        st.caption(f"{shown} of {len(segs)} segments")
        with st.container(height=520, border=True):
            st.markdown("".join(rows) or '<div class="mi-note">No matching segments.</div>', unsafe_allow_html=True)

        with st.expander("🧑‍🤝‍🧑 Speakers — name them"):
            if not speakers:
                st.caption("No speaker labels were detected. Enable diarization in the server settings to label speakers.")
            for sp in speakers:
                c1, c2 = st.columns([1, 2])
                c1.markdown(f'<div class="mi-pill">{theme.esc(sp["speaker_label"])}</div>', unsafe_allow_html=True)
                new = c2.text_input("Display name", sp["display_name"] or "", key=f"spk_{sp['id']}",
                                    label_visibility="collapsed", placeholder="Real name (confirmed by you)")
                if new != (sp["display_name"] or "") and c2.button("Save", key=f"spk_save_{sp['id']}"):
                    if guard(lambda: api.rename_speaker(mid, sp["id"], new)):
                        st.toast("Speaker updated — search index refreshed", icon="✅")
                        st.rerun()
        with st.expander("✂️ Correct a segment"):
            opts = {f'{theme.fmt_time(s_["start_time"])} · {s_["text"][:70]}': s_ for s_ in segs}
            pick = st.selectbox("Segment", list(opts), key=f"seg_pick_{mid}")
            seg = opts[pick]
            new_text = st.text_area("Text", seg["text"], key=f"seg_text_{seg['id']}", height=90)
            sp_opts = {"— no speaker —": None, **{(sp["display_name"] or sp["speaker_label"]): sp["id"] for sp in speakers}}
            cur = next((k for k, v in sp_opts.items() if v == seg["speaker_id"]), "— no speaker —")
            new_sp = st.selectbox("Speaker", list(sp_opts), index=list(sp_opts).index(cur), key=f"seg_sp_{seg['id']}")
            if st.button("Save correction", key=f"seg_save_{seg['id']}", type="primary"):
                payload: Dict[str, Any] = {"text": new_text}
                if sp_opts[new_sp] is None:
                    payload["clear_speaker"] = True
                else:
                    payload["speaker_id"] = sp_opts[new_sp]
                if guard(lambda: api.update_segment(mid, seg["id"], **payload)):
                    st.toast("Segment corrected", icon="✅")
                    st.rerun()

# ---------------------------------------------------------------------- ask
with t_ask:
    st.caption("Answers come only from this meeting's transcript, with the supporting excerpts attached.")
    chat_ui([mid], key=f"meeting_{mid}", placeholder="Ask about this meeting…")

# ------------------------------------------------------------------- export
with t_exp:
    st.markdown('<div class="mi-card"><h4>Export reviewed notes</h4><p>Includes summary, decisions, action items with '
                'owners and due dates, risks, open questions and evidence references. Your corrections are included.</p></div>',
                unsafe_allow_html=True)
    inc = st.toggle("Include full transcript", value=True, key=f"inc_{mid}")
    c1, c2 = st.columns(2)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", mt["title"]).strip("_")[:50] or "meeting"
    with c1:
        if st.button("📝 Prepare Markdown", width="stretch", key=f"prep_md_{mid}"):
            st.session_state[f"md_{mid}"] = guard(lambda: api.export(mid, "md", inc))
        if st.session_state.get(f"md_{mid}"):
            st.download_button("⬇️ Download .md", st.session_state[f"md_{mid}"], f"{safe}.md", "text/markdown",
                               width="stretch", type="primary")
    with c2:
        if st.button("📄 Prepare PDF", width="stretch", key=f"prep_pdf_{mid}"):
            st.session_state[f"pdf_{mid}"] = guard(lambda: api.export(mid, "pdf", inc))
        if st.session_state.get(f"pdf_{mid}"):
            st.download_button("⬇️ Download .pdf", st.session_state[f"pdf_{mid}"], f"{safe}.pdf", "application/pdf",
                               width="stretch", type="primary")

# ----------------------------------------------------------------- pipeline
with t_pipe:
    events = mt.get("events", [])
    if not events:
        st.caption("No pipeline events recorded yet.")
    else:
        total = sum(e["duration_ms"] for e in events)
        toks = sum((e.get("input_tokens") or 0) + (e.get("output_tokens") or 0) for e in events)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total processing time", f"{total / 1000:.1f}s")
        c2.metric("LLM tokens", f"{toks:,}" if toks else "n/a")
        c3.metric("Retries", sum(e.get("retries", 0) for e in events))
        st.dataframe(
            [{"Stage": e["stage"], "Status": e["status"], "Time (ms)": e["duration_ms"], "Provider": e.get("provider") or "",
              "Model": e.get("model") or "", "In tokens": e.get("input_tokens"), "Out tokens": e.get("output_tokens"),
              "Retries": e.get("retries", 0), "Detail": e.get("detail") or ""} for e in events],
            width="stretch", hide_index=True)
