"""Design system: CSS, animations and small HTML component helpers.

Every dynamic value that reaches ``unsafe_allow_html`` is escaped (transcripts are untrusted content)."""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

PIPELINE_STAGES = [("ingest", "Ingest"), ("transcribe", "Transcribe"), ("diarize", "Speakers"), ("chunk", "Chunk"),
                   ("extract", "Analyse"), ("validate", "Validate"), ("verify", "Verify"), ("persist", "Save"),
                   ("index", "Index")]

STATUS_META = {
    "completed": ("Ready", "ok"), "processing": ("Processing", "info"), "queued": ("Queued", "warn"),
    "failed": ("Failed", "bad"), "uploaded": ("Uploaded", "muted"),
}
VERIFY_META = {
    "verified": ("Verified", "ok"), "weak": ("Weak evidence", "warn"), "unsupported": ("Unsupported", "bad"),
    "conflicting": ("Conflict", "bad"), "unverified": ("Unverified", "muted"),
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_time(seconds: Optional[float]) -> str:
    seconds = int(round(seconds or 0))
    h, rem = divmod(max(seconds, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #070b1a; --bg2: #0c1228; --surface: rgba(255,255,255,.045); --surface-2: rgba(255,255,255,.07);
  --border: rgba(255,255,255,.09); --border-2: rgba(255,255,255,.16);
  --text: #e9edfb; --muted: #94a0c4; --faint: #64709a;
  --indigo: #6366f1; --violet: #8b5cf6; --cyan: #22d3ee; --emerald: #34d399; --amber: #fbbf24; --rose: #fb7185;
  --grad: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #22d3ee 130%);
  --radius: 18px; --shadow: 0 10px 40px -12px rgba(0,0,0,.6);
}
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp {
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(99,102,241,.22), transparent 60%),
    radial-gradient(800px 480px at 105% 5%, rgba(34,211,238,.13), transparent 60%),
    radial-gradient(700px 500px at 60% 115%, rgba(139,92,246,.18), transparent 60%),
    var(--bg);
  background-attachment: fixed; color: var(--text);
}
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 5rem; max-width: 1240px; }
h1, h2, h3, h4 { letter-spacing: -.02em; color: var(--text); }
p, li, label, span { color: inherit; }
a { color: var(--cyan) !important; text-decoration: none; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 10px; height: 10px; } ::-webkit-scrollbar-thumb { background: #2a3358; border-radius: 10px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(12,18,40,.96), rgba(7,11,26,.98)); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { border-radius: 12px !important; margin: 2px 0; transition: all .2s ease; }
[data-testid="stSidebarNavLink"]:hover { background: var(--surface-2) !important; transform: translateX(3px); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: linear-gradient(135deg, rgba(99,102,241,.35), rgba(139,92,246,.22)) !important; border: 1px solid rgba(139,92,246,.45); }
.mi-brand { display:flex; align-items:center; gap:.7rem; padding:.4rem .4rem 1rem; }
.mi-logo { width:42px; height:42px; border-radius:13px; background: var(--grad); display:grid; place-items:center; font-size:1.3rem; box-shadow: 0 8px 24px -6px rgba(99,102,241,.7); animation: float 6s ease-in-out infinite; }
.mi-brand b { font-size:1.02rem; letter-spacing:-.01em; display:block; line-height:1.15; }
.mi-brand small { color: var(--muted); font-size:.72rem; }

/* ---------- hero ---------- */
.mi-hero { position:relative; padding: 2.1rem 2.2rem; border-radius: 26px; border:1px solid var(--border); overflow:hidden;
  background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(139,92,246,.08) 45%, rgba(34,211,238,.10)); box-shadow: var(--shadow);
  animation: fadeUp .6s ease both; margin-bottom: 1.4rem; }
.mi-hero::before { content:""; position:absolute; inset:-40%; background: conic-gradient(from 90deg, transparent, rgba(99,102,241,.22), transparent 30%); animation: spin 14s linear infinite; }
.mi-hero > * { position:relative; }
.mi-eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.7rem; font-weight:700; color: var(--cyan); }
.mi-hero h1 { font-size: clamp(1.7rem, 3.4vw, 2.7rem); line-height:1.1; margin:.35rem 0 .5rem; font-weight:800;
  background: linear-gradient(100deg, #fff 10%, #c7d2fe 45%, #67e8f9 90%); background-size:200% auto; -webkit-background-clip:text; background-clip:text; color:transparent; animation: gradientShift 8s linear infinite; }
.mi-hero p { color: var(--muted); max-width: 62ch; margin: 0; font-size: 1.02rem; line-height:1.55; }

/* ---------- cards ---------- */
.mi-card { background: var(--surface); border:1px solid var(--border); border-radius: var(--radius); padding: 1.1rem 1.25rem; backdrop-filter: blur(14px);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; animation: fadeUp .5s ease both; margin-bottom:.85rem; }
.mi-card:hover { transform: translateY(-3px); border-color: var(--border-2); box-shadow: var(--shadow); }
.mi-card h4 { margin:0 0 .25rem; font-size:1.02rem; }
.mi-card .mi-sub { color: var(--muted); font-size:.82rem; }
.mi-card p { color:#cdd5ee; line-height:1.6; margin:.4rem 0 0; font-size:.93rem; }
.mi-row { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }
.mi-spread { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap; }

.mi-metric { background: var(--surface); border:1px solid var(--border); border-radius: var(--radius); padding: 1.05rem 1.2rem; position:relative; overflow:hidden; animation: fadeUp .55s ease both; transition: transform .25s ease, border-color .25s; height:100%; }
.mi-metric:hover { transform: translateY(-3px); border-color: var(--border-2); }
.mi-metric::after { content:""; position:absolute; right:-30px; top:-30px; width:110px; height:110px; border-radius:50%; background: var(--accent, var(--indigo)); opacity:.14; filter: blur(8px); }
.mi-metric .ic { font-size:1.25rem; opacity:.95; }
.mi-metric .val { font-size:2rem; font-weight:800; letter-spacing:-.03em; line-height:1.1; margin-top:.35rem; }
.mi-metric .lbl { color: var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
.mi-metric .hint { color: var(--faint); font-size:.75rem; margin-top:.15rem; }

/* ---------- badges & chips ---------- */
.mi-badge { display:inline-flex; align-items:center; gap:.4rem; padding:.18rem .62rem; border-radius:999px; font-size:.72rem; font-weight:600; border:1px solid transparent; white-space:nowrap; }
.mi-badge.ok { color:#6ee7b7; background:rgba(52,211,153,.12); border-color:rgba(52,211,153,.3); }
.mi-badge.info { color:#7dd3fc; background:rgba(34,211,238,.12); border-color:rgba(34,211,238,.3); }
.mi-badge.warn { color:#fcd34d; background:rgba(251,191,36,.12); border-color:rgba(251,191,36,.3); }
.mi-badge.bad { color:#fda4af; background:rgba(251,113,133,.12); border-color:rgba(251,113,133,.3); }
.mi-badge.muted { color:#a5b0d3; background:rgba(148,160,196,.12); border-color:rgba(148,160,196,.25); }
.mi-badge.violet { color:#c4b5fd; background:rgba(139,92,246,.14); border-color:rgba(139,92,246,.35); }
.mi-dot { width:7px; height:7px; border-radius:50%; background: currentColor; display:inline-block; }
.mi-dot.pulse { animation: pulse 1.4s ease-in-out infinite; }
.mi-chip { display:inline-flex; align-items:center; gap:.3rem; font-family:'JetBrains Mono', monospace; font-size:.72rem; padding:.12rem .5rem; border-radius:8px; background: rgba(99,102,241,.14); color:#c7d2fe; border:1px solid rgba(99,102,241,.3); margin-right:.3rem; }
.mi-pill { display:inline-block; padding:.2rem .65rem; border-radius:999px; font-size:.78rem; background: var(--surface-2); border:1px solid var(--border); margin:.15rem .25rem .15rem 0; color:#dbe2f7; }

/* ---------- progress ---------- */
.mi-progress { height:8px; border-radius:99px; background: rgba(255,255,255,.07); overflow:hidden; margin:.55rem 0 .2rem; }
.mi-progress > i { display:block; height:100%; border-radius:99px; background: var(--grad); background-size:200% 100%; animation: shimmer 1.6s linear infinite; transition: width .6s ease; }
.mi-stepper { display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.7rem; }
.mi-step { font-size:.7rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border); color: var(--faint); background: rgba(255,255,255,.02); }
.mi-step.done { color:#6ee7b7; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }
.mi-step.now { color:#fff; border-color: rgba(139,92,246,.7); background: linear-gradient(135deg, rgba(99,102,241,.5), rgba(139,92,246,.4)); animation: pulseGlow 1.6s ease-in-out infinite; }

/* ---------- transcript & evidence ---------- */
.mi-seg { display:grid; grid-template-columns: 68px 1fr; gap:.75rem; padding:.6rem .85rem; border-radius:14px; border:1px solid transparent; transition: background .2s, border-color .2s; }
.mi-seg:hover { background: var(--surface); border-color: var(--border); }
.mi-seg .t { font-family:'JetBrains Mono', monospace; font-size:.75rem; color: var(--cyan); padding-top:.2rem; }
.mi-seg .who { font-weight:700; font-size:.82rem; margin-right:.5rem; }
.mi-seg .txt { color:#d6dcf2; line-height:1.55; font-size:.93rem; }
.mi-seg.hl { background: rgba(251,191,36,.08); border-color: rgba(251,191,36,.3); }
.mi-quote { border-left:3px solid var(--violet); background: rgba(139,92,246,.07); padding:.5rem .8rem; border-radius:0 12px 12px 0; margin:.4rem 0; font-size:.86rem; color:#cfd6ef; }
.mi-quote b { color:#e9edfb; } .mi-quote .t { font-family:'JetBrains Mono', monospace; color: var(--cyan); font-size:.72rem; margin-right:.4rem; }
.mi-note { font-size:.78rem; color: var(--muted); margin-top:.35rem; }

/* ---------- empty ---------- */
.mi-empty { text-align:center; padding: 3rem 1.5rem; border:1px dashed var(--border-2); border-radius: 22px; background: var(--surface); animation: fadeUp .5s ease both; }
.mi-empty .em { font-size:2.6rem; animation: float 5s ease-in-out infinite; display:inline-block; }
.mi-empty h3 { margin:.6rem 0 .3rem; } .mi-empty p { color: var(--muted); max-width: 52ch; margin: 0 auto; }

/* ---------- streamlit widgets ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 12px; border:1px solid var(--border-2); background: var(--surface-2); color: var(--text); font-weight:600; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--violet); transform: translateY(-1px); box-shadow: 0 8px 22px -10px rgba(139,92,246,.7); color:#fff; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: var(--grad); border: none; color:#fff; box-shadow: 0 10px 28px -10px rgba(99,102,241,.8); }
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); transform: translateY(-2px); }
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div, .stDateInput input, .stNumberInput input { background: rgba(255,255,255,.04) !important; border-radius: 12px !important; border:1px solid var(--border) !important; color: var(--text) !important; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(139,92,246,.25) !important; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed rgba(139,92,246,.5); border-radius: 22px; background: linear-gradient(135deg, rgba(99,102,241,.08), rgba(34,211,238,.05)); padding: 2rem; transition: all .25s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cyan); background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(34,211,238,.1)); transform: scale(1.005); }
.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--border); overflow-x:auto; }
.stTabs [data-baseweb="tab"] { border-radius: 12px 12px 0 0; padding:.55rem 1rem; color: var(--muted); font-weight:600; transition: color .2s, background .2s; }
.stTabs [data-baseweb="tab"]:hover { color:#fff; background: var(--surface); }
.stTabs [aria-selected="true"] { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--grad) !important; height:3px; border-radius:3px; }
[data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius: 14px !important; background: var(--surface); }
[data-testid="stChatMessage"] { background: var(--surface); border:1px solid var(--border); border-radius: 18px; padding: .9rem 1rem; animation: fadeUp .4s ease both; }
[data-testid="stChatInput"] textarea { border-radius: 16px !important; }
[data-testid="stMetric"] { background: var(--surface); border:1px solid var(--border); border-radius: 16px; padding:.8rem 1rem; }
[data-testid="stAlert"] { border-radius: 14px; }
div[data-testid="stForm"] { border:1px solid var(--border); border-radius: var(--radius); background: var(--surface); }

/* ---------- animations ---------- */
@keyframes fadeUp { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform:none; } }
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.35; transform:scale(1.5); } }
@keyframes pulseGlow { 0%,100% { box-shadow: 0 0 0 0 rgba(139,92,246,.55); } 50% { box-shadow: 0 0 0 7px rgba(139,92,246,0); } }
@keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes gradientShift { to { background-position: 200% center; } }

/* stagger children in grids */
[data-testid="stHorizontalBlock"] > div:nth-child(1) .mi-metric, [data-testid="stHorizontalBlock"] > div:nth-child(1) .mi-card { animation-delay: .02s; }
[data-testid="stHorizontalBlock"] > div:nth-child(2) .mi-metric, [data-testid="stHorizontalBlock"] > div:nth-child(2) .mi-card { animation-delay: .09s; }
[data-testid="stHorizontalBlock"] > div:nth-child(3) .mi-metric, [data-testid="stHorizontalBlock"] > div:nth-child(3) .mi-card { animation-delay: .16s; }
[data-testid="stHorizontalBlock"] > div:nth-child(4) .mi-metric, [data-testid="stHorizontalBlock"] > div:nth-child(4) .mi-card { animation-delay: .23s; }

/* ---------- responsive ---------- */
@media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } .mi-hero { padding: 1.4rem 1.2rem; border-radius: 20px; } }
@media (max-width: 640px) { .mi-seg { grid-template-columns: 54px 1fr; padding:.5rem .5rem; } .mi-metric .val { font-size:1.6rem; } .mi-hero p { font-size:.95rem; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="mi-brand"><div class="mi-logo">🎙️</div><div><b>Meeting Intelligence</b>'
        '<small>Transcribe · Understand · Act</small></div></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "AI Meeting Intelligence") -> None:
    st.markdown(f'<div class="mi-hero"><div class="mi-eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def badge(status: str, meta: Optional[Dict[str, tuple]] = None, label: Optional[str] = None) -> str:
    text, tone = (meta or STATUS_META).get(status, (status.title(), "muted"))
    dot = '<span class="mi-dot pulse"></span>' if status in ("processing", "queued") else '<span class="mi-dot"></span>'
    return f'<span class="mi-badge {tone}">{dot}{esc(label or text)}</span>'


def verify_badge(status: str) -> str:
    return badge(status, VERIFY_META)


def chip(text: str) -> str:
    return f'<span class="mi-chip">{esc(text)}</span>'


def pill(text: str) -> str:
    return f'<span class="mi-pill">{esc(text)}</span>'


def metric_card(icon: str, label: str, value: Any, hint: str = "", accent: str = "var(--indigo)") -> str:
    return (f'<div class="mi-metric" style="--accent:{accent}"><div class="ic">{esc(icon)}</div>'
            f'<div class="val">{esc(value)}</div><div class="lbl">{esc(label)}</div>'
            f'<div class="hint">{esc(hint)}</div></div>')


def progress_html(fraction: float, stage: Optional[str] = None) -> str:
    pct = max(0, min(100, int((fraction or 0) * 100)))
    keys = [k for k, _ in PIPELINE_STAGES]
    idx = keys.index(stage) if stage in keys else (len(keys) if stage == "done" else -1)
    steps = []
    for i, (k, label) in enumerate(PIPELINE_STAGES):
        cls = "done" if i < idx or stage == "done" else ("now" if i == idx else "")
        steps.append(f'<span class="mi-step {cls}">{esc(label)}</span>')
    return (f'<div class="mi-progress"><i style="width:{pct}%"></i></div>'
            f'<div class="mi-note">{pct}% · {esc((stage or "starting").title())}</div>'
            f'<div class="mi-stepper">{"".join(steps)}</div>')


def empty_state(icon: str, title: str, text: str) -> None:
    st.markdown(f'<div class="mi-empty"><span class="em">{esc(icon)}</span><h3>{esc(title)}</h3>'
                f'<p>{esc(text)}</p></div>', unsafe_allow_html=True)


def evidence_html(evidence: Iterable[Dict[str, Any]]) -> str:
    out = []
    for e in evidence:
        who = f'<b>{esc(e.get("speaker"))}</b> ' if e.get("speaker") else ""
        out.append(f'<div class="mi-quote"><span class="t">{esc(fmt_time(e.get("start_time")))}</span>'
                   f'{chip(e.get("ref", ""))} {who}{esc(e.get("text"))}</div>')
    return "".join(out) or '<div class="mi-note">No evidence cited.</div>'


def meeting_card(mt: Dict[str, Any]) -> str:
    counts = mt.get("counts") or {}
    status = mt.get("status", "uploaded")
    bits = [f'🕒 {esc(fmt_duration(mt.get("duration")))}']
    if counts.get("decisions"):
        bits.append(f'✅ {counts["decisions"]} decisions')
    if counts.get("action_items"):
        bits.append(f'📌 {counts["action_items"]} actions')
    if counts.get("risks"):
        bits.append(f'⚠️ {counts["risks"]} risks')
    review = badge("weak", VERIFY_META, "Needs review") if mt.get("needs_review") else ""
    body = ""
    if status in ("processing", "queued"):
        body = progress_html(mt.get("progress", 0), mt.get("stage"))
    elif status == "failed":
        body = f'<p style="color:#fda4af">{esc(mt.get("error_message") or "Processing failed.")}</p>'
    elif mt.get("overview"):
        o = mt["overview"]
        body = f'<p>{esc(o if len(o) < 230 else o[:227] + "…")}</p>'
    created = esc((mt.get("created_at") or "")[:16].replace("T", " "))
    return (f'<div class="mi-card"><div class="mi-spread"><div><h4>{esc(mt.get("title"))}</h4>'
            f'<div class="mi-sub">{created} UTC · {" · ".join(bits)}</div></div>'
            f'<div class="mi-row">{review}{badge(status)}</div></div>{body}</div>')
