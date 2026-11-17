"""Streamlit entry point.  Run:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="AI Meeting Intelligence Agent", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="expanded")

import api_client as api  # noqa: E402
import theme  # noqa: E402
from common import guard  # noqa: E402

theme.inject_css()
theme.sidebar_brand()


def login_screen(cfg: dict) -> None:
    theme.hero("Welcome back", "Sign in to access your private meeting knowledge base.", "Secure workspace")
    left, mid, right = st.columns([1, 1.4, 1])
    with mid:
        if cfg.get("allow_registration"):
            tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        else:
            (tab_in,) = st.tabs(["Sign in"])
            tab_up = None
        with tab_in:
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary", width="stretch"):
                    res = guard(lambda: api.login(u, p))
                    if res:
                        st.session_state["token"], st.session_state["user"] = res["access_token"], res["user"]
                        st.rerun()
        if tab_up is not None:
            with tab_up:
                with st.form("register"):
                    u = st.text_input("Choose a username")
                    e = st.text_input("Email (optional)")
                    p = st.text_input("Password (min 8 characters)", type="password")
                    if st.form_submit_button("Create account", type="primary", width="stretch"):
                        res = guard(lambda: api.register(u, p, e))
                        if res:
                            st.session_state["token"], st.session_state["user"] = res["access_token"], res["user"]
                            st.rerun()


# --------------------------------------------------------------- backend gate
try:
    health = api.health()
except api.ApiError as exc:
    theme.hero("Backend not reachable", "The Streamlit UI needs the FastAPI service to be running.", "Setup")
    st.error(exc.message)
    st.code("cd backend\nuvicorn app.main:app --reload --port 8000", language="bash")
    st.caption("Or start everything at once with `python run.py` from the project root.")
    if st.button("Retry connection", type="primary"):
        st.rerun()
    st.stop()

cfg = guard(api.auth_config, {"auth_enabled": False}, show=False)
if cfg.get("auth_enabled") and not st.session_state.get("token"):
    login_screen(cfg)
    st.stop()

if cfg.get("auth_enabled"):
    user = st.session_state.get("user") or {}
    st.sidebar.markdown(f'<div class="mi-note">Signed in as <b>{theme.esc(user.get("username", "user"))}</b></div>',
                        unsafe_allow_html=True)
    if st.sidebar.button("Sign out", width="stretch"):
        for k in ("token", "user", "selected_meeting"):
            st.session_state.pop(k, None)
        st.rerun()

pages = [
    st.Page("pages/dashboard.py", title="Dashboard", icon="🏠", default=True),
    st.Page("pages/upload.py", title="New meeting", icon="⬆️"),
    st.Page("pages/meeting.py", title="Meeting review", icon="📝"),
    st.Page("pages/chat.py", title="Ask your meetings", icon="💬"),
    st.Page("pages/tasks.py", title="Action board", icon="📌"),
    st.Page("pages/system.py", title="System", icon="⚙️"),
]
nav = st.navigation(pages)
if health.get("status") != "ok":
    st.sidebar.warning("Service degraded — see System page.")
nav.run()
