"""Thin HTTP client for the FastAPI backend with friendly error handling."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))


class ApiError(Exception):
    def __init__(self, message: str, status: int = 0, code: str = "error"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


class BackendDown(ApiError):
    pass


def _headers() -> Dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, path: str, *, raw: bool = False, timeout: Optional[float] = None, **kwargs) -> Any:
    try:
        resp = requests.request(method, f"{API_URL}{path}", headers=_headers(), timeout=timeout or TIMEOUT, **kwargs)
    except requests.ConnectionError as exc:
        raise BackendDown(f"Cannot reach the backend at {API_URL}. Is it running?") from exc
    except requests.Timeout as exc:
        raise ApiError("The backend took too long to respond.") from exc
    if resp.status_code == 401 and st.session_state.get("token"):
        st.session_state.pop("token", None)
        st.session_state.pop("user", None)
    if resp.status_code >= 400:
        try:
            err = resp.json().get("error", {})
            raise ApiError(err.get("message", resp.text[:200]), resp.status_code, err.get("code", "error"))
        except ValueError:
            raise ApiError(resp.text[:200] or f"HTTP {resp.status_code}", resp.status_code) from None
    if raw:
        return resp
    return resp.json() if resp.content else None


# ----------------------------------------------------------------- system
def info() -> Dict[str, Any]:
    return _request("GET", "/", timeout=5)


def health() -> Dict[str, Any]:
    return _request("GET", "/health", timeout=5)


def stats() -> Dict[str, Any]:
    return _request("GET", "/stats")


# ------------------------------------------------------------------- auth
def auth_config() -> Dict[str, Any]:
    return _request("GET", "/auth/config", timeout=5)


def login(username: str, password: str) -> Dict[str, Any]:
    return _request("POST", "/auth/login", json={"username": username, "password": password})


def register(username: str, password: str, email: Optional[str] = None) -> Dict[str, Any]:
    return _request("POST", "/auth/register", json={"username": username, "password": password,
                                                    "email": email or None})


# --------------------------------------------------------------- meetings
def list_meetings(status: Optional[str] = None, q: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {k: v for k, v in {"status": status, "q": q, "limit": 200}.items() if v}
    return _request("GET", "/meetings", params=params)


def get_meeting(mid: str) -> Dict[str, Any]:
    return _request("GET", f"/meetings/{mid}")


def upload(file_name: str, data: bytes, mime: str, title: str, auto_process: bool = True) -> Dict[str, Any]:
    return _request("POST", "/meetings/upload", files={"file": (file_name, data, mime)},
                    data={"title": title, "auto_process": str(auto_process).lower()}, timeout=600)


def create_demo() -> List[Dict[str, Any]]:
    return _request("POST", "/meetings/demo")


def start_processing(mid: str, force: bool = False) -> Dict[str, Any]:
    return _request("POST", f"/meetings/{mid}/transcribe", params={"force": str(force).lower()})


def rename_meeting(mid: str, title: str) -> Dict[str, Any]:
    return _request("PATCH", f"/meetings/{mid}", json={"title": title})


def delete_meeting(mid: str) -> Dict[str, Any]:
    return _request("DELETE", f"/meetings/{mid}")


def transcript(mid: str) -> Dict[str, Any]:
    return _request("GET", f"/meetings/{mid}/transcript")


def notes(mid: str) -> Dict[str, Any]:
    return _request("GET", f"/meetings/{mid}/summary")


def edit_summary(mid: str, **fields) -> Dict[str, Any]:
    return _request("PATCH", f"/meetings/{mid}/summary", json=fields)


def rename_speaker(mid: str, speaker_id: int, name: str) -> Dict[str, Any]:
    return _request("PATCH", f"/meetings/{mid}/speakers/{speaker_id}", json={"display_name": name})


def update_segment(mid: str, segment_id: int, **fields) -> Dict[str, Any]:
    return _request("PATCH", f"/meetings/{mid}/segments/{segment_id}", json=fields)


def audio_bytes(mid: str) -> bytes:
    return _request("GET", f"/meetings/{mid}/audio", raw=True, timeout=120).content


def export(mid: str, fmt: str, include_transcript: bool = True) -> bytes:
    return _request("GET", f"/meetings/{mid}/export", raw=True, timeout=60,
                    params={"format": fmt, "include_transcript": str(include_transcript).lower()}).content


# ------------------------------------------------------------ review items
def action_items(status: Optional[str] = None, owner: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {k: v for k, v in {"status": status, "owner": owner}.items() if v}
    return _request("GET", "/action-items", params=params)


def update_action_item(item_id: int, **fields) -> Dict[str, Any]:
    return _request("PATCH", f"/action-items/{item_id}", json=fields)


def update_decision(item_id: int, **fields) -> Dict[str, Any]:
    return _request("PATCH", f"/decisions/{item_id}", json=fields)


def update_risk(item_id: int, **fields) -> Dict[str, Any]:
    return _request("PATCH", f"/risks/{item_id}", json=fields)


# -------------------------------------------------------------- chat/other
def chat(question: str, meeting_ids: Optional[List[str]] = None, history: Optional[List[Dict[str, str]]] = None
         ) -> Dict[str, Any]:
    return _request("POST", "/chat", json={"question": question, "meeting_ids": meeting_ids or None,
                                           "history": history or []}, timeout=120)


def feedback(item_type: str, rating: int, meeting_id: Optional[str] = None, item_id: Optional[str] = None,
             comment: Optional[str] = None) -> Dict[str, Any]:
    return _request("POST", "/feedback", json={"item_type": item_type, "rating": rating, "meeting_id": meeting_id,
                                               "item_id": item_id, "comment": comment})
