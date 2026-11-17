"""Shared FastAPI dependencies: database session and the authenticated user."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AuthError
from app.core.security import decode_access_token
from app.db import models as m
from app.db.database import get_db
from app.services import repository as repo

_bearer = HTTPBearer(auto_error=False)


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
                     db: Session = Depends(get_db)) -> m.User:
    """Single-user mode (AUTH_ENABLED=false) maps every request to the built-in local user.
    Multi-user mode requires a valid bearer token; every query is then scoped to that user."""
    if not get_settings().auth_enabled:
        return repo.get_or_create_default_user(db)
    if creds is None or not creds.credentials:
        raise AuthError("Authentication required.")
    user_id = decode_access_token(creds.credentials)
    user = db.get(m.User, user_id) if user_id else None
    if user is None:
        raise AuthError("Invalid or expired token.")
    return user
