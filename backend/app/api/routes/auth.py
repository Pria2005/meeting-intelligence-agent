from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.errors import AuthError, ConflictError, ForbiddenError
from app.core.security import create_access_token, hash_password, verify_password
from app.db import models as m
from app.db.database import get_db
from app.schemas import api as schemas

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/config", response_model=schemas.AuthConfig)
def auth_config() -> schemas.AuthConfig:
    s = get_settings()
    return schemas.AuthConfig(auth_enabled=s.auth_enabled, allow_registration=s.allow_registration)


@router.post("/register", response_model=schemas.TokenResponse, status_code=201)
def register(body: schemas.RegisterRequest, db: Session = Depends(get_db)) -> schemas.TokenResponse:
    s = get_settings()
    if not s.auth_enabled:
        raise ForbiddenError("Authentication is disabled (single-user mode).")
    if not s.allow_registration:
        raise ForbiddenError("Registration is disabled on this server.")
    if db.scalar(select(m.User).where(m.User.username == body.username)):
        raise ConflictError("That username is already taken.")
    if body.email and db.scalar(select(m.User).where(m.User.email == body.email)):
        raise ConflictError("That email is already registered.")
    user = m.User(username=body.username, email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return schemas.TokenResponse(access_token=create_access_token(user.id), user=schemas.UserOut.model_validate(user))


@router.post("/login", response_model=schemas.TokenResponse)
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)) -> schemas.TokenResponse:
    if not get_settings().auth_enabled:
        raise ForbiddenError("Authentication is disabled (single-user mode).")
    user = db.scalar(select(m.User).where(m.User.username == body.username))
    if user is None or not user.password_hash or not verify_password(body.password, user.password_hash):
        raise AuthError("Incorrect username or password.")
    return schemas.TokenResponse(access_token=create_access_token(user.id), user=schemas.UserOut.model_validate(user))


@router.get("/me", response_model=schemas.UserOut)
def me(user: m.User = Depends(get_current_user)) -> schemas.UserOut:
    return schemas.UserOut.model_validate(user)
