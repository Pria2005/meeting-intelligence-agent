"""Corrections to extracted items (PRD FR-11) and cross-meeting task listing."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import NotFoundError
from app.db import models as m
from app.db.database import get_db
from app.schemas import api as schemas
from app.services import repository as repo

router = APIRouter(tags=["review"])


def _owned(db: Session, user: m.User, model, item_id: int):
    item = db.scalar(select(model).join(m.Meeting, m.Meeting.id == model.meeting_id)
                     .where(model.id == item_id, m.Meeting.user_id == user.id))
    if item is None:
        raise NotFoundError("Item not found.")
    return item


def _apply(item, data: dict) -> None:
    for key, value in data.items():
        if isinstance(value, str):
            value = value.strip() or None if key in {"owner", "due_date"} else value.strip()
        setattr(item, key, value)
    item.user_edited = True
    # A human touching the item resolves the review flag unless they set it explicitly.
    if "needs_review" not in data:
        item.needs_review = False


@router.get("/action-items", response_model=List[schemas.ActionItemOutAPI], summary="List action items across meetings")
def list_action_items(status: Optional[str] = Query(default=None, pattern="^(open|in_progress|done|dropped)$"),
                      owner: Optional[str] = Query(default=None, max_length=120),
                      meeting_id: Optional[str] = Query(default=None),
                      db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    stmt = (select(m.ActionItem).join(m.Meeting, m.Meeting.id == m.ActionItem.meeting_id)
            .where(m.Meeting.user_id == user.id).order_by(m.Meeting.created_at.desc(), m.ActionItem.id))
    if status:
        stmt = stmt.where(m.ActionItem.status == status)
    if owner:
        stmt = stmt.where(m.ActionItem.owner.ilike(f"%{owner}%"))
    if meeting_id:
        stmt = stmt.where(m.ActionItem.meeting_id == meeting_id)
    return [repo.action_out(db, a) for a in db.scalars(stmt).all()]


@router.patch("/action-items/{item_id}", response_model=schemas.ActionItemOutAPI, summary="Update an action item")
def update_action_item(item_id: int, body: schemas.ActionItemUpdate, db: Session = Depends(get_db),
                       user: m.User = Depends(get_current_user)):
    item = _owned(db, user, m.ActionItem, item_id)
    _apply(item, body.model_dump(exclude_unset=True))
    db.flush()
    repo.refresh_review_flag(db, db.get(m.Meeting, item.meeting_id))
    db.commit()
    return repo.action_out(db, item)


@router.patch("/decisions/{item_id}", response_model=schemas.DecisionOutAPI, summary="Update a decision")
def update_decision(item_id: int, body: schemas.DecisionUpdate, db: Session = Depends(get_db),
                    user: m.User = Depends(get_current_user)):
    item = _owned(db, user, m.Decision, item_id)
    _apply(item, body.model_dump(exclude_unset=True))
    db.flush()
    repo.refresh_review_flag(db, db.get(m.Meeting, item.meeting_id))
    db.commit()
    return repo.decision_out(db, item)


@router.patch("/risks/{item_id}", response_model=schemas.RiskOutAPI, summary="Update a risk / open question")
def update_risk(item_id: int, body: schemas.RiskUpdate, db: Session = Depends(get_db),
                user: m.User = Depends(get_current_user)):
    item = _owned(db, user, m.Risk, item_id)
    _apply(item, body.model_dump(exclude_unset=True))
    db.flush()
    repo.refresh_review_flag(db, db.get(m.Meeting, item.meeting_id))
    db.commit()
    return repo.risk_out(db, item)
