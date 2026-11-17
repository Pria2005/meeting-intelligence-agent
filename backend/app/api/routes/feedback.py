from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db import models as m
from app.db.database import get_db
from app.schemas import api as schemas
from app.services import repository as repo

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=schemas.FeedbackOut, status_code=201,
             summary="Rate the quality of a generated item")
def create_feedback(body: schemas.FeedbackCreate, db: Session = Depends(get_db),
                    user: m.User = Depends(get_current_user)):
    if body.meeting_id:
        repo.get_meeting(db, user, body.meeting_id)  # authorisation check
    fb = m.Feedback(user_id=user.id, meeting_id=body.meeting_id, item_type=body.item_type, item_id=body.item_id,
                    rating=body.rating, comment=body.comment)
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return schemas.FeedbackOut.model_validate(fb)
