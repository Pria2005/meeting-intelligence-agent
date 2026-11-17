from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db import models as m
from app.db.database import get_db
from app.schemas import api as schemas
from app.services.rag import answer_question

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=schemas.ChatResponse,
             summary="Ask a question; answers are grounded in retrieved transcript evidence")
def chat(body: schemas.ChatRequest, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    return answer_question(db, user, body)
