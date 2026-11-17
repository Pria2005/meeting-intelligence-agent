"""FastAPI application entry point.  Run:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, chat, feedback, items, meetings, system
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.db.database import init_db
from app.services import jobs

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level, json_logs=s.environment == "production")
    init_db()
    recovered = jobs.recover_stale_jobs()
    if recovered:
        logger.warning("Marked %d interrupted job(s) as failed after restart", recovered)
    logger.info("%s v%s started (auth=%s, llm=%s)", s.app_name, s.app_version, s.auth_enabled, s.resolved_llm_provider)
    yield
    jobs.shutdown()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.app_name, version=s.app_version, lifespan=lifespan,
        description="Meeting transcription, structured notes, and evidence-grounded Q&A.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    register_exception_handlers(app)
    for router in (system.router, auth.router, meetings.router, items.router, chat.router, feedback.router):
        app.include_router(router)
    return app


app = create_app()
