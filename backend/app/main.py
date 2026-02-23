from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.config import settings
from app.logging_config import setup_logging
from app.middleware.logging import RequestLoggingMiddleware

setup_logging(settings.LOG_LEVEL)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="LCM DocIntel",
        description="Enterprise Multilingual RAG Platform",
        version="0.1.0",
    )

    # Middleware (order matters — last added is first executed)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestLoggingMiddleware)

    # Routers
    application.include_router(api_v1_router)

    return application


app = create_app()
