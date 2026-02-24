from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_v1_router
from app.clients import close_clients
from app.config import settings
from app.logging_config import setup_logging
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.rate_limit import limiter
from app.middleware.security_headers import SecurityHeadersMiddleware

setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks.

    On startup in dev mode, seeds the database with the dev admin user so that
    all protected endpoints are accessible without a real Keycloak instance.

    Args:
        application: The FastAPI application instance (unused directly but
            required by the lifespan protocol).
    """
    if settings.ENVIRONMENT == "dev":
        from app.auth import seed_dev_user
        from app.database import async_session_factory

        async with async_session_factory() as session:
            await seed_dev_user(session)

    yield
    await close_clients()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI instance with middleware and routers applied.
    """
    application = FastAPI(
        title="LCM DocIntel",
        description="Enterprise Multilingual RAG Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    Instrumentator().instrument(application).expose(application, endpoint="/metrics")

    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware (order matters — last added is first executed)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(RequestLoggingMiddleware)

    # Routers
    application.include_router(api_v1_router)

    return application


app = create_app()
