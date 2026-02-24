from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.admin import router as admin_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router
from app.api.v1.ws import router as ws_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router, tags=["health"])
api_v1_router.include_router(documents_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(conversations_router)
api_v1_router.include_router(admin_router)
api_v1_router.include_router(ws_router)
