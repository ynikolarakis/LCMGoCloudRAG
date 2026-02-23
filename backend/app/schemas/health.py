from __future__ import annotations

from pydantic import BaseModel


class ServiceStatus(BaseModel):
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: ServiceStatus
    qdrant: ServiceStatus
    redis: ServiceStatus
    llm: ServiceStatus
