from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    client_id: str = "default"


class Citation(BaseModel):
    source: str
    page: int | None = None
    content_preview: str


class QueryResponseSchema(BaseModel):
    answer: str
    citations: list[Citation]
    model_used: str
    latency_ms: int
