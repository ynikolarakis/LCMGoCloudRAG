from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.models import User
from app.pipelines.query import query_documents
from app.schemas.query import Citation, QueryRequest, QueryResponseSchema

logger = structlog.get_logger()
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponseSchema)
async def submit_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
) -> QueryResponseSchema:
    """Submit a question to the RAG pipeline.

    Embeds the question, retrieves relevant chunks from Qdrant,
    and generates an answer with citations using the LLM.

    Args:
        request: QueryRequest containing the question and optional client_id.
        current_user: The authenticated user, injected by FastAPI.

    Returns:
        QueryResponseSchema with the answer, citations, model used, and latency.
    """
    result = query_documents(
        question=request.question,
        client_id=current_user.client_id,
    )

    citations = [
        Citation(
            source=c["source"],
            page=c.get("page"),
            content_preview=c.get("content_preview", ""),
        )
        for c in result["citations"]
    ]

    logger.info(
        "query_response_built",
        question_length=len(request.question),
        citation_count=len(citations),
        latency_ms=result["latency_ms"],
        user_id=str(current_user.id),
    )

    return QueryResponseSchema(
        answer=result["answer"],
        citations=citations,
        model_used=result["model_used"],
        latency_ms=result["latency_ms"],
    )
