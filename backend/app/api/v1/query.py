from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.auth import get_current_user
from app.database import get_db_session
from app.exceptions import AppError
from app.guardrails import check_faithfulness, scan_input
from app.models import Query, QueryResponse, User
from app.models.base import AuditAction
from app.pipelines.query import query_documents
from app.schemas.query import Citation, QueryRequest, QueryResponseSchema

logger = structlog.get_logger()
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponseSchema)
async def submit_query(
    request_body: QueryRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> QueryResponseSchema:
    """Submit a question to the RAG pipeline.

    Creates a Query row, runs the pipeline, creates a QueryResponse row,
    and writes audit log entries.

    Args:
        request_body: QueryRequest containing the question and optional client_id.
        request: The raw FastAPI Request used to extract the client IP.
        current_user: The authenticated user, injected by FastAPI.
        session: The async database session, injected by FastAPI.

    Returns:
        QueryResponseSchema with the answer, citations, model used, and latency.
    """
    # 1. Create Query row
    query_row = Query(
        user_id=current_user.id,
        query_text=request_body.question,
        client_id=current_user.client_id,
    )
    session.add(query_row)
    await session.flush()

    # 2. Audit: query submitted
    client_ip = request.client.host if request.client else None
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.QUERY_SUBMITTED,
        resource_type="query",
        resource_id=query_row.id,
        details={"query_text": request_body.question[:200]},
        ip_address=client_ip,
        client_id=current_user.client_id,
    )

    # 3. Input guardrail check
    guardrail_result = scan_input(request_body.question)
    if guardrail_result["blocked"]:
        await write_audit_log(
            session=session,
            user_id=current_user.id,
            action=AuditAction.GUARDRAIL_BLOCKED,
            resource_type="query",
            resource_id=query_row.id,
            details={"reason": guardrail_result["reason"], "risk_score": guardrail_result["risk_score"]},
            ip_address=client_ip,
            client_id=current_user.client_id,
        )
        raise AppError(
            detail=f"Query blocked by guardrail: {guardrail_result['reason']}",
            status_code=400,
        )

    # 4. Run RAG pipeline
    result = query_documents(
        question=request_body.question,
        client_id=current_user.client_id,
    )

    # 5. Build citations
    citations = [
        Citation(
            source=c["source"],
            page=c.get("page"),
            content_preview=c.get("content_preview", ""),
        )
        for c in result["citations"]
    ]

    citations_json = [c.model_dump() for c in citations]

    # 5. Output guardrail: faithfulness check
    retrieved_context = "\n\n".join(doc.content for doc in result["retrieved_docs"] if doc.content)
    if retrieved_context:
        hhem_score, is_faithful = check_faithfulness(
            context=retrieved_context,
            response=result["answer"],
        )
        if not is_faithful:
            result["answer"] = "Cannot verify answer against available documents."
            citations = []
            citations_json = []

    # 6. Create QueryResponse row
    response_row = QueryResponse(
        query_id=query_row.id,
        response_text=result["answer"],
        citations=citations_json,
        latency_ms=result["latency_ms"],
        model_used=result["model_used"],
    )
    session.add(response_row)
    await session.flush()

    # 6. Audit: response generated
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.RESPONSE_GENERATED,
        resource_type="query_response",
        resource_id=response_row.id,
        details={"latency_ms": result["latency_ms"], "citation_count": len(citations)},
        ip_address=client_ip,
        client_id=current_user.client_id,
    )

    logger.info(
        "query_response_built",
        query_id=str(query_row.id),
        question_length=len(request_body.question),
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
