from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.audit import write_audit_log
from app.auth import DEV_USER_ID
from app.config import settings
from app.database import async_session_factory
from app.guardrails import check_faithfulness, scan_input
from app.models import Query, QueryResponse
from app.models.base import AuditAction
from app.pipelines.query import retrieve_context
from app.pipelines.streaming import stream_llm_response

logger = structlog.get_logger()

router = APIRouter(tags=["websocket"])


async def _persist_query_response(
    question: str,
    answer: str,
    citations_json: list[dict[str, Any]],
    latency_ms: int,
    model_used: str,
    faithfulness_score: float,
    client_id: str,
) -> None:
    """Persist a Query and QueryResponse row to the database.

    Opens its own session and commits independently of the WebSocket handler
    so that DB errors do not disrupt the already-sent WebSocket messages.

    Args:
        question: The user's original question text.
        answer: The final answer string sent to the client.
        citations_json: List of citation dicts to store as JSONB.
        latency_ms: End-to-end pipeline latency in milliseconds.
        model_used: Name/identifier of the LLM model that generated the answer.
        faithfulness_score: HHEM faithfulness score for the response.
        client_id: Tenant identifier for the query row.
    """
    async with async_session_factory() as session:
        query_row = Query(
            user_id=DEV_USER_ID,
            query_text=question,
            client_id=client_id,
        )
        session.add(query_row)
        await session.flush()

        response_row = QueryResponse(
            query_id=query_row.id,
            response_text=answer,
            citations=citations_json,
            latency_ms=latency_ms,
            model_used=model_used,
            faithfulness_score=faithfulness_score,
        )
        session.add(response_row)
        await session.flush()

        await session.commit()

    logger.info(
        "ws_query_persisted",
        latency_ms=latency_ms,
        citation_count=len(citations_json),
        client_id=client_id,
    )


async def _write_query_audit(
    action: AuditAction,
    details: dict[str, Any],
    client_id: str,
) -> None:
    """Write an audit log entry from a WebSocket handler context.

    Opens its own session and commits independently so that audit writes
    are isolated from the main WebSocket message flow.

    Args:
        action: The AuditAction enum value describing what occurred.
        details: Additional context dict stored as JSONB in the audit row.
        client_id: Tenant identifier for the audit entry.
    """
    async with async_session_factory() as session:
        await write_audit_log(
            session=session,
            user_id=DEV_USER_ID,
            action=action,
            resource_type="query",
            details=details,
            client_id=client_id,
        )
        await session.commit()


@router.websocket("/ws")
async def websocket_query(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming RAG query responses.

    Accepts JSON messages from the client and dispatches on the ``type`` field.
    Supported message types:

    * ``ping`` — heartbeat; responds with ``{"type": "pong"}``
    * ``query`` — run the RAG pipeline; streams back status, token, citations,
      and done messages

    Query flow:
        1. Run input guardrail (``scan_input``); send error and return if blocked
        2. Send ``{"type": "status", "status": "processing"}``
        3. Run ``retrieve_context`` in a thread (synchronous Haystack pipeline)
        4. Stream real tokens from Ollama via ``stream_llm_response`` async generator
        5. Run faithfulness check on the assembled answer; append a note if unfaithful
        6. Send ``{"type": "citations", "citations": [...]}``
        7. Send ``{"type": "done", "latency_ms": <int>}``
        8. Persist Query + QueryResponse rows (fire-and-forget)
        9. Write audit log entries (fire-and-forget)

    Args:
        websocket: The Starlette WebSocket connection.
    """
    await websocket.accept()
    logger.info("ws_connection_opened", client=str(websocket.client))

    try:
        while True:
            try:
                data: dict[str, Any] = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("ws_connection_closed", client=str(websocket.client))
                return

            msg_type: str = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "query":
                await _handle_query(websocket, data)
                continue

            # Unknown message type — ignore gracefully
            logger.warning("ws_unknown_message_type", msg_type=msg_type)

    except WebSocketDisconnect:
        logger.info("ws_connection_closed", client=str(websocket.client))
    except Exception as exc:
        logger.error("ws_unhandled_error", error=str(exc))
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": "Internal server error"})


async def _handle_query(websocket: WebSocket, data: dict[str, Any]) -> None:
    """Handle a single ``query`` message on the WebSocket.

    Runs retrieval, then streams real tokens from Ollama, then fires off
    persistence and audit tasks in the background.

    Args:
        websocket: The active WebSocket connection.
        data: The parsed JSON message dict containing at minimum ``question``.
    """
    import time

    question: str = data.get("question", "").strip()
    client_id: str = settings.CLIENT_ID

    if not question:
        await websocket.send_json({"type": "error", "detail": "question field is required"})
        return

    # 1. Input guardrail
    guardrail_result: dict[str, Any] = scan_input(question)
    if guardrail_result["blocked"]:
        reason: str = guardrail_result.get("reason") or "guardrail_blocked"
        logger.warning("ws_query_blocked", reason=reason, client_id=client_id)
        asyncio.ensure_future(
            _write_query_audit(
                action=AuditAction.GUARDRAIL_BLOCKED,
                details={"reason": reason, "risk_score": guardrail_result.get("risk_score", 0.0)},
                client_id=client_id,
            )
        )
        await websocket.send_json({"type": "error", "detail": reason})
        return

    # 2. Acknowledge
    await websocket.send_json({"type": "status", "status": "processing"})

    # 3. Retrieve context (synchronous pipeline in thread)
    start_time = time.perf_counter()
    retrieval_result: dict[str, Any] = await asyncio.to_thread(
        retrieve_context,
        question=question,
        client_id=client_id,
    )

    formatted_context: str = retrieval_result["formatted_context"]
    citations_raw: list[dict[str, Any]] = retrieval_result.get("citations", [])
    retrieved_docs = retrieval_result.get("documents", [])

    # 4. Stream real tokens from Ollama
    answer_parts: list[str] = []
    try:
        async for token in stream_llm_response(formatted_context, question):
            answer_parts.append(token)
            await websocket.send_json({"type": "token", "token": token})
    except Exception as exc:
        logger.error("ws_streaming_error", error=str(exc))
        if not answer_parts:
            await websocket.send_json({"type": "error", "detail": "Streaming failed"})
            return

    answer = "".join(answer_parts)
    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # 5. Faithfulness check on assembled answer
    faithfulness_score: float = 1.0
    retrieved_context = "\n\n".join(doc.content for doc in retrieved_docs if doc.content)
    if retrieved_context and answer:
        faithfulness_score, is_faithful = check_faithfulness(
            context=retrieved_context,
            response=answer,
        )
        if not is_faithful:
            await websocket.send_json(
                {
                    "type": "token",
                    "token": "\n\n[Note: Response could not be verified against source documents.]",
                }
            )

    # 6. Citations
    citations_json: list[dict[str, Any]] = [
        {
            "source": c.get("source", "unknown"),
            "page": c.get("page"),
            "content_preview": c.get("content_preview", ""),
        }
        for c in citations_raw
    ]
    await websocket.send_json({"type": "citations", "citations": citations_json})

    # 7. Done
    await websocket.send_json({"type": "done", "latency_ms": latency_ms})

    logger.info(
        "ws_query_complete",
        question_length=len(question),
        answer_length=len(answer),
        citation_count=len(citations_json),
        latency_ms=latency_ms,
        client_id=client_id,
    )

    # 8. Fire-and-forget: persist and audit
    asyncio.ensure_future(
        _persist_query_response(
            question=question,
            answer=answer,
            citations_json=citations_json,
            latency_ms=latency_ms,
            model_used=settings.LLM_MODEL,
            faithfulness_score=faithfulness_score,
            client_id=client_id,
        )
    )
    asyncio.ensure_future(
        _write_query_audit(
            action=AuditAction.QUERY_SUBMITTED,
            details={"query_text": question[:200]},
            client_id=client_id,
        )
    )
    asyncio.ensure_future(
        _write_query_audit(
            action=AuditAction.RESPONSE_GENERATED,
            details={"latency_ms": latency_ms, "citation_count": len(citations_json)},
            client_id=client_id,
        )
    )
