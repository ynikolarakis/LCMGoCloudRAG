from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import structlog
from docling.document_converter import DocumentConverter
from haystack import Document as HaystackDocument
from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import DocumentChunk
from app.pipelines.chunking import chunk_text, count_tokens

logger = structlog.get_logger()


def _get_document_store() -> QdrantDocumentStore:
    """Create a QdrantDocumentStore instance."""
    return QdrantDocumentStore(
        url=settings.QDRANT_URL,
        index=settings.QDRANT_COLLECTION,
        embedding_dim=settings.EMBEDDING_DIMENSION,
        recreate_index=False,
        wait_result_from_api=True,
    )


def _build_ingestion_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    """Build the Haystack ingestion pipeline: embed -> write to Qdrant."""
    pipeline = Pipeline()

    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_token("ollama"),
        model=settings.EMBEDDING_MODEL,
        api_base_url=settings.EMBEDDING_BASE_URL,
        meta_fields_to_embed=[],
    )

    writer = DocumentWriter(document_store=document_store)

    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline


def parse_document(file_path: str | Path) -> tuple[str, int]:
    """Parse a document using Docling and return (text, page_count).

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (extracted_text, page_count).
    """
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    text = result.document.export_to_markdown()
    # Estimate page count from Docling metadata
    page_count = len(result.document.pages) if hasattr(result.document, "pages") else 1
    return text, page_count


def ingest_document(
    file_path: str | Path,
    filename: str,
    client_id: str = "default",
) -> list[HaystackDocument]:
    """Full ingestion: parse -> chunk -> embed -> store in Qdrant.

    Args:
        file_path: Path to the uploaded file.
        filename: Original filename for citation metadata.
        client_id: Client identifier for multi-tenancy.

    Returns:
        List of Haystack Documents that were indexed.
    """
    logger.info("ingestion_started", filename=filename, client_id=client_id)

    # 1. Parse document
    text, page_count = parse_document(file_path)
    logger.info("document_parsed", filename=filename, page_count=page_count, chars=len(text))

    # 2. Chunk text
    chunks = chunk_text(
        text,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        max_tokens=settings.CHUNK_MAX_TOKENS,
        min_tokens=settings.CHUNK_MIN_TOKENS,
        overlap_pct=settings.CHUNK_OVERLAP_PCT,
    )
    logger.info("document_chunked", filename=filename, num_chunks=len(chunks))

    # 3. Create Haystack Documents with metadata
    haystack_docs: list[HaystackDocument] = []
    for seq, chunk_content in enumerate(chunks):
        doc = HaystackDocument(
            id=str(uuid4()),
            content=chunk_content,
            meta={
                "source": filename,
                "page_num": seq + 1,  # Approximate; proper page mapping in Phase 2
                "chunk_seq": seq,
                "token_count": count_tokens(chunk_content),
                "client_id": client_id,
            },
        )
        haystack_docs.append(doc)

    # 4. Embed and write to Qdrant via Haystack pipeline
    store = _get_document_store()
    pipeline = _build_ingestion_pipeline(store)
    result = pipeline.run({"embedder": {"documents": haystack_docs}})
    written_count = result.get("writer", {}).get("documents_written", len(haystack_docs))
    logger.info("ingestion_complete", filename=filename, chunks_written=written_count)

    return haystack_docs


async def create_document_chunks(
    session: AsyncSession,
    doc_id: UUID,
    chunks: list[str],
    client_id: str = "default",
) -> list[DocumentChunk]:
    """Create DocumentChunk rows in the database for each text chunk.

    Args:
        session: Active async database session.
        doc_id: Parent Document UUID.
        chunks: List of chunk text strings.
        client_id: Tenant identifier.

    Returns:
        List of DocumentChunk ORM objects (not yet committed).
    """
    chunk_rows: list[DocumentChunk] = []
    for seq, chunk_content in enumerate(chunks):
        chunk_row = DocumentChunk(
            document_id=doc_id,
            chunk_seq=seq,
            content=chunk_content,
            token_count=count_tokens(chunk_content),
            client_id=client_id,
        )
        session.add(chunk_row)
        chunk_rows.append(chunk_row)

    await session.flush()
    return chunk_rows
