from typing import List
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base_crud import CRUDBase
from app.models import Document, DocumentChunk
from app.schemas.search_schema import BaseChunkMetadata, ImageSearchResult


class CRUDSearch(CRUDBase[DocumentChunk, None, None]):
    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        dataset_ids: List[UUID],
        file_ids: List[UUID] | None = None,
        db_session: AsyncSession | None = None,
    ) -> List[ImageSearchResult]:
        db_session = db_session or super().get_db().session

        # Build the base query
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.dataset_id.in_(dataset_ids))
        )

        # Add file_ids filter if provided for more targeted search
        if file_ids:
            stmt = stmt.where(Document.file_id.in_(file_ids))

        stmt = stmt.order_by(
            DocumentChunk.chunk_embedding.op("<->")(query_embedding)  # Vector search
        ).limit(top_k)

        result = await db_session.execute(stmt)
        matches = result.all()

        # Transform results into ImageSearchResult
        # match[0] is DocumentChunk, match[1] is Document
        search_results = [
            ImageSearchResult(
                text=match[0].chunk_text,
                file_id=match[1].file_id,
                file_path=match[1].file_path,
                description=(
                    match[0].chunk_metadata.get("description", None)
                    if match[0].chunk_metadata
                    else None
                ),
                dataset_id=match[1].dataset_id,
                search_score=float(np.dot(match[0].chunk_embedding, query_embedding)),
                mimetype=match[1].mime_type,
                chunk_metadata=BaseChunkMetadata(
                    match_type=match[0].chunk_type.name,
                    base64=None,
                    chunk_id=match[0].id,
                    table_html=(
                        match[0].chunk_metadata.get("table_html", None)
                        if match[0].chunk_metadata
                        else None
                    ),
                ),
            )
            for match in matches
        ]
        return search_results


search_crud = CRUDSearch(DocumentChunk)
