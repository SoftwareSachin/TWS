from typing import Sequence
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.crud.base_crud import CRUDBase
from app.models import Document, DocumentChunk
from app.models.document_model import DocumentTypeEnum
from app.models.image_media_model import ImageMedia
from app.schemas.image_media_schema import IImageMediaCreate, IImageMediaUpdate


class CRUDImageMedia(CRUDBase[ImageMedia, IImageMediaCreate, IImageMediaUpdate]):

    async def get_documents_by_dataset_ids(
        self, *, dataset_ids: list[UUID], db_session: AsyncSession | None = None
    ) -> Sequence[Document]:
        db_session = db_session or super().get_db().session
        # statement = select(Document).where(
        #     Document.dataset_id.in_(dataset_ids),
        #     Document.document_type == DocumentTypeEnum.Image,
        # )
        statement = select(Document).where(
            Document.dataset_id.in_(dataset_ids),
            or_(
                Document.document_type == DocumentTypeEnum.Image,
                Document.document_type == DocumentTypeEnum.Audio,
            ),
        )
        result = await db_session.execute(statement)
        return result.scalars().all()

    async def get_chunks_by_document_id(
        self, *, document_id: UUID, db_session: AsyncSession | None = None
    ) -> Sequence[DocumentChunk]:
        db_session = db_session or super().get_db().session
        statement = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
        result = await db_session.execute(statement)
        return result.scalars().all()


image = CRUDImageMedia(ImageMedia)
