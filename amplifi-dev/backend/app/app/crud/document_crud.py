from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.schemas.document_schema import (
    DocumentChunkCreate,
    DocumentChunkUpdate,
    DocumentCreate,
    DocumentUpdate,
)


class CRUDDocument:
    """CRUD operations for Document and DocumentChunk models"""

    async def create_document(
        self, db: AsyncSession, *, obj_in: DocumentCreate
    ) -> Document:
        """Create a new document"""
        db_obj = Document(
            file_id=obj_in.file_id,
            dataset_id=obj_in.dataset_id,
            document_type=obj_in.document_type,
            file_path=obj_in.file_path,
            description=obj_in.description,
            metadata=obj_in.metadata,
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update_document(
        self,
        db: AsyncSession,
        *,
        db_obj: Document,
        obj_in: Union[DocumentUpdate, Dict[str, Any]],
    ) -> Document:
        """Update a document"""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        for field in update_data:
            setattr(db_obj, field, update_data[field])

        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def get_document(
        self, db: AsyncSession, document_id: UUID
    ) -> Optional[Document]:
        """Get a document by ID"""
        query = select(Document).where(
            Document.id == document_id, Document.deleted_at.is_(None)
        )
        result = await db.execute(query)
        return result.scalars().first()

    async def get_document_by_file_id(
        self, db: AsyncSession, file_id: UUID, document_type: DocumentTypeEnum = None
    ) -> Optional[Document]:
        """Get a document by file ID, optionally filtering by document type"""
        query = select(Document).where(
            Document.file_id == file_id, Document.deleted_at.is_(None)
        )
        if document_type:
            query = query.where(Document.document_type == document_type)
        result = await db.execute(query)
        return result.scalars().first()

    async def get_documents_by_dataset(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        document_type: Optional[DocumentTypeEnum] = None,
        status: Optional[DocumentProcessingStatusEnum] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Document]:
        """Get all documents for a dataset, with optional filters"""
        query = select(Document).where(
            Document.dataset_id == dataset_id, Document.deleted_at.is_(None)
        )

        if document_type:
            query = query.where(Document.document_type == document_type)

        if status:
            query = query.where(Document.processing_status == status)

        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_document_ids_by_dataset(
        self,
        db: AsyncSession,
        dataset_id: UUID,
    ) -> list[UUID]:
        query = select(Document.id).where(
            Document.dataset_id == dataset_id, Document.deleted_at.is_(None)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def delete_document(self, db: AsyncSession, *, document_id: UUID) -> bool:
        """Soft delete a document and its chunks"""
        # First soft-delete associated chunks
        await self.delete_chunks_by_document(db, document_id=document_id)

        # Then soft-delete the document
        now = datetime.utcnow()
        query = (
            update(Document)
            .where(Document.id == document_id, Document.deleted_at.is_(None))
            .values(deleted_at=now)
        )

        result = await db.execute(query)
        await db.commit()
        return result.rowcount > 0

    # Document Chunk CRUD operations

    async def create_document_chunk(
        self, db: AsyncSession, *, obj_in: DocumentChunkCreate
    ) -> DocumentChunk:
        """Create a new document chunk"""
        db_obj = DocumentChunk(
            document_id=obj_in.document_id,
            chunk_type=obj_in.chunk_type,
            chunk_text=obj_in.chunk_text,
            metadata=obj_in.metadata,
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def create_document_chunks(
        self, db: AsyncSession, *, objs_in: List[DocumentChunkCreate]
    ) -> List[DocumentChunk]:
        """Create multiple document chunks in a batch"""
        db_objs = [
            DocumentChunk(
                document_id=obj_in.document_id,
                chunk_type=obj_in.chunk_type,
                chunk_text=obj_in.chunk_text,
                metadata=obj_in.metadata,
            )
            for obj_in in objs_in
        ]
        db.add_all(db_objs)
        await db.commit()

        # Return the created objects
        for db_obj in db_objs:
            await db.refresh(db_obj)

        return db_objs

    async def update_document_chunk(
        self,
        db: AsyncSession,
        *,
        db_obj: DocumentChunk,
        obj_in: Union[DocumentChunkUpdate, Dict[str, Any]],
    ) -> DocumentChunk:
        """Update a document chunk"""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        for field in update_data:
            setattr(db_obj, field, update_data[field])

        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def get_document_chunk(
        self, db: AsyncSession, chunk_id: UUID
    ) -> Optional[DocumentChunk]:
        """Get a document chunk by ID"""
        query = select(DocumentChunk).where(
            DocumentChunk.id == chunk_id, DocumentChunk.deleted_at.is_(None)
        )
        result = await db.execute(query)
        return result.scalars().first()

    async def get_chunks_by_document(
        self,
        db: AsyncSession,
        document_id: UUID,
        chunk_type: Optional[ChunkTypeEnum] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[DocumentChunk]:
        """Get all chunks for a document, optionally filtered by type"""
        query = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id, DocumentChunk.deleted_at.is_(None)
        )

        if chunk_type:
            query = query.where(DocumentChunk.chunk_type == chunk_type)

        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_chunks_text_by_document(
        self,
        db: AsyncSession,
        document_id: UUID,
    ) -> list[str]:
        query = select(DocumentChunk.chunk_text).where(
            DocumentChunk.document_id == document_id, DocumentChunk.deleted_at.is_(None)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def delete_document_chunk(self, db: AsyncSession, *, chunk_id: UUID) -> bool:
        """Soft delete a document chunk"""
        now = datetime.utcnow()
        query = (
            update(DocumentChunk)
            .where(DocumentChunk.id == chunk_id, DocumentChunk.deleted_at.is_(None))
            .values(deleted_at=now)
        )

        result = await db.execute(query)
        await db.commit()
        return result.rowcount > 0

    async def delete_chunks_by_document(
        self, db: AsyncSession, *, document_id: UUID
    ) -> bool:
        """Soft delete all chunks for a document"""
        now = datetime.utcnow()
        query = (
            update(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )

        result = await db.execute(query)
        await db.commit()
        return result.rowcount > 0

    def get_document_ids_by_dataset_sync(
        self,
        db: Session,
        dataset_id: UUID,
    ) -> list[UUID]:
        """Sync version of get_document_ids_by_dataset"""
        query = select(Document.id).where(
            Document.dataset_id == dataset_id, Document.deleted_at.is_(None)
        )
        result = db.execute(query)
        return list(result.scalars().all())

    def get_chunks_text_by_document_sync(
        self,
        db: Session,
        document_id: UUID,
    ) -> list[str]:
        """Sync version of get_chunks_text_by_document"""
        query = select(DocumentChunk.chunk_text).where(
            DocumentChunk.document_id == document_id, DocumentChunk.deleted_at.is_(None)
        )
        result = db.execute(query)
        return list(result.scalars().all())


document = CRUDDocument()
