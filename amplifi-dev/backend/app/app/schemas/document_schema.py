from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.document_chunk_model import ChunkTypeEnum
from app.models.document_model import DocumentProcessingStatusEnum, DocumentTypeEnum


# Base schema for Document
class DocumentBase(BaseModel):
    file_id: UUID
    dataset_id: UUID
    document_type: DocumentTypeEnum
    file_path: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Schema for creating a Document
class DocumentCreate(DocumentBase):
    pass


# Schema for updating a Document
class DocumentUpdate(BaseModel):
    processing_status: Optional[DocumentProcessingStatusEnum] = None
    description: Optional[str] = None
    description_embedding: Optional[List[float]] = None
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


# Schema for reading a Document
class DocumentRead(DocumentBase):
    id: UUID
    processing_status: DocumentProcessingStatusEnum
    description_embedding: Optional[List[float]] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    task_id: Optional[str] = None
    error_message: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    ingestion_id: Optional[UUID] = None

    class Config:
        orm_mode = True


# Base schema for DocumentChunk
class DocumentChunkBase(BaseModel):
    document_id: UUID
    chunk_type: ChunkTypeEnum
    chunk_text: str
    metadata: Optional[Dict[str, Any]] = None


# Schema for creating a DocumentChunk
class DocumentChunkCreate(DocumentChunkBase):
    pass


# Schema for updating a DocumentChunk
class DocumentChunkUpdate(BaseModel):
    chunk_text: Optional[str] = None
    chunk_embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


# Schema for reading a DocumentChunk
class DocumentChunkRead(DocumentChunkBase):
    id: UUID
    chunk_embedding: Optional[List[float]] = None
    created_at: datetime
    deleted_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# Schema for a list of Documents with their chunks
class DocumentWithChunks(DocumentRead):
    chunks: List[DocumentChunkRead] = []


# Response schema for document processing status
class DocumentProcessingStatusResponse(BaseModel):
    document_id: UUID
    status: DocumentProcessingStatusEnum
    file_id: UUID
    document_type: DocumentTypeEnum
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# Schema for image ingestion request
class ImageIngestionRequest(BaseModel):
    file_ids: List[UUID]
    dataset_id: UUID
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Schema for image ingestion response
class ImageIngestionResponse(BaseModel):
    ingestion_id: UUID
    task_ids: List[str]
    file_count: int
