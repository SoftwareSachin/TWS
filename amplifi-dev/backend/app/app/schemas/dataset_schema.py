from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, field_validator

from app.schemas.rag_generation_schema import RagContext


class datasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    file_ids: List[UUID] = []

    class config:
        from_attribute: True


class IDatasetCreate(datasetBase):
    source_id: Optional[UUID] = None

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty or just whitespace")
        return v


class IDatasetRead(datasetBase):
    id: UUID
    source_id: Optional[UUID] = None
    source_type: Optional[str] = None
    r2r_collection_id: Optional[UUID] = None
    knowledge_graph: Optional[bool] = False
    graph_build_phase: Optional[str] = None
    graph_build_requested_at: Optional[datetime] = None
    graph_build_completed_at: Optional[datetime] = None
    last_extraction_check_at: Optional[datetime] = None
    graph_status: Optional[bool] = None

    class Config:
        from_attributes = True


class IDatasetUpdate(datasetBase):
    source_id: Optional[UUID] = None

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty or just whitespace")
        return v

    class Config:
        from_attributes = True


class IChunkInfo(RagContext):
    text: str
    document_chunk_id: Optional[UUID] = None  # Is this just chunk id?
    chunk_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    dataset_id: Optional[UUID] = None
    ingestion_id: Optional[UUID] = None
    version: Optional[str] = None
    chunk_order: Optional[int] = None
    vector: Optional[List[float]] = None


class IChunkInfoRead(BaseModel):
    file_id: UUID
    filename: Optional[str] = None
    chunks: List[IChunkInfo]
    total_chunks: Optional[int] = None


class DatasetFileTypeGroup(BaseModel):
    text_datasets: List[UUID]
    image_datasets: List[UUID]
    mixed_datasets: List[UUID]
