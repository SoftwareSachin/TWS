from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, Relationship

from app.be_core.config import settings
from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.document_model import (  # Using renamed file, still containing Document class
        Document,
    )

# Use the same embedding dimension as in document_model.py
EMBEDDING_DIM = settings.EMBEDDING_DIMENSIONS


class ChunkTypeEnum(PyEnum):
    """Types of document chunks supported"""

    ImageDescription = "image_description"  # Overall description of the image
    ImageText = "image_text"  # Text extracted from image via OCR
    ImageObject = "image_object"  # Object detected in image
    AudioSegment = "audio_segment"  # Segment of audio transcript
    Speaker = "speaker"  # Speaker identification in audio/video
    VideoScene = "video_scene"  # Scene in video
    VideoSegment = (
        "video_segment"  # Time-based segment of video with transcript/caption
    )
    PDFText = "PDFText"
    PDFTable = "PDFTable"


class DocumentChunk(BaseUUIDModel, table=True):
    """Model for storing chunks of content from documents (images, audio, video)"""

    __tablename__ = "document_chunks"

    # Core fields
    document_id: UUID = Field(foreign_key="documents.id")
    chunk_type: ChunkTypeEnum = Field()
    chunk_text: str = Field()  # The actual content of the chunk

    # Split tracking
    split_id: Optional[UUID] = Field(
        default=None, index=True, foreign_key="file_splits.id"
    )  # Track which split created this chunk

    # Vectorized embeddings for search
    # Use pgvector type for chunk embedding
    chunk_embedding: Optional[List[float]] = Field(
        sa_column=Column(Vector(EMBEDDING_DIM)), default=None
    )

    # Metadata specific to chunk type
    chunk_metadata: Optional[Dict[str, Any]] = Field(
        default=None, sa_type=JSON
    )  # Chunk-specific metadata stored as JSON
    # Examples:
    # For ImageText: coordinates, confidence
    # For ImageObject: coordinates, confidence
    # For AudioSegment: start_time, end_time, confidence
    # For Speaker: speaker_id, speaker_segments [{"start_time: "00:00:10", "end_time": "00:00:20" }]
    # For VideoScene: scene_id, start_time, end_time

    # Relationship to parent document
    document: "Document" = Relationship(back_populates="chunks")

    class Config:
        arbitrary_types_allowed = True


Index(
    "ix_document_chunks_chunk_embedding_hnsw",
    DocumentChunk.__table__.c.chunk_embedding,
    postgresql_using="hnsw",
    postgresql_ops={"chunk_embedding": "vector_cosine_ops"},
    postgresql_with={"m": 16, "ef_construction": 100},
)
