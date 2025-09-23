from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, Relationship

from app.be_core.config import settings
from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.dataset_model import Dataset
    from app.models.document_chunk_model import DocumentChunk
    from app.models.file_model import File
    from app.models.file_split_model import FileSplit

# Default embedding dimension - adjust based on your model
EMBEDDING_DIM = settings.EMBEDDING_DIMENSIONS


class DocumentTypeEnum(PyEnum):
    """Types of documents supported"""

    Image = "image"
    Audio = "audio"
    Video = "video"
    PDF = "PDF"
    Pdf = "Pdf"  # Legacy value for backward compatibility - use PDF instead
    # This enum value exists because earlier database migrations created 'Pdf'
    # but later migrations standardized to 'PDF'. Both values are maintained
    # to ensure existing data remains valid while new code uses the standard 'PDF'.
    Markdown = "Markdown"
    HTML = "HTML"
    CSV = "CSV"
    XLSX = "XLSX"
    PPTX = "PPTX"
    DOCX = "DOCX"


class DocumentProcessingStatusEnum(PyEnum):
    """Status of document processing"""

    Queued = "queued"  # Document has been queued for processing
    Processing = "processing"  # Document is being processed
    Extracting = "extracting"  # Content extraction is in progress
    ExtractionCompleted = "extraction_completed"  # Content extraction has completed
    AnalysisCompleted = "analysis_completed"  # Document analysis has completed
    Success = "success"  # Document has been successfully processed
    Failed = "failed"  # Document processing has failed
    Exception = "exception"  # An exception occurred during processing
    Splitting = "splitting"  # Document is being split for processing
    Not_Started = "not_started"  # Document processing has not started


class Document(BaseUUIDModel, table=True):
    """Model for storing documents (images, audio, video) and their extracted metadata"""

    __tablename__ = "documents"

    # Core fields
    name: Optional[str] = Field(default=None)  # Name of the document
    file_id: UUID = Field(
        foreign_key="files.id", index=True
    )  # Reference to original file
    dataset_id: UUID = Field(foreign_key="datasets.id", nullable=False, index=True)
    document_type: DocumentTypeEnum = (
        Field()
    )  # Type of document (image, audio, video, pdf)
    processing_status: DocumentProcessingStatusEnum = Field(
        default=DocumentProcessingStatusEnum.Queued
    )

    # Extracted content
    description: Optional[str] = Field(default=None)  # Full content description
    # For images: AI-generated description of image contents
    # For audio: Full transcript of audio
    # For video: Full transcript or description of video content

    # Vectorized embeddings
    # Use pgvector type for description embedding
    description_embedding: Optional[List[float]] = Field(
        sa_column=Column(Vector(EMBEDDING_DIM)), default=None
    )

    # Processing metadata
    task_id: Optional[str] = Field(default=None)  # Celery task ID
    error_message: Optional[str] = Field(
        default=None
    )  # Error message if processing failed
    file_path: str = Field()  # Path to the original file
    file_size: Optional[int] = Field(default=None)  # Size of the file in bytes
    mime_type: Optional[str] = Field(default=None)  # MIME type of the file

    # Media-specific metadata stored as JSON
    document_metadata: Optional[Dict[str, Any]] = Field(
        default=None, sa_type=JSON
    )  # Media-specific metadata stored as JSON
    # For images: width, height, etc.
    # For audio: duration, sample_rate, etc.
    # For video: duration, frame_rate, resolution, etc.

    # Processing timestamps
    ingestion_id: Optional[UUID] = Field(default=None)  # ID of the ingestion batch
    processed_at: Optional[datetime] = Field(default=None)

    # Flag to indicate if this document was split
    is_split_document: Optional[bool] = Field(default=False, nullable=True)

    # Count of successfully ingested splits
    successful_splits_count: Optional[int] = Field(default=0, nullable=True)

    # Count of total splits
    total_splits_count: Optional[int] = Field(default=0, nullable=True)

    # Relationships
    chunks: List["DocumentChunk"] = Relationship(back_populates="document")
    dataset: "Dataset" = Relationship(back_populates="documents")
    file: Optional["File"] = Relationship(back_populates="documents")
    file_splits: Optional[List["FileSplit"]] = Relationship(
        back_populates="document"
    )  # Reference to file splits if applicable

    class Config:
        arbitrary_types_allowed = True


# Add indexes for document queries
Index(
    "idx_document_file_dataset",
    Document.file_id,
    Document.dataset_id,
    Document.deleted_at,
    postgresql_where=Document.deleted_at.is_(None),
)
Index("idx_document_file_id", Document.file_id)
Index("idx_document_dataset_id", Document.dataset_id)
Index("idx_document_processing_status", Document.processing_status)
