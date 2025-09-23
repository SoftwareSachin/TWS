from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from sqlmodel import JSON, Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.document_model import Document
    from app.models.file_ingestion_model import FileIngestion
    from app.models.file_model import File


class SplitFileStatusType(PyEnum):
    Processing = "Processing"
    Failed = "Failed"
    Success = "Success"
    Not_Started = "Not_Started"
    Exception = "Exception"


class FileSplit(BaseUUIDModel, table=True):
    """Model for tracking file splits created during the ingestion process."""

    __tablename__ = "file_splits"

    # The original file ID that this split belongs to
    original_file_id: UUID = Field(foreign_key="files.id", nullable=False)

    # The dataset ID this split belongs to (for dataset isolation)
    dataset_id: UUID = Field(foreign_key="datasets.id", nullable=False)

    # The split index (0-based) to maintain order
    split_index: int = Field(nullable=False)

    # The total number of splits for the original file
    total_splits: int = Field(nullable=False)

    # Path to the split file on disk
    split_file_path: str = Field(nullable=False)

    # Size of the split in bytes
    size: int = Field(nullable=False)

    # Estimated token count for this split
    token_count: int = Field(nullable=False)

    # Status of the split's ingestion
    status: SplitFileStatusType = Field(
        default=SplitFileStatusType.Not_Started, nullable=False
    )

    # Optional ingestion task ID for tracking
    task_id: Optional[str] = Field(default=None)

    # Timestamp when ingestion finished
    finished_at: datetime | None = Field(default=None)

    # Original file ingestion ID that created this split
    file_ingestion_id: UUID = Field(foreign_key="file_ingestion.id", nullable=True)

    document_id: Optional[UUID] = Field(
        foreign_key="documents.id", nullable=True
    )  # Reference to the document if applicable (for ingestion version 2.0)

    # Hash of the configuration used to create this split
    config_hash: Optional[str] = Field(default=None, nullable=True)

    # Metadata for storing split-specific information (like image counts)
    split_metadata: Optional[Dict[str, Any]] = Field(
        default=None, sa_type=JSON, nullable=True
    )

    # Relationships
    original_file: Optional["File"] = Relationship(back_populates="file_splits")
    file_ingestion: Optional["FileIngestion"] = Relationship(
        back_populates="file_splits"
    )
    # Optional relationship to the document if applicable
    document: Optional["Document"] = Relationship(back_populates="file_splits")
