from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.dataset_model import Dataset
    from app.models.file_model import File
    from app.models.file_split_model import FileSplit


class FileIngestionStatusType(PyEnum):
    Processing = "Processing"
    Failed = "Failed"
    Success = "Success"
    Not_Started = "Not_Started"
    Exception = "Exception"
    Splitting = "Splitting"


class FileIngestion(BaseUUIDModel, table=True):
    __tablename__ = "file_ingestion"

    name: Optional[str] = Field(default=None)
    ingestion_id: UUID = Field(nullable=False)
    task_id: Optional[str] = Field(default=None)
    file_id: UUID = Field(foreign_key="files.id", nullable=False)
    dataset_id: UUID = Field(foreign_key="datasets.id", nullable=False)
    status: FileIngestionStatusType = Field(
        default=FileIngestionStatusType.Not_Started, nullable=False
    )
    finished_at: datetime | None = Field(default=None)

    # Flag to indicate if this ingestion used file splitting
    is_split_ingestion: Optional[bool] = Field(default=False, nullable=True)

    # Count of successfully ingested splits
    successful_splits_count: Optional[int] = Field(default=0, nullable=True)

    # Count of total splits
    total_splits_count: Optional[int] = Field(default=0, nullable=True)

    # Relationships
    file: Optional["File"] = Relationship(back_populates="file_ingestion")
    dataset: Optional["Dataset"] = Relationship(back_populates="file_ingestion")
    file_splits: Optional[List["FileSplit"]] = Relationship(
        back_populates="file_ingestion"
    )
