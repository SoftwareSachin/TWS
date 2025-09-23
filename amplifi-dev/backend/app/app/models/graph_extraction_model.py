from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional
from uuid import UUID

from sqlmodel import Field

from app.models.base_uuid_model import BaseUUIDModel


class ExtractionStatusEnum(str, PyEnum):
    PENDING = "PENDING"  # Initial state, extraction requested but not completed
    COMPLETED = "COMPLETED"  # Document extracted successfully
    FAILED = "FAILED"  # Extraction failed
    TIMED_OUT = "TIMED_OUT"  # Extraction timed out


class GraphBuildingPhase(str, PyEnum):
    NOT_STARTED = "NOT_STARTED"  # Process hasn't begun
    EXTRACTION = "EXTRACTION"  # Currently extracting entities from documents
    BUILD_INITIATED = "BUILD_INITIATED"  # Graph building has been initiated
    COMPLETED = "COMPLETED"  # Process marked complete (independent of R2R status)
    FAILED = "FAILED"  # Process failed


class GraphExtractionStatus(BaseUUIDModel, table=True):
    """
    Tracks the status of document extractions for knowledge graph building.
    This allows for precise progress tracking and resumption of failed builds.
    """

    __tablename__ = "graph_extraction_status"

    dataset_id: UUID = Field(foreign_key="datasets.id", index=True)
    collection_id: str = Field(index=True)
    document_id: str = Field(index=True)
    status: str = Field(default=ExtractionStatusEnum.PENDING)
    extraction_started_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_completed_at: Optional[datetime] = Field(default=None)
    retry_count: int = Field(default=0)
    error_message: Optional[str] = Field(default=None)
