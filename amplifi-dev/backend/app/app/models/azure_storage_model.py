from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.source_model import Source


class AzureStorage(BaseUUIDModel, table=True):
    __tablename__ = "azure_storage"

    source_id: UUID = Field(foreign_key="sources.id", primary_key=True)
    container_name: str = Field(..., nullable=False)
    sas_url: str = Field(..., nullable=False)

    # Auto-detection settings
    auto_detection_enabled: Optional[bool] = Field(default=False, nullable=True)
    last_monitored: Optional[datetime] = Field(default=None, nullable=True)
    monitoring_frequency_minutes: Optional[int] = Field(default=5, nullable=True)

    # Re-ingestion setting
    re_ingest_updated_blobs: Optional[bool] = Field(default=False, nullable=True)

    source: "Source" = Relationship(back_populates="azure_storage")
