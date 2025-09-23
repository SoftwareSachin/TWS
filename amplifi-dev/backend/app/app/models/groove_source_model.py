# In app/models/groove_source_model.py

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.source_model import Source


class GrooveSource(SQLModel, table=True):
    __tablename__ = "groove_sources"  # Table name

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)

    # This is the foreign key linking back to the main 'sources' table
    source_id: UUID = Field(foreign_key="sources.id", unique=True, index=True)

    source_name: str = Field(nullable=False)
    # Store the path to the secret in Vault, NOT the key itself
    api_key_vault_path: str = Field(nullable=False)

    # Auto-detection settings (similar to Azure)
    auto_detection_enabled: Optional[bool] = Field(default=False, nullable=True)
    last_monitored: Optional[datetime] = Field(default=None, nullable=True)
    monitoring_frequency_minutes: Optional[int] = Field(default=30, nullable=True)

    # Groove-specific settings
    last_ticket_number: Optional[int] = Field(default=None, nullable=True)
    ticket_batch_size: Optional[int] = Field(default=10, nullable=True)
    re_ingest_updated_tickets: Optional[bool] = Field(default=False, nullable=True)

    # Define the one-to-one relationship back to the Source model
    source: "Source" = Relationship(back_populates="groove_source")

    deleted_at: Optional[datetime] = Field(default=None)
