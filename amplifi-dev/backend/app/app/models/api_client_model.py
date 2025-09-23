from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.organization_model import Organization


class ApiClientBase(SQLModel):
    """Base model for API client data"""

    client_id: str = Field(
        unique=True, index=True, description="Unique client identifier"
    )
    name: str = Field(description="Human-readable name for the API client")
    description: str | None = Field(
        default=None, description="Optional description of the client"
    )
    organization_id: UUID = Field(
        foreign_key="organizations.id",
        description="Organization this client belongs to",
    )
    expires_at: datetime | None = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=182),
        sa_column=Column(DateTime(timezone=True)),
        description="Expiration date for this client",
    )


class ApiClient(BaseUUIDModel, ApiClientBase, table=True):
    """API Client model for storing client credentials"""

    __tablename__ = "api_clients"

    # Relationships
    organization: "Organization" = Relationship(
        back_populates="api_clients", sa_relationship_kwargs={"lazy": "joined"}
    )

    # Metadata
    last_used_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="Last time this client was used",
    )

    class Config:
        from_attributes = True
