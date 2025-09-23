from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.utils.uuid6 import uuid7

if TYPE_CHECKING:
    from app.models import User


class EntityType(Enum):
    Workspace = "Workspace"
    Dataset = "Dataset"
    Source_Connector = "Source Connector"
    File = "File"
    Destination = "Destination"
    Workflow = "Workflow"


class OperationType(Enum):
    Create = "Create"
    Update = "Update"
    Delete = "Delete"


class AuditLog(SQLModel, table=True):
    id: UUID = Field(
        default_factory=uuid7,
        primary_key=True,
        index=True,
        nullable=False,
    )
    operation: OperationType = Field(nullable=False)
    entity: EntityType = Field(nullable=False)
    entity_id: UUID = Field(nullable=False)
    entity_name: str = Field(nullable=False)
    user_id: UUID = Field(foreign_key="User.id", nullable=False)
    user_name: str = Field(nullable=False)
    logged_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    user: Optional["User"] = Relationship(
        back_populates="audit_logs", sa_relationship_kwargs={"lazy": "joined"}
    )
