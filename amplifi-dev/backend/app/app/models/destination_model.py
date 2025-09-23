from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlalchemy import Column, String
from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.databricks_info import DatabricksInfo
    from app.models.organization_model import Organization
    from app.models.pg_vector_info import PgVectorInfo
    from app.models.workflow_model import Workflow


class Destination(BaseUUIDModel, table=True):
    __tablename__ = "destinations"

    organization_id: UUID = Field(foreign_key="organizations.id")
    name: str = Field(..., nullable=False)
    is_active: bool = Field(default=False)
    description: Optional[str] = Field(default=None)

    # New destination_type column without Enum
    destination_type: Optional[str] = Field(sa_column=Column(String, nullable=False))

    destination_pgvector: Optional["PgVectorInfo"] = Relationship(
        back_populates="destination", sa_relationship_kwargs={"lazy": "select"}
    )

    destination_databricks: Optional["DatabricksInfo"] = Relationship(
        back_populates="destination", sa_relationship_kwargs={"lazy": "select"}
    )

    organization: Optional["Organization"] = Relationship(back_populates="destinations")
    workflows: List["Workflow"] = Relationship(back_populates="destination")
