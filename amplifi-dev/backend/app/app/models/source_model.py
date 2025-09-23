from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.aws_s3_storage_model import AWSS3Storage
from app.models.azure_fabric_model import AzureFabric
from app.models.azure_storage_model import AzureStorage
from app.models.base_uuid_model import BaseUUIDModel
from app.models.groove_source_model import GrooveSource  # <-- 1. ADD THIS IMPORT
from app.models.mysql_source_model import MySQLSource
from app.models.pg_vector_model import PGVector
from app.models.pull_status_model import SourcePullStatus

if TYPE_CHECKING:
    from app.models.workspace_model import Workspace


class Source(BaseUUIDModel, table=True):
    __tablename__ = "sources"

    workspace_id: UUID = Field(foreign_key="workspaces.id", nullable=False)
    source_type: str = Field(..., nullable=False)

    workspace: "Workspace" = Relationship(back_populates="sources")
    azure_storage: Optional["AzureStorage"] = Relationship(
        sa_relationship_kwargs={"uselist": False}, back_populates="source"
    )
    aws_s3_storage: Optional["AWSS3Storage"] = Relationship(
        sa_relationship_kwargs={"uselist": False}, back_populates="source"
    )
    azure_fabric: Optional["AzureFabric"] = Relationship(
        sa_relationship_kwargs={"uselist": False}, back_populates="source"
    )
    source_pgvector: Optional["PGVector"] = Relationship(
        sa_relationship_kwargs={"uselist": False}, back_populates="source"
    )
    source_mysql: Optional["MySQLSource"] = Relationship(
        sa_relationship_kwargs={"uselist": False}, back_populates="source"
    )
    groove_source: Optional["GrooveSource"] = Relationship(
        sa_relationship_kwargs={"uselist": False}, back_populates="source"
    )
    pull_status: Optional["SourcePullStatus"] = Relationship(back_populates="source")
