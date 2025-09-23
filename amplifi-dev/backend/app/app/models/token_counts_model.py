from uuid import UUID

from sqlmodel import Field

from app.models.base_uuid_model import BaseUUIDModel


class OrganizationEmbeddingTokenCount(BaseUUIDModel, table=True):
    __tablename__ = "organization_embedding_token_counts"

    organization_id: UUID = Field(foreign_key="organizations.id")
    org_level_token_count: int = Field(default=0)


class WorkspaceEmbeddingTokenCount(BaseUUIDModel, table=True):
    __tablename__ = "workspace_embedding_token_counts"

    org_token_count_id: UUID = Field(
        foreign_key="organization_embedding_token_counts.id"
    )
    workspace_id: UUID = Field(foreign_key="workspaces.id")
    workspace_level_token_count: int = Field(default=0)


class DatasetEmbeddingTokenCount(BaseUUIDModel, table=True):
    __tablename__ = "dataset_embedding_token_counts"

    org_token_count_id: UUID = Field(
        foreign_key="organization_embedding_token_counts.id"
    )
    workspace_id: UUID = Field(foreign_key="workspaces.id")
    dataset_id: UUID = Field(foreign_key="datasets.id")
    dataset_level_token_count: int = Field(default=0)
