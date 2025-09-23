from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlalchemy import JSON, Index
from sqlmodel import Column, Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel
from app.models.chat_app_datsets_model import LinkChatAppDatasets
from app.models.chunking_config_model import ChunkingConfig
from app.models.dataset_file_link_model import DatasetFileLink
from app.models.file_ingestion_model import FileIngestion
from app.models.transferred_files_model import TransferredFiles
from app.models.vanna_trainings_model import VannaTraining

if TYPE_CHECKING:
    from app.models.chat_app_model import ChatApp
    from app.models.document_model import Document  # Import the Document model
    from app.models.graph_model import Graph  # Import the Graph model
    from app.models.workflow_model import Workflow
    from app.models.workspace_model import Workspace


class DatasetBase(SQLModel):
    name: str = Field(nullable=False)
    description: Optional[str] = None
    source_id: Optional[UUID] = Field(default=None, nullable=True)
    r2r_collection_id: Optional[UUID] = Field(default=None, nullable=True)
    knowledge_graph: Optional[bool] = Field(default=False, nullable=True)


class Dataset(BaseUUIDModel, DatasetBase, table=True):
    __tablename__ = "datasets"
    # Current phase of graph building process
    graph_build_phase: Optional[str] = Field(default=None, nullable=True)

    # When the graph building was requested
    graph_build_requested_at: Optional[datetime] = Field(default=None, nullable=True)

    # Time when build API was called (graph considered built at this point)
    graph_build_completed_at: Optional[datetime] = Field(default=None, nullable=True)

    # Last time we checked extraction status
    last_extraction_check_at: Optional[datetime] = Field(default=None, nullable=True)

    workspace_id: UUID = Field(foreign_key="workspaces.id")

    # Add dedicated ingestion tracking fields
    ingestion_status: Optional[str] = Field(default=None, nullable=True)
    ingestion_stats: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    ingestion_last_updated: Optional[datetime] = Field(default=None, nullable=True)

    workspace: "Workspace" = Relationship(
        back_populates="datasets", sa_relationship_kwargs={"lazy": "selectin"}
    )
    files: List["File"] = Relationship(  # noqa: F821 # type: ignore
        back_populates="datasets", link_model=DatasetFileLink
    )
    file_ingestion: List["FileIngestion"] = Relationship(back_populates="dataset")
    embedding_configs: List["EmbeddingConfig"] = Relationship(back_populates="dataset")  # type: ignore # noqa: F821
    chunking_configs: List["ChunkingConfig"] = Relationship(back_populates="dataset")
    workflows: List["Workflow"] = Relationship(back_populates="datasets")
    transferred_files: List["TransferredFiles"] = Relationship(back_populates="dataset")
    chatapps: List["ChatApp"] = Relationship(
        back_populates="datasets",
        link_model=LinkChatAppDatasets,
        sa_relationship_kwargs={
            "lazy": "selectin",
        },
    )
    # relationship to Document model
    documents: List["Document"] = Relationship(back_populates="dataset")

    # relationship to Graph model
    graphs: List["Graph"] = Relationship(back_populates="dataset")
    vanna_trainings: List["VannaTraining"] = Relationship(back_populates="dataset")


# Add indexes
Index(
    "idx_dataset_workspace_active",
    Dataset.workspace_id,
    Dataset.deleted_at,
    postgresql_where=Dataset.deleted_at.is_(None),
)
Index(
    "idx_dataset_name_workspace",
    Dataset.name,
    Dataset.workspace_id,
    postgresql_where=Dataset.deleted_at.is_(None),
)
Index("idx_dataset_name", Dataset.name)
