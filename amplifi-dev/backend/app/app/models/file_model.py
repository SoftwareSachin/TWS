from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlalchemy import Column, Index, Integer, String
from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel
from app.models.dataset_file_link_model import DatasetFileLink
from app.models.dataset_model import Dataset
from app.models.document_model import Document
from app.models.file_ingestion_model import FileIngestion
from app.schemas.file_schema import FileStatusEnum

if TYPE_CHECKING:
    from app.models.file_split_model import FileSplit
    from app.models.workspace_model import Workspace


class FileBase(SQLModel):
    filename: str = Field(nullable=False)
    mimetype: str = Field(nullable=False)
    size: Optional[int] = None
    file_path: str = Field(nullable=False)
    status: Optional[FileStatusEnum] = Field(sa_column=Column(String, nullable=True))


class File(BaseUUIDModel, FileBase, table=True):
    __tablename__ = "files"
    workspace_id: UUID = Field(foreign_key="workspaces.id")
    source_id: Optional[UUID] = Field(nullable=True)
    # Flag to indicate if the file requires splitting
    requires_splitting: Optional[bool] = Field(default=False, nullable=True)
    rows: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    columns: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )

    datasets: List[Dataset] = Relationship(
        back_populates="files", link_model=DatasetFileLink
    )
    file_ingestion: Optional[List["FileIngestion"]] = Relationship(
        back_populates="file"
    )
    # Link to file splits for large files
    file_splits: Optional[List["FileSplit"]] = Relationship(
        back_populates="original_file"
    )

    documents: Optional[List["Document"]] = Relationship(
        back_populates="file", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

    # Establish a back-reference to Workspace
    workspace: "Workspace" = Relationship(back_populates="files")


# Add indexes after the model definition
Index("idx_file_filename", File.filename)
Index(
    "idx_file_filename_lower",
    File.filename,
    postgresql_using="btree",
    postgresql_ops={"filename": "text_pattern_ops"},
)
# For PostgreSQL with pg_trgm extension
Index(
    "idx_file_filename_trgm",
    File.filename,
    postgresql_using="gin",
    postgresql_ops={"filename": "gin_trgm_ops"},
)
Index("idx_file_workspace_id", File.workspace_id)
