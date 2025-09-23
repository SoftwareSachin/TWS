from uuid import UUID

from sqlalchemy import Index
from sqlmodel import Field, SQLModel

from app.utils.uuid6 import uuid7


class DatasetFileLink(SQLModel, table=True):
    __tablename__ = "dataset_file_links"
    r2r_id: UUID = Field(default_factory=uuid7, nullable=True)
    dataset_id: UUID = Field(foreign_key="datasets.id", primary_key=True)
    file_id: UUID = Field(foreign_key="files.id", primary_key=True)


# Add composite indexes
Index(
    "idx_dataset_file_link_composite",
    DatasetFileLink.dataset_id,
    DatasetFileLink.file_id,
)
Index("idx_dataset_file_link_dataset_id", DatasetFileLink.dataset_id)
Index("idx_dataset_file_link_file_id", DatasetFileLink.file_id)
