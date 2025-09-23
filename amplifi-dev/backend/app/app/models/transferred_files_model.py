from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.dataset_model import Dataset
    from app.models.workflow_model import Workflow
    from app.models.workflow_run_model import WorkflowRun


class TransferredFiles(SQLModel, table=True):
    __tablename__ = "transferred_files"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    file_id: UUID
    workflow_run_id: UUID = Field(foreign_key="workflow_runs.id", nullable=False)
    dataset_id: UUID = Field(foreign_key="datasets.id", nullable=False)
    workflow_id: UUID = Field(foreign_key="workflows.id", nullable=False)
    transferred_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="pending")

    workflow_run: "WorkflowRun" = Relationship(back_populates="transferred_files")
    dataset: "Dataset" = Relationship(back_populates="transferred_files")
    workflow: "Workflow" = Relationship(back_populates="transferred_files")
