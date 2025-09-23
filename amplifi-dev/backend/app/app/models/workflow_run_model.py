from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.models.workflow_model import Workflow
from app.schemas.workflow_schema import WorkflowRunStatusType
from app.utils.uuid6 import uuid7

if TYPE_CHECKING:
    from app.models.transferred_files_model import TransferredFiles


class WorkflowRunBase(SQLModel):
    status: WorkflowRunStatusType = Field(
        default=WorkflowRunStatusType.Not_Started, nullable=False
    )
    id: UUID = Field(default_factory=uuid7, primary_key=True, nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)


class WorkflowRun(WorkflowRunBase, table=True):
    __tablename__ = "workflow_runs"
    workflow_id: UUID = Field(foreign_key="workflows.id", nullable=False)

    workflow: "Workflow" = Relationship(back_populates="workflow_runs")
    transferred_files: list["TransferredFiles"] = Relationship(
        back_populates="workflow_run"
    )
