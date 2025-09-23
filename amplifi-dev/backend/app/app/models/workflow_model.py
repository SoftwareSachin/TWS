from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.dataset_model import Dataset
    from app.models.destination_model import Destination
    from app.models.organization_model import Organization
    from app.models.schedule_config_model import ScheduleConfig
    from app.models.transferred_files_model import TransferredFiles
    from app.models.workflow_run_model import WorkflowRun


class WorkflowBase(SQLModel):
    name: str = Field(nullable=False)
    is_active: bool = Field(default=False)
    description: Optional[str] = Field(default=None)


class Workflow(BaseUUIDModel, WorkflowBase, table=True):
    __tablename__ = "workflows"
    organization_id: UUID = Field(foreign_key="organizations.id", nullable=False)
    destination_id: UUID = Field(foreign_key="destinations.id", nullable=False)
    dataset_id: UUID = Field(foreign_key="datasets.id", nullable=False)
    scheduled: bool = Field(default=False)

    organization: "Organization" = Relationship(back_populates="workflows")
    destination: Optional["Destination"] = Relationship(back_populates="workflows")
    schedule_configs: Optional["ScheduleConfig"] = Relationship(
        back_populates="workflows", sa_relationship_kwargs={"lazy": "selectin"}
    )
    workflow_runs: List["WorkflowRun"] = Relationship(back_populates="workflow")
    datasets: "Dataset" = Relationship(back_populates="workflows")
    transferred_files: List["TransferredFiles"] = Relationship(
        back_populates="workflow"
    )
