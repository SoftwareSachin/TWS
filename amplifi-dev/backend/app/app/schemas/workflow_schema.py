from enum import Enum
from typing import Optional
from uuid import UUID

from croniter import croniter
from pydantic import BaseModel, Field, validator


class WorkflowRunStatusType(str, Enum):
    Processing = "Processing"
    Failed = "Failed"
    Success = "Success"
    Not_Started = "Not_Started"


class WorkflowScheduleConfig(BaseModel):
    cron_expression: Optional[str] = Field(None)

    @validator("cron_expression")
    def validate_cron_expression(cls, value):
        if value and not croniter.is_valid(value):
            raise ValueError("Invalid cron expression")
        return value

    class Config:
        title = "WorkflowScheduleConfig"


class IWorkflowCreate(BaseModel):
    name: str = Field(..., title="Name")
    description: Optional[str] = Field(None, title="Description")
    is_active: bool = Field(default=True, title="Is Active")
    destination_id: UUID = Field(..., title="Destination Id")
    dataset_id: UUID = Field(..., title="Dataset Id")
    schedule_config: Optional[WorkflowScheduleConfig] = Field(
        None, title="Schedule Config"
    )

    class Config:
        title = "IWorkflowCreate"


class IWorkflowRead(BaseModel):
    name: str = Field(..., title="Name")
    description: Optional[str] = Field(None, title="Description")
    id: UUID = Field(..., title="Id")
    is_active: bool = Field(default=True, title="Is Active")
    dataset_id: UUID = Field(..., title="Dataset Id")
    dataset_name: Optional[str] = Field(None, title="Dataset Name")
    destination_id: UUID = Field(..., title="Destination Id")
    destination_name: Optional[str] = Field(None, title="Destination Name")
    schedule_config: Optional[WorkflowScheduleConfig] = Field(
        None, title="Schedule Config"
    )

    class Config:
        title = "IWorkflowRead"


class IWorkflowUpdate(BaseModel):
    name: str = Field(..., title="Name")
    description: Optional[str] = Field(None, title="Description")
    is_active: bool = Field(default=True, title="Is Active")
    schedule_config: Optional[WorkflowScheduleConfig] = Field(
        None, title="Schedule Config"
    )

    class Config:
        title = "IWorkflowUpdate"


class IWorkflowExecute(BaseModel):
    run_config: Optional[dict] = Field(None, title="Run Config")

    class Config:
        title = "IWorkflowExecute"


class IWorkflowExecutionRead(BaseModel):
    status: str = Field(default="Success", title="Status")

    class Config:
        title = "IWorkflowExecutionRead"


class IWorkflowRunRead(BaseModel):
    run_id: UUID = Field(..., title="Run Id")
    status: WorkflowRunStatusType = Field(..., title="Run Status")
    created_at: str = Field(..., title="Started At")
    finished_at: Optional[str] = Field(None, title="Finished At")

    class Config:
        title = "IWorkflowRunRead"
