from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.models.workflow_model import WorkflowBase


class workflow_response_scheduleConfig(BaseModel):
    cron_expression: Optional[str]


class workflow_response_schema(WorkflowBase):
    id: UUID
    destination_id: UUID
    destination_name: Optional[str] = None
    dataset_id: UUID
    dataset_name: Optional[str] = None
    schedule_config: Optional[workflow_response_scheduleConfig] = None
