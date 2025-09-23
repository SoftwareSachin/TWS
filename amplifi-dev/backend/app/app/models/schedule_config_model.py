from typing import Optional
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel

from app.models.workflow_model import Workflow


class ScheduleConfigBase(SQLModel):
    pass


class ScheduleConfig(ScheduleConfigBase, table=True):
    __tablename__ = "schedule_config"
    id: UUID = Field(primary_key=True, default_factory=uuid4)
    cron_expression: Optional[str] = Field(default=None)
    workflow_id: UUID = Field(foreign_key="workflows.id", nullable=False)

    workflows: "Workflow" = Relationship(back_populates="schedule_configs")
