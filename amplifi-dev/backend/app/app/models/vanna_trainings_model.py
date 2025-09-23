from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.dataset_model import Dataset


class VannaTrainingBase(SQLModel):
    documentation: Optional[str] = Field(default=None)
    version_id: int = Field(default=1)


class VannaTraining(BaseUUIDModel, VannaTrainingBase, table=True):
    __tablename__ = "vanna_trainings"

    dataset_id: UUID = Field(foreign_key="datasets.id")
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    question_sql_pairs: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSON)
    )

    # Relationships
    dataset: Optional["Dataset"] = Relationship(
        back_populates="vanna_trainings",
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "VannaTraining.dataset_id == Dataset.id",
        },
    )
