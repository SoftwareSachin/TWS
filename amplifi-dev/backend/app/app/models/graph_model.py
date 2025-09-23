# V2 Graph (in-house setup)
from enum import Enum
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.dataset_model import Dataset
    from app.models.extracted_entity_model import ExtractedEntity


class GraphStatus(str, Enum):
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"
    NOT_STARTED = "not_started"


class GraphBase(SQLModel):
    entities_status: GraphStatus = Field(
        default=GraphStatus.NOT_STARTED, nullable=False
    )
    relationships_status: GraphStatus = Field(
        default=GraphStatus.NOT_STARTED, nullable=False
    )
    error_message: Optional[str] = Field(default=None, nullable=True)


class Graph(BaseUUIDModel, GraphBase, table=True):
    __tablename__ = "graphs"
    dataset_id: UUID = Field(foreign_key="datasets.id")
    dataset: "Dataset" = Relationship(back_populates="graphs")
    extracted_entities: list["ExtractedEntity"] = Relationship(back_populates="graph")
