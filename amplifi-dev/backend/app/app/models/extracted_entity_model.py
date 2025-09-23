from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.graph_model import Graph


class ExtractedEntity(BaseUUIDModel, table=True):
    __tablename__ = "extracted_entities"

    # Foreign keys
    graph_id: UUID = Field(foreign_key="graphs.id", index=True, nullable=False)

    # Entity attributes
    name: str = Field(index=True, nullable=False)
    entity_type: str = Field(index=True, nullable=False)
    description: Optional[str] = Field(default=None, nullable=True)

    # Relationships
    graph: "Graph" = Relationship(back_populates="extracted_entities")
