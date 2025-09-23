from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.models.graph_model import GraphBase


class IGraphCreate(GraphBase):
    pass


class IGraphCreateEntities(GraphBase):
    entity_types: Optional[list[str]] = None


class GraphCreate(GraphBase):
    entity_types: Optional[list[str]] = None
    dataset_id: UUID


class GraphRead(GraphBase):
    entity_types: Optional[list[str]] = None
    id: UUID
    dataset_id: UUID

    class Config:
        from_attributes = True


class GraphReadEntityTypes(BaseModel):
    entity_types: list[dict]


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str


class ExtractedEntityRead(ExtractedEntity):
    id: UUID


class EntityRelationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_type: str


class GraphEntitiesRelationshipsResponse(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[EntityRelationship]
    total_entities: int
    total_relationships: int
