from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.source_model import Source


class PGVector(BaseUUIDModel, table=True):
    __tablename__ = "source_pgvector"

    source_id: UUID = Field(foreign_key="sources.id", primary_key=True)
    host: str = Field(..., nullable=False)
    port: int = Field(..., nullable=False)
    database_name: str = Field(..., nullable=False)
    username: str = Field(..., nullable=False)
    password: str = Field(..., nullable=False)

    source: Optional["Source"] = Relationship(back_populates="source_pgvector")
