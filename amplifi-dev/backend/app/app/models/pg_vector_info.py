from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.destination_model import Destination


class PgVectorInfo(BaseUUIDModel, table=True):
    __tablename__ = "destination_pgvector"

    destination_id: UUID = Field(foreign_key="destinations.id")
    host: str = Field(..., nullable=False)
    port: int = Field(default=5432)
    database_name: str = Field(..., nullable=False)
    table_name: str = Field(..., nullable=True)
    username_reference: str = Field(..., nullable=False)
    password_reference: str = Field(..., nullable=False, alias="password")

    destination: Optional["Destination"] = Relationship(
        back_populates="destination_pgvector"
    )
