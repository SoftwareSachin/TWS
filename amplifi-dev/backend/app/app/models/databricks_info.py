from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.destination_model import Destination


class DatabricksInfo(BaseUUIDModel, table=True):
    __tablename__ = "destination_databricks"

    destination_id: UUID = Field(foreign_key="destinations.id", nullable=False)

    workspace_url: str = Field(..., nullable=False)
    token: str = Field(..., nullable=False)
    warehouse_id: str = Field(..., nullable=True)
    database_name: Optional[str] = Field(default=None)
    table_name: str = Field(..., nullable=False)

    destination: "Destination" = Relationship(back_populates="destination_databricks")
