from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.source_model import Source


class AzureFabric(BaseUUIDModel, table=True):
    __tablename__ = "azure_fabric"

    source_id: UUID = Field(foreign_key="sources.id", primary_key=True)
    container_name: str = Field(..., nullable=False)
    sas_url: str = Field(..., nullable=False)

    source: "Source" = Relationship(back_populates="azure_fabric")
