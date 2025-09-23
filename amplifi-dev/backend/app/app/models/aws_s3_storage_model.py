from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.source_model import Source


class AWSS3Storage(BaseUUIDModel, table=True):
    __tablename__ = "aws_s3_storage"

    source_id: UUID = Field(foreign_key="sources.id", primary_key=True)
    bucket_name: str = Field(..., nullable=False)
    access_id: str = Field(..., nullable=False)
    access_secret: str = Field(..., nullable=False)

    source: "Source" = Relationship(back_populates="aws_s3_storage")
