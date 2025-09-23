from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.source_model import Source


class PullStatusEnum(str, Enum):
    NOT_STARTED = "not_started"
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"


class SourcePullStatus(BaseUUIDModel, table=True):
    __tablename__ = "source_pull_status"

    source_id: UUID = Field(foreign_key="sources.id", nullable=False, unique=True)
    pull_status: PullStatusEnum = Field(
        default=PullStatusEnum.NOT_STARTED, nullable=False
    )

    source: "Source" = Relationship(back_populates="pull_status")
