from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, SQLModel

if TYPE_CHECKING:
    pass


class LinkChatAppDatasets(SQLModel, table=True):
    __tablename__ = "chat_app_datasets"
    chatapp_id: UUID = Field(
        default=None, nullable=False, foreign_key="chatapps.id", primary_key=True
    )
    dataset_id: UUID = Field(
        default=None, nullable=False, foreign_key="datasets.id", primary_key=True
    )
