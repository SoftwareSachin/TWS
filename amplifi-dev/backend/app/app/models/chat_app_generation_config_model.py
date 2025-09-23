from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.schemas.rag_generation_schema import ChatModelEnum
from app.utils.uuid6 import uuid7

if TYPE_CHECKING:
    from app.models.chat_app_model import ChatApp


class ChatAppGenerationConfigBase(SQLModel):
    max_chunks_retrieved: Optional[int] = 3
    llm_model: ChatModelEnum = Field(
        default=ChatModelEnum.GPT35, description="Model for chat completion"
    )

    temperature: Optional[int] = 1
    top_p: Optional[int] = 1
    max_tokens_to_sample: Optional[int] = None


class ISqlGenerationConfigCreate(SQLModel):
    llm_model: ChatModelEnum = Field(
        default=ChatModelEnum.GPT35, description="Model for chat completion"
    )


class ChatAppGenerationConfig(ChatAppGenerationConfigBase, table=True):
    __tablename__ = "chat_app_generation_configs"
    id: UUID = Field(
        default_factory=uuid7,
        primary_key=True,
        index=True,
        nullable=False,
    )
    chatapp_id: UUID = Field(foreign_key="chatapps.id", nullable=True)
    chatapp: "ChatApp" = Relationship(
        back_populates="chat_app_generation_config",
    )
