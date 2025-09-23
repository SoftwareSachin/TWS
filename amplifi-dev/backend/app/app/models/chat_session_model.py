from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.chat_app_model import ChatApp
    from app.models.chat_history_model import ChatHistory
    from app.models.user_model import User


class ChatSession(BaseUUIDModel, SQLModel, table=True):
    __tablename__ = "chat_sessions"
    title: Optional[str] = None
    user_id: UUID = Field(foreign_key="User.id")
    user: "User" = Relationship(back_populates="chat_sessions")
    chatapp_id: UUID = Field(foreign_key="chatapps.id")
    chatapp: "ChatApp" = Relationship(back_populates="chat_sessions")
    chat_histories: List["ChatHistory"] = Relationship(
        back_populates="chat_sessions",
        sa_relationship_kwargs={
            "lazy": "selectin",
            "primaryjoin": "ChatSession.id == ChatHistory.chat_session_id",
        },
    )
    total_tokens_per_chat_session: int = Field(default=0, nullable=False)
    input_tokens_per_chat_session: int = Field(default=0, nullable=False)
    output_tokens_per_chat_session: int = Field(default=0, nullable=False)
