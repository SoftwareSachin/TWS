from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, Relationship, SQLModel

from app.utils.uuid6 import uuid7

if TYPE_CHECKING:
    from app.models.chat_session_model import ChatSession


class ChatHistoryBase(SQLModel):
    user_query: str
    contexts: Optional[list[dict[str, Any]]] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=True)
    )
    llm_response: str
    llm_model: str
    input_tokens: Optional[int] = Field(default=None)
    output_tokens: Optional[int] = Field(default=None)
    total_tokens: Optional[int] = Field(default=None)
    pydantic_message: Optional[list[dict[str, Any]]] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=True)
    )


class ChatHistory(ChatHistoryBase, table=True):
    __tablename__ = "chat_histories"
    id: UUID = Field(
        default_factory=uuid7,
        primary_key=True,
        index=True,
        nullable=False,
    )
    chat_session_id: UUID = Field(foreign_key="chat_sessions.id")
    chat_sessions: "ChatSession" = Relationship(back_populates="chat_histories")
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
