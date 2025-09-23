from contextvars import Context  # noqa: F401
from datetime import datetime
from enum import Enum  # noqa: F401
from typing import Annotated, List, Literal, Optional
from uuid import UUID

from annotated_types import Len
from pydantic import BaseModel, field_validator

from app.models.chat_app_generation_config_model import (
    ChatAppGenerationConfigBase,
    ISqlGenerationConfigCreate,
)
from app.models.chat_history_model import ChatHistoryBase


class IChatAppTypeEnum(str, Enum):
    unstructured_chat_app = "unstructured_chat_app"
    sql_chat_app = "sql_chat_app"
    agentic = "agentic"


class ChatAppBase(BaseModel):
    name: str
    description: Optional[str] = None
    voice_enabled: bool = False


class ChatAppV1Base(ChatAppBase):
    chat_app_type: IChatAppTypeEnum
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_retention_days: Optional[int] = 7


class ChatAppV2Base(ChatAppBase):
    pass


class IChatAppCreate(ChatAppV1Base):
    chat_app_type: Literal[IChatAppTypeEnum.unstructured_chat_app]
    datasets: Annotated[List[UUID], Len(min_length=1)]
    generation_config: ChatAppGenerationConfigBase
    graph_enabled: bool = False

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty or just whitespace")
        return v


class IChatAppV2Create(ChatAppV2Base):
    agent_id: UUID

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name must not be empty or just whitespace")
        return v

    @field_validator("agent_id")
    def agent_must_not_be_blank(cls, v: UUID) -> UUID:
        if not v:
            raise ValueError("Agent id is required.")
        return v


class ISqlChatAppCreate(ChatAppV1Base):
    chat_app_type: Literal[IChatAppTypeEnum.sql_chat_app]
    generation_config: ISqlGenerationConfigCreate
    dataset_id: UUID

    @field_validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty or just whitespace")
        return v


class ISqlChatAppRead(ChatAppV1Base, from_attributes=True):
    id: UUID
    generation_config: ChatAppGenerationConfigBase


class IChatAppRead(ChatAppBase, from_attributes=True):
    id: UUID
    datasets: Annotated[List[UUID], Len(min_length=0)]
    generation_config: Optional[ChatAppGenerationConfigBase] = None
    graph_enabled: Optional[bool] = False
    created_at: Optional[datetime] = None
    chat_app_type: Optional[str] = None
    system_prompt: Optional[str] = None


class IChatAppV2Read(ChatAppV2Base, from_attributes=True):
    id: UUID
    agent_id: UUID
    created_at: Optional[datetime] = None
    chat_app_type: Optional[str] = None
    system_prompt: Optional[str] = None
    workspace_id: Optional[UUID] = None


class IChatSessionCreate(BaseModel):
    title: str


class IChatSessionRead(BaseModel, from_attributes=True):
    id: UUID
    title: str
    chatapp_id: UUID
    created_at: Optional[datetime] = None


class ChatHistoryLine(ChatHistoryBase):
    created_at: datetime


class IChatHistoryLineCreate(ChatHistoryBase):
    pass


class IChatHistoryCreate(BaseModel):
    histories: List[IChatHistoryLineCreate]


class ChatAppRagGenerationRequest(BaseModel):
    query: str
    history: List[dict] = []


# class UnstructuredChatAppRAGGenerationRead(BaseModel):
#     answer: str
#     full_response: str
#     contexts_found: List[Context] = []
#     input_tokens: int
#     output_tokens: int


# class QuestionRequest(BaseModel):
#     question: str
class IChatAppReadBase(ChatAppBase, from_attributes=True):
    id: UUID
    generation_config: ChatAppGenerationConfigBase


class IUnstructuredChatAppRead(IChatAppReadBase):
    datasets: Annotated[List[UUID], Len(min_length=0)]


class ISqlChatAppReads(IChatAppReadBase):
    pass  # No datasets field for SQL chat apps


class ChatRequest(BaseModel):
    chat_app_id: UUID
    chat_session_id: Optional[UUID] = None
    query: str
