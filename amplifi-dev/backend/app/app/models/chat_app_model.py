from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.models import Agent
from app.models.base_uuid_model import BaseUUIDModel
from app.models.chat_app_datsets_model import LinkChatAppDatasets

if TYPE_CHECKING:
    from app.models.chat_app_generation_config_model import ChatAppGenerationConfig
    from app.models.chat_session_model import ChatSession
    from app.models.dataset_model import Dataset
    from app.models.workspace_model import Workspace


class ChatAppBase(SQLModel):
    chat_app_type: str = Field(nullable=True)
    name: str = Field(nullable=False)
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_retention_days: Optional[int] = 7
    voice_enabled: bool = False
    graph_enabled: bool = False


class ChatApp(BaseUUIDModel, ChatAppBase, table=True):
    __tablename__ = "chatapps"
    workspace_id: UUID = Field(foreign_key="workspaces.id")
    workspace: "Workspace" = Relationship(back_populates="chatapps")
    chat_app_generation_config: "ChatAppGenerationConfig" = Relationship(
        back_populates="chatapp",
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    chat_sessions: List["ChatSession"] = Relationship(
        back_populates="chatapp",
    )
    datasets: List["Dataset"] = Relationship(
        back_populates="chatapps",
        link_model=LinkChatAppDatasets,
        sa_relationship_kwargs={
            "lazy": "selectin",
        },
    )
    agent_id: Optional[UUID] = Field(
        default=None, foreign_key="agents.id", nullable=True
    )
    agent: "Agent" = Relationship(back_populates="chatapps")
    total_tokens_per_chat_app: int = Field(default=0, nullable=False)
    input_tokens_per_chat_app: int = Field(default=0, nullable=False)
    output_tokens_per_chat_app: int = Field(default=0, nullable=False)
