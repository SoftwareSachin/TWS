from typing import TYPE_CHECKING, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel
from app.schemas.rag_generation_schema import ChatModelEnum

if TYPE_CHECKING:
    from app.models.agent_tools_model import AgentTool
    from app.models.chat_app_model import ChatApp
    from app.models.workspace_agent_model import WorkspaceAgent


class Agent(BaseUUIDModel, table=True):
    __tablename__ = "agents"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    name: str = Field(index=True, nullable=False, max_length=100)
    description: Optional[str] = None
    prompt_instructions: str = Field(nullable=False)
    llm_provider: Optional[str] = Field(default=None, max_length=50)
    llm_model: ChatModelEnum = Field(
        default=ChatModelEnum.GPT4o, description="Model for agent"
    )
    temperature: float = Field(default=0.7)
    system_prompt: Optional[str] = None
    memory_enabled: bool = Field(default=False)
    agent_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    # relationships
    tools: List["AgentTool"] = Relationship(
        back_populates="agent", sa_relationship_kwargs={"lazy": "selectin"}
    )
    chatapps: List["ChatApp"] = Relationship(
        back_populates="agent", sa_relationship_kwargs={"lazy": "selectin"}
    )
    workspace_links: List["WorkspaceAgent"] = Relationship(
        back_populates="agent", sa_relationship_kwargs={"lazy": "selectin"}
    )
