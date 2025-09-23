from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.agent_model import Agent
from app.schemas.rag_generation_schema import ChatModelEnum


class WorkspaceToolInfo(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None


class IAgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    prompt_instructions: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: ChatModelEnum = ChatModelEnum.GPT4o
    temperature: Optional[float] = 0.7
    system_prompt: Optional[str] = None
    memory_enabled: Optional[bool] = False
    agent_metadata: Optional[dict] = None
    workspace_tool_ids: List[UUID]


class IWorkspaceAgentRead(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    prompt_instructions: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: ChatModelEnum
    temperature: Optional[float] = 0.7
    system_prompt: Optional[str] = None
    memory_enabled: Optional[bool] = False
    agent_metadata: Optional[dict] = None
    workspace_tool_ids: List[UUID]
    workspace_tools: List[WorkspaceToolInfo] = []
    tool_type: Optional[str]

    @classmethod
    def from_model(
        cls,
        agent: Agent,
        workspace_tool_ids: Optional[List[UUID]] = None,
        workspace_tools: Optional[List[WorkspaceToolInfo]] = None,
        tool_type: Optional[str] = None,
    ):
        return cls(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            prompt_instructions=agent.prompt_instructions,
            llm_provider=agent.llm_provider,
            llm_model=agent.llm_model,
            temperature=agent.temperature,
            system_prompt=agent.system_prompt,
            memory_enabled=agent.memory_enabled,
            agent_metadata=agent.agent_metadata,
            workspace_tool_ids=workspace_tool_ids or [],
            workspace_tools=workspace_tools or [],
            tool_type=tool_type,
        )


# class Meta(BaseModel):
#     info: Optional[str] = None


# class IPostResponseBase(BaseModel):
#     message: Optional[str] = "Data created correctly"
#     meta: Optional[Meta] = None
#     data: Optional[IWorkspaceAgentRead] = None


# class PageBase(BaseModel):
#     items: List[IWorkspaceAgentRead]
#     total: int
#     page: int
#     size: int
#     pages: Optional[int] = None
#     previous_page: Optional[int] = None
#     next_page: Optional[int] = None


# class IGetResponsePaginated(BaseModel):
#     message: Optional[str] = ""
#     meta: Optional[Meta] = None
#     data: PageBase


class IAgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    prompt_instructions: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: ChatModelEnum = None
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None
    memory_enabled: Optional[bool] = None
    agent_metadata: Optional[dict] = None
    workspace_tool_ids: Optional[List[UUID]] = None

    class Config:
        orm_mode = True
