from typing import Dict, Optional
from uuid import UUID

from pydantic import BaseModel


class TokenCountBase(BaseModel):
    organization_id: UUID
    org_level_tokens: int
    workspace_map: Optional[Dict[UUID, int]] = None
    dataset_map: Optional[Dict[UUID, int]] = None
    dataset_level_tokens: int
    workspace_level_tokens: int


class ITokenCountCreate(TokenCountBase):
    pass


class ITokenCountRead(TokenCountBase):
    class Config:
        from_attributes = True


class TokenCountLLMResponseBase(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class WorkspaceTokenCountLLMResponse(TokenCountLLMResponseBase):
    workspace_id: UUID


class OrganizationTokenCountLLMResponse(TokenCountLLMResponseBase):
    organization_id: UUID


class ChatAppTokenCountLLMResponse(TokenCountLLMResponseBase):
    chatapp_id: UUID


class ChatSessionTokenCountLLMResponse(TokenCountLLMResponseBase):
    chat_session_id: UUID


class TokenCountsLLMResponse(BaseModel):
    workspace_token_counts: WorkspaceTokenCountLLMResponse
    organization_token_counts: Optional[OrganizationTokenCountLLMResponse] = None
    chat_app_token_counts: Optional[ChatAppTokenCountLLMResponse] = None
    chat_session_token_counts: Optional[ChatSessionTokenCountLLMResponse] = None
