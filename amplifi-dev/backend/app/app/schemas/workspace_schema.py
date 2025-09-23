from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel

from app.utils.partial import optional


# Workspace Base Model (like GroupBase)
class WorkspaceBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_active: bool = True


# IWorkspaceCreate model
class IWorkspaceCreate(WorkspaceBase):
    pass  # No extra fields


# IWorkspaceRead model
class IWorkspaceRead(WorkspaceBase):
    id: UUID
    organization_id: UUID
    # total_files : Optional[int] = 0 by default 0
    total_files: Optional[int] = None  # by default null
    processed_chunks: int

    class Config:
        # orm_mode = True
        from_attributes = True  # This allows Pydantic to convert from SQLAlchemy models


class IWorkspacePerformSearchRead(BaseModel):
    vector_search_results: List[str]  # List of vector search results
    kg_search_results: List[str]  # List of knowledge graph search results


class WorkspaceUserIDList(BaseModel):
    user_ids: List[UUID]


# IWorkspaceUpdate model - all fields optional
@optional()
class IWorkspaceUpdate(WorkspaceBase):
    pass
