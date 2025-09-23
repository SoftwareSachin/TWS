from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.workspace_schema import IWorkspaceRead


class IOrganizationRead(BaseModel):
    name: str = Field(..., description="Name")  # Required name of the organization
    description: Optional[str] = Field(
        default=None, description="Description"
    )  # Optional description of the organization
    domain: Optional[str] = Field(
        default=None, description="Domain"
    )  # Optional domain of the organization
    id: UUID = Field(
        ..., description="Id"
    )  # Required unique identifier for the organization

    class Config:
        from_attributes = True


class IOrganizationReadWithWorkspaces(BaseModel):
    name: str = Field(..., description="Name")  # Required name of the organization
    description: Optional[str] = Field(
        default=None, description="Description"
    )  # Optional description of the organization
    id: UUID = Field(
        ..., description="Id"
    )  # Required unique identifier for the organization
    workspaces: List[IWorkspaceRead] = Field(
        ..., description="List of Workspaces"
    )  # List of workspaces related to the organization


class IOrganizationUpdate(BaseModel):
    name: Optional[str] = Field(
        default=None, description="Name"
    )  # Optional name of the organization
    description: Optional[str] = Field(
        default=None, description="Description"
    )  # Optional description of the organization


class IOrganizationCreate(BaseModel):
    name: str = Field(..., description="Name")  # Required name of the organization
    description: Optional[str] = Field(
        default=None, description="Description"
    )  # Optional description of the organization
    domain: str = Field(..., description="Domain")
