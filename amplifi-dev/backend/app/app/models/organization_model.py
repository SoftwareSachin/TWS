from typing import TYPE_CHECKING, List, Optional

from pydantic import EmailStr
from sqlmodel import Column, Field, Relationship, SQLModel, String

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.api_client_model import ApiClient
    from app.models.destination_model import (
        Destination,
    )
    from app.models.tools_models import Tool
    from app.models.user_model import User
    from app.models.workflow_model import Workflow
    from app.models.workspace_model import Workspace


class OrganizationBase(SQLModel):
    name: str = Field(..., index=True)


class Organization(BaseUUIDModel, OrganizationBase, table=True):
    __tablename__ = "organizations"

    domain: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)

    workspaces: List["Workspace"] = Relationship(  # noqa: F821
        back_populates="organization"
    )

    destinations: List["Destination"] = Relationship(back_populates="organization")
    workflows: List["Workflow"] = Relationship(back_populates="organization")

    users: List["User"] = Relationship(back_populates="organization")  # noqa: F821
    tools: List["Tool"] = Relationship(back_populates="organization")
    api_clients: List["ApiClient"] = Relationship(
        back_populates="organization"
    )  # noqa: F821

    r2r_user: EmailStr = Field(sa_column=Column(String))
    r2r_password: str | None = Field(default=None, nullable=True)
