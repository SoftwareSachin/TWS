from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.tools_models import Tool


class SystemToolBase(SQLModel):
    python_module: str
    function_name: str
    is_async: bool = Field(default=False, description="Run via Celery if True")

    input_schema: Optional[str] = Field(
        default=None, description="Pydantic-style input schema as JSON string"
    )
    output_schema: Optional[str] = Field(
        default=None, description="Optional output schema as JSON string"
    )
    function_signature: Optional[str] = Field(
        default=None, description="Python-style signature"
    )


class SystemTool(BaseUUIDModel, SystemToolBase, table=True):
    __tablename__ = "system_tools"

    tool_id: UUID = Field(foreign_key="tools.id", unique=True)
    tool: "Tool" = Relationship(back_populates="system_tool")
