from typing import Optional

from sqlmodel import Field, SQLModel

# from app.models.base_uuid_model import BaseUUIDModel


class DeploymentInfo(SQLModel, table=True):  # Inherits from BaseUUIDModel
    __tablename__ = "deployment_info"

    version: str = Field(
        ..., unique=True, primary_key=True
    )  # Ensures version is unique
    product_documentation_link: Optional[str] = Field(default=None)
    technical_documentation_link: Optional[str] = Field(default=None)
