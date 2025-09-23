from typing import Optional
from uuid import UUID  # noqa: F401

from pydantic import BaseModel, HttpUrl

# # IDeploymentInfoRead Model
# class IDeploymentInfoRead(BaseModel):
#     version: str
#     product_documentation_link: Optional[HttpUrl] = None
#     technical_documentation_link: Optional[HttpUrl] = None


# IDeploymentInfoRead Model
class DeploymentInfoBase(BaseModel):
    version: str
    product_documentation_link: Optional[HttpUrl] = None
    technical_documentation_link: Optional[HttpUrl] = None


class IDeploymentInfoCreate(DeploymentInfoBase):
    pass


class IDeploymentInfoRead(DeploymentInfoBase):
    pass
