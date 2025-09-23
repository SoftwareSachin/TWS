from fastapi import APIRouter, Depends, HTTPException

from app.api import deps
from app.crud.platform_crud import (  # Assuming this is your CRUD instance
    deployment_info,
)
from app.schemas.platform_schema import IDeploymentInfoRead
from app.schemas.response_schema import IGetResponseBase
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData

router = APIRouter()


# Directly fetch the latest deployment info without using get_db dependency
@router.get("/deployment_info", response_model=IGetResponseBase[IDeploymentInfoRead])
async def get_deployment_info(
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
        )
    ),
):
    """
    Retrieves the latest deployment information.

    Required roles:
    - admin
    - developer
    - member
    """
    # Fetch the latest deployment info using the CRUD method
    deployment = await deployment_info.get_latest()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment info not found")

    # Return the data wrapped in IGetResponseBase
    return IGetResponseBase(data=deployment)
