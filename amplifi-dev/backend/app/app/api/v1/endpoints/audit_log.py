from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi_pagination import Params

from app.api import deps
from app.crud.log_crud import log_crud as crud
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData

router = APIRouter()


@router.get("/workspace/{workspace_id}/logs")
async def get_logs(
    workspace_id: UUID,
    params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.workspace_check),
) -> List[str]:
    """
    Retrieves logs for the specified workspace.

    Required roles:
    - admin
    - developer

    Note:
    - It takes in the workspace id and returns (paginated)logs for create, update and delete operation in that workspace, its associated datasets, files and source connectors.
    """
    logs = await crud.get_workspace_logs(workspace_id=workspace_id, params=params)
    return logs
