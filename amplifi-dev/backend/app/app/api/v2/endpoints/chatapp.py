from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Params

from app.api import deps
from app.be_core.logger import logger
from app.crud.chat_app_crud import chatapp
from app.models import ChatApp
from app.models.chat_app_generation_config_model import (
    ChatAppGenerationConfigBase,
)
from app.schemas.chat_schema import (
    IChatAppRead,
    IChatAppV2Create,
    IChatAppV2Read,
)
from app.schemas.common_schema import IOrderEnum
from app.schemas.response_schema import (  # IPostResponseBase,
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData

router = APIRouter()
crud = chatapp


@router.post("/workspace/{workspace_id}/chat_app")
async def create_chatapps(
    workspace_id: UUID,
    request_data: IChatAppV2Create,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.user_workspace_access_check),
) -> IPostResponseBase[ChatApp]:
    try:
        logger.info(f"Creating chat app for workspace {workspace_id}")
        chat_db = await crud.create_chatapp_v2(
            workspace_id=workspace_id, obj_in=request_data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return create_response(data=chat_db, message="ChatApp created successfully")


@router.put(
    "/workspace/{workspace_id}/chat_app/{chatapp_id}",
    response_model=IPostResponseBase[IChatAppV2Read],
)
async def update_chatapp_v2(
    workspace_id: UUID,
    chatapp_id: UUID,
    request_data: IChatAppV2Create,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.chatapp_check),
) -> IPostResponseBase[IChatAppV2Read]:
    """
    Updates an existing ChatApp in a workspace.

    Required roles:
    - admin
    - developer
    """
    try:
        updated_chatapp = await crud.update_chatapp_v2(
            chatapp_id=chatapp_id,
            workspace_id=workspace_id,
            obj_in=request_data,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return create_response(data=updated_chatapp, message="ChatApp updated successfully")


@router.get(
    "/workspace/{workspace_id}/chat_app/{chatapp_id}",
    response_model=IGetResponseBase[IChatAppV2Read],
)
async def get_chatapp(
    chatapp_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.chatapp_check),
) -> IGetResponseBase[IChatAppV2Read]:
    """
    Retrieves a chat app by its ID.

    Required roles:
    - admin
    - member
    - developer
    """
    chat_app_record = await crud.get_chatapp_v2_by_id(chatapp_id=chatapp_id)
    if not chat_app_record:
        raise HTTPException(
            status_code=404, detail=f"ChatApp with id {chatapp_id} not found"
        )

    return create_response(
        data=chat_app_record, message="ChatApp retrieved successfully"
    )


@router.get(
    "/workspace/{workspace_id}/chat_apps",
    response_model=IGetResponsePaginated[IChatAppV2Read],
    summary="Get V2 chat apps",
)
async def get_chatapps_v2(
    workspace_id: UUID,
    params: Params = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.user_workspace_access_check),
) -> IGetResponsePaginated[IChatAppV2Read]:
    """
    Retrieves all V2 chat apps (agent-based) of a workspace.

    Required roles:
    - admin
    - member
    - developer
    """
    paginated_chatapps = await crud.get_chatapps_v2(
        workspace_id=workspace_id,
        pagination_params=params,
        order=order,
    )

    pydantic_chatapps = [
        IChatAppV2Read(**chatapp.model_dump()) for chatapp in paginated_chatapps.items
    ]

    paginated_response = IGetResponsePaginated.create(
        items=pydantic_chatapps,
        total=paginated_chatapps.total,
        params=params,
    )
    return create_response(data=paginated_response)


@router.get("/my_chat_app", response_model=IGetResponsePaginated[IChatAppRead])
async def get_chatapps_user(
    params: Params = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
) -> IGetResponsePaginated[IChatAppRead]:
    """
    Retrieves all chat apps of current user (based on the workspaces they have access to).

    Required roles:
    - admin
    - member
    - developer
    """
    paginated_chatapps = await crud.get_chatapps_by_user_id(
        user_id=current_user.id,
        pagination_params=params,
        order=order,
    )
    pydantic_datasets = [
        IChatAppRead(
            **chatapp.model_dump(),
            datasets=[dataset.id for dataset in chatapp.datasets],
            generation_config=(
                ChatAppGenerationConfigBase(
                    **chatapp.chat_app_generation_config.model_dump()
                )
                if chatapp.chat_app_generation_config
                else ChatAppGenerationConfigBase()
            ),
        )
        for chatapp in paginated_chatapps.items
    ]
    paginated_response = IGetResponsePaginated.create(
        items=pydantic_datasets,
        total=paginated_chatapps.total,
        params=params,
    )
    return create_response(data=paginated_response)


@router.delete(
    "/workspace/{workspace_id}/chat_app/{chatapp_id}",
    response_model=IGetResponseBase[IChatAppRead],
)
async def delete_chatapp(
    workspace_id: UUID,
    chatapp_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.chatapp_check),
) -> IGetResponseBase[IChatAppRead]:
    """
    Deletes a chat app by its ID.

    Required roles:
    - admin
    - developer
    """
    chat_app_record = await crud.delete_chatapp_by_id(chatapp_id=chatapp_id)
    return create_response(
        data=chat_app_record,
        message=f"ChatApp {chat_app_record.name} deleted successfully",
    )
