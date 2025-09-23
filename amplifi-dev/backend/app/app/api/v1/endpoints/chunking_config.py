from typing import Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app import crud
from app.api import deps
from app.models.chunking_config_model import ChunkingConfig
from app.schemas.chunking_config_response_schema import (
    R2RProviderChunkingConfigResponse,
    UnstructuredProviderChunkingConfigResponse,
)
from app.schemas.chunking_config_schema import (
    R2RChunkingConfig,
    UnstructuredChunkingConfig,
)
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.exceptions import IdNotFoundException

router = APIRouter()


@router.post("/dataset/{dataset_id}/chunking_config")
async def add_chunking_config_into_a_dataset(
    dataset_id: UUID,
    chunking_config: Union[UnstructuredChunkingConfig, R2RChunkingConfig],
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
) -> IPostResponseBase[
    Union[
        UnstructuredProviderChunkingConfigResponse,
        R2RProviderChunkingConfigResponse,
    ]
]:
    """
    Adds Chunking configuration into dataset

    Required roles:
    - admin
    - developer

    Note:
    - Adds chunking configuration (if do not exist already) and updates chunking configuration (if exist already) to a dataset.
    - For now, chunking config can be of only unstructured type.
    """
    try:
        new_chunking_config = (
            await crud.chunking_config.create_or_update_chunking_config(
                obj_in=chunking_config,
                dataset_id=dataset_id,
            )
        )
    except ValueError as e:
        raise HTTPException(
            400,
            f"Invalid Value for Provider: '{chunking_config.provider}'. Raised Error: {e}",
        )
    return create_response(message="Data created correctly", data=new_chunking_config)


@router.get("/dataset/{dataset_id}/chunking_config/{chunking_config_id}")
async def get_chunking_config_by_id(
    dataset_id: UUID,
    chunking_config_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
) -> IGetResponseBase[
    Union[
        UnstructuredChunkingConfig,
        R2RChunkingConfig,
    ]
]:
    """
    Gets Chunking configuration by its id

    Required roles:
    - admin
    - member
    - developer
    """
    chunking_config = await crud.chunking_config.get_chunking_config(
        chunking_config_id=chunking_config_id, dataset_id=dataset_id
    )
    if chunking_config:
        return create_response(message="Data got correctly", data=chunking_config)
    else:
        raise IdNotFoundException(ChunkingConfig, chunking_config_id)


# @router.delete("/dataset/{dataset_id}/chunking_config/{chunking_config_id}")
async def remove_chunking_config(
    dataset_id: UUID,
    chunking_config_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IDeleteResponseBase[Union[UnstructuredChunkingConfig, R2RChunkingConfig]]:
    """
    Deletes a chunking configuration by id

    Required roles:
    - admin
    """
    chunking_config = await crud.chunking_config.delete_chunking_config(
        chunking_config_id=chunking_config_id, dataset_id=dataset_id
    )
    return create_response(message="Data deleted correctly", data=chunking_config)
