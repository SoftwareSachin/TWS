# from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.api import deps
from app.models.embeddingConfig_model import EmbeddingConfig
from app.schemas.embeddingConfig_schema import (
    IEmbeddingConfigCreate,
    IEmbeddingConfigRead,
    IEmbeddingConfigUpdate,
)
from app.schemas.response_schema import (  # IDeleteResponseBase
    IGetResponseBase,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.exceptions import IdNotFoundException

router = APIRouter()


# Create Embedding Configuration Route
@router.post("/dataset/{dataset_id}/embedding_config")
async def create_embedding_config(
    dataset_id: UUID,
    obj_in: IEmbeddingConfigCreate,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
) -> IPostResponseBase[IEmbeddingConfigRead]:
    """
    Adds a new embedding configuration into a dataset.

    Required roles:
    - admin
    - developer
    """
    new_embedding_config = (
        await crud.embeddingConfig_crud.embedding_config.create_embedding_config(
            dataset_id=dataset_id, obj_in=obj_in, db_session=db
        )
    )
    return create_response(
        data=new_embedding_config, message="Embedding configuration added successfully"
    )


# Get Embedding Configuration by ID
@router.get("/dataset/{dataset_id}/embedding_config/{embedding_config_id}")
async def get_embedding_config_by_id(
    dataset_id: UUID,
    embedding_config_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
) -> IGetResponseBase[IEmbeddingConfigRead]:
    """
    Retrieves an embedding configuration by its ID.

    Required roles:
    - admin
    - member
    - developer
    """
    embedding_config = (
        await crud.embeddingConfig_crud.embedding_config.get_embedding_config(
            dataset_id=dataset_id,
            embedding_config_id=embedding_config_id,
            db_session=db,
        )
    )
    if embedding_config:
        return create_response(data=embedding_config)
    else:
        raise IdNotFoundException(EmbeddingConfig, embedding_config_id)


# Update Embedding Configuration by ID
@router.put("/dataset/{dataset_id}/embedding_config/{embedding_config_id}")
async def update_embedding_config(
    dataset_id: UUID,
    embedding_config_id: UUID,
    obj_in: IEmbeddingConfigUpdate,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
) -> IGetResponseBase[IEmbeddingConfigRead]:
    """
    Updates an existing embedding configuration in a dataset.

    Required roles:
    - admin
    - developer
    """
    updated_embedding_config = (
        await crud.embeddingConfig_crud.embedding_config.update_embedding_config(
            dataset_id=dataset_id,
            embedding_config_id=embedding_config_id,
            obj_in=obj_in,
            db_session=db,
        )
    )
    return create_response(
        data=updated_embedding_config,
        message="Embedding configuration updated successfully",
    )


# # Delete Embedding Configuration by ID
# @router.delete("/dataset/{dataset_id}/embedding_config/{embedding_config_id}")
# async def delete_embedding_config(
#     dataset_id: UUID,
#     embedding_config_id: UUID,
#     db: AsyncSession = Depends(deps.get_db),
#     current_user: UserData= Depends(deps.get_current_user()),
# ) -> IDeleteResponseBase[IEmbeddingConfigRead]:
#     """
#     Deletes an embedding configuration by its ID.
#
#     Required roles:
#     - admin
#     - developer
#     """
#     deleted_embedding_config = await crud.embeddingConfig_crud.embedding_config.delete_embedding_config(
#         dataset_id=dataset_id,
#         embedding_config_id=embedding_config_id,
#         db_session=db
#     )
#     if deleted_embedding_config:
#         return create_response(data=deleted_embedding_config, message="Embedding configuration deleted successfully")
#     else:
#         raise IdNotFoundException(EmbeddingConfig, embedding_config_id)
