from datetime import date, datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.api.deps import (  # get_r2r_client_sync,
    get_current_user,
    user_workspace_access_check,
)
from app.be_core.logger import logger
from app.crud.token_counts_crud import token_count
from app.schemas.response_schema import IPostResponseBase, create_response
from app.schemas.role_schema import IRoleEnum
from app.schemas.token_counts_schema import ITokenCountCreate, TokenCountsLLMResponse
from app.schemas.user_schema import UserData

router = APIRouter()


def get_default_start_end_dates(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    start_date_increment: timedelta = timedelta(days=30),
) -> tuple[date, date]:
    if start_date is None:
        start_date = datetime.now(timezone.utc) - start_date_increment

    if not end_date or end_date > datetime.now(timezone.utc).date():
        logger.debug("end date is in future. Defaulting to current date")
        end_date = datetime.now(timezone.utc)

    return start_date, end_date


@router.get("/workspace/{workspace_id}/tokens")
async def get_chat_llm_token_counts(
    workspace_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    organization_id: Optional[UUID] = None,
    chat_session_id: Optional[UUID] = None,
    chatapp_id: Optional[UUID] = None,
    chat_app_type: str = "unstructured_chat_app",
    start_date: Optional[date] = Query(
        default=None, description="Start date in YYYY-MM-DD format"
    ),
    end_date: Optional[date] = Query(
        default=None, description="End date in YYYY-MM-DD format"
    ),
    current_user: UserData = Depends(
        get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(user_workspace_access_check),
) -> IPostResponseBase[TokenCountsLLMResponse]:

    start_date, end_date = get_default_start_end_dates(
        start_date=start_date,
        end_date=end_date,
        start_date_increment=timedelta(days=30),
    )

    try:
        workspace_token_counts = await token_count.get_workspace_llm_token_counts(
            workspace_id, start_date, db, chat_app_type, end_date
        )
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e

    organization_token_counts = None
    if organization_id:
        try:
            organization_token_counts = (
                await token_count.get_organization_llm_token_counts(
                    organization_id, start_date, db, chat_app_type, end_date
                )
            )
        except HTTPException as e:
            logger.error(f"HTTP Exception: {e.detail}")
            raise e

    chatapp_token_counts = None
    if chatapp_id:
        try:
            chatapp_token_counts = await token_count.get_chatapp_llm_token_counts(
                chatapp_id, start_date, db, chat_app_type, end_date
            )
        except HTTPException as e:
            logger.error(f"HTTP Exception: {e.detail}")
            raise e

    chat_session_token_counts = None
    if chat_session_id:
        try:
            chat_session_token_counts = (
                await token_count.get_chat_session_llm_token_counts(chat_session_id, db)
            )
        except HTTPException as e:
            logger.error(f"HTTP Exception: {e.detail}")
            raise e

    response = TokenCountsLLMResponse(
        workspace_token_counts=workspace_token_counts,
        organization_token_counts=organization_token_counts,
        chat_app_token_counts=chatapp_token_counts,
        chat_session_token_counts=chat_session_token_counts,
    )

    return create_response(
        data=response,
        message="calculated token counts for LLM responses succesfully",
    )


## create a function that calculates the embedding tokens
## Look for if ingestion completed?
## do it on dataset-level, workspace level, org level?
async def _validate_workspace_and_datasets(
    organization_id: UUID,
    workspace_ids: Optional[List[UUID]] = None,
    dataset_ids: Optional[List[UUID]] = None,
    db: AsyncSession = Depends(deps.get_db),
):
    if workspace_ids:
        for workspace_id in workspace_ids:
            try:
                await token_count.workspace_exists_in_org_include_soft_delete(
                    workspace_id, organization_id, db
                )
            except HTTPException as e:
                logger.error(f"HTTP Exception: {e.detail}")
                raise e

    if dataset_ids:
        for dataset_id in dataset_ids:
            try:
                await token_count.dataset_exists_in_org_include_soft_delete(
                    dataset_id, organization_id, db
                )
            except HTTPException as e:
                logger.error(f"HTTP Exception: {e.detail}")
                raise e


@router.post("/organization/{organization_id}/embedding_token_count")
async def get_embedding_token_count(
    organization_id: UUID,
    workspace_ids: Optional[List[UUID]] = None,
    dataset_ids: Optional[List[UUID]] = None,
    db: AsyncSession = Depends(deps.get_db),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.organization_check),
) -> IPostResponseBase[ITokenCountCreate]:

    client = None  # get_r2r_client_sync()
    if not client:
        logger.error("Failed to connect to R2R", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to connect to R2R")

    ## added error handling if an inputted workspace or an dataset is not existent
    await _validate_workspace_and_datasets(
        organization_id, workspace_ids, dataset_ids, db
    )

    r2r_token_map = {}
    for doc in client.documents.list().results:
        r2r_token_map[doc.id] = (
            doc.total_tokens
        )  ## r2r stores as str so converting str everywhere

    org_dataset_ids = await token_count.get_dataset_ids_by_org_no(organization_id, db)
    dataset_token_counts, total_org_tokens = (
        await token_count.calculate_dataset_token_counts(
            org_dataset_ids, r2r_token_map, db
        )
    )

    workspaces_total = 0
    filtered_workspace_token_counts = {}
    if workspace_ids:
        for workspace_id in workspace_ids:
            curr = 0
            curr_dataset_ids = await token_count.get_datasets_for_workspace_no(
                workspace_id, db
            )

            for curr_dataset in curr_dataset_ids:
                if curr_dataset in dataset_token_counts:
                    curr += dataset_token_counts[curr_dataset]

            workspaces_total += curr
            filtered_workspace_token_counts[workspace_id] = workspaces_total

    datasets_total = 0
    filtered_dataset_token_counts = {}
    if dataset_ids:
        for dataset_id in dataset_ids:
            if dataset_id in dataset_token_counts:
                datasets_total += dataset_token_counts[dataset_id]
                filtered_dataset_token_counts[dataset_id] = dataset_token_counts[
                    dataset_id
                ]

    obj_in = ITokenCountCreate(
        organization_id=organization_id,
        org_level_tokens=total_org_tokens,
        workspace_map=filtered_workspace_token_counts,
        dataset_map=filtered_dataset_token_counts,
        workspace_level_tokens=workspaces_total,
        dataset_level_tokens=datasets_total,
    )

    token_counts_record = await token_count.create_token_count(
        obj_in=obj_in, db_session=db
    )

    return create_response(
        data=token_counts_record,
        message="calculated token counts for embeddings succesfully",
    )
