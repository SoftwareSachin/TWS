from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Params

from app import crud
from app.api import deps
from app.be_core.logger import logger
from app.crud.ingest_crud import ingestion_crud
from app.schemas.common_schema import IOrderEnum
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    IPutResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.schemas.workflow_schema import (
    IWorkflowCreate,
    IWorkflowExecute,
    IWorkflowExecutionRead,
    IWorkflowRead,
    IWorkflowRunRead,
    IWorkflowUpdate,
)

router = APIRouter()


@router.post("/organization/{organization_id}/workflow")
async def add_workflow(
    organization_id: UUID,
    workflow: IWorkflowCreate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
) -> IPostResponseBase[IWorkflowRead]:
    """
    Adds a new workflow to an organization.

    Required roles:
    - admin
    - developer

    Note:
    - In workflow create, schedule config must be a cron expression. if not, workflow will not be schedule and will raise an error.
    """
    new_workflow = await crud.workflow.create_workflow(
        obj_in=workflow,
        organization_id=organization_id,
    )

    return create_response(data=new_workflow)


@router.get("/organization/{organization_id}/workflow")
async def get_all_workflows(
    organization_id: UUID,
    params: Params = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.organization_check),
) -> IGetResponsePaginated[IWorkflowRead]:
    """
    Retrieves all workflows for an organization.

    Required roles:
    - admin
    - member
    - developer
    """
    paginate_workflows = await crud.workflow.get_all_workflows(
        organization_id=organization_id, params=params, order=order
    )
    pagninate_response = IGetResponsePaginated.create(
        items=paginate_workflows.items, total=paginate_workflows.total, params=params
    )

    return create_response(data=pagninate_response)


@router.post("/organization/{organization_id}/workflow/{workflow_id}/execute")
async def execute_workflow(
    organization_id: UUID,
    workflow_id: UUID,
    workflow_execute: Optional[IWorkflowExecute] = None,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
) -> IPostResponseBase[IWorkflowExecutionRead]:
    """
    Executes a specific workflow.

    Required roles:
    - admin
    - developer

    Corner Cases:
    - Ensure there is at least one file in the dataset before execution.
    - Check if all files in the dataset are ingested before proceeding.
    """

    # Check if there is at least one file in the dataset
    dataset_files = await crud.workflow.get_files_by_workflow_id(
        workflow_id=workflow_id, organization_id=organization_id
    )

    logger.info(f"files present in dataset : {dataset_files}")

    # Check the ingestion status for all files in the dataset
    workflow = await crud.workflow.get_workflow_by_id(
        workflow_id=workflow_id, organization_id=organization_id
    )
    dataset_id = workflow.dataset_id

    logger.info(
        f"Now checking ingestion status of the files present in the dataset for dataset_id : {dataset_id}"
    )

    flag_all_ingested = await ingestion_crud.are_all_files_ingested(
        dataset_id=dataset_id
    )

    logger.info(f"Ingestion status for dataset {dataset_id}: {flag_all_ingested}")

    if not flag_all_ingested:
        raise HTTPException(
            status_code=409,
            detail="All files are not ingested yet. Please wait for the ingestion to complete.",
        )

    logger.debug(f"all files ingested for dataset id {dataset_id}: {flag_all_ingested}")

    status = await crud.workflow.execute_workflow(
        workflow_id=workflow_id,
        organization_id=organization_id,
        user_id=current_user.id,
    )

    return create_response(data={"status": status})


@router.get("/organization/{organization_id}/workflow/{workflow_id}/runs")
async def get_workflow_run_history(
    organization_id: UUID,
    workflow_id: UUID,
    params: Params = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.organization_check),
) -> IGetResponsePaginated[IWorkflowRunRead]:
    """
    Retrieves the run history of a workflow.

    Required roles:
    - admin
    - member
    - developer
    """
    pagninate_workflow_runs = await crud.workflow.get_workflow_run_history(
        workflow_id=workflow_id,
        organization_id=organization_id,
    )

    pagninate_response = IGetResponsePaginated.create(
        items=pagninate_workflow_runs.items,
        total=pagninate_workflow_runs.total,
        params=params,
    )

    return create_response(data=pagninate_response)


@router.put("/organization/{organization_id}/workflow/{workflow_id}")
async def update_workflow(
    organization_id: UUID,
    workflow_id: UUID,
    workflow_update: IWorkflowUpdate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
) -> IPutResponseBase[IWorkflowRead]:
    """
    Updates a workflow.

    Required roles:
    - admin
    - developer
    """
    updated_workflow = await crud.workflow.update_workflow(
        workflow_id=workflow_id,
        organization_id=organization_id,
        obj_in=workflow_update,
    )

    return create_response(data=updated_workflow)


@router.get("/organization/{organization_id}/workflow/{workflow_id}")
async def get_workflow_by_id(
    organization_id: UUID,
    workflow_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.organization_check),
) -> IGetResponseBase[IWorkflowRead]:
    """
    Retrieves a workflow by its ID.

    Required roles:
    - admin
    - member
    - developer
    """
    workflow = await crud.workflow.get_workflow_by_id(
        workflow_id=workflow_id,
        organization_id=organization_id,
    )
    return create_response(data=workflow)


@router.delete("/organization/{organization_id}/workflow/{workflow_id}")
async def remove_workflow(
    organization_id: UUID,
    workflow_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
) -> IDeleteResponseBase[IWorkflowRead]:
    """
    Removes a workflow.

    Required roles:
    - admin
    - developer
    """
    deleted_workflow = await crud.workflow.remove_workflow(
        workflow_id=workflow_id,
        organization_id=organization_id,
    )

    return create_response(data=deleted_workflow)


@router.post("/organization/{organization_id}/workflow/{workflow_id}/stop")
async def stop_workflow(
    workflow_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
) -> IPostResponseBase[None]:
    """
    Stops a workflow.

    Required roles:
    - admin
    - developer
    """
    await crud.workflow.stop_workflow(workflow_id=workflow_id)
    return create_response(message="Workflow stopped successfully", data=None)


@router.post("/organization/{organization_id}/workflow/{workflow_id}/start")
async def start_workflow(
    workflow_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.organization_check),
) -> IPostResponseBase[None]:
    """
    Starts a workflow.

    Required roles:
    - admin
    - developer
    """
    await crud.workflow.start_workflow(
        workflow_id=workflow_id,
    )
    return create_response(message="Workflow started successfully", data=None)
