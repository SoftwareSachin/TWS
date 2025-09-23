from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import dataset_check, get_current_user, get_db
from app.be_core.celery import celery
from app.crud import extracted_entity_crud
from app.crud.graph_crud import crud_graph
from app.models.graph_model import GraphStatus
from app.schemas.graph_schema import (
    ExtractedEntityRead,
    GraphBase,
    GraphCreate,
    GraphEntitiesRelationshipsResponse,
    GraphRead,
    GraphReadEntityTypes,
    IGraphCreateEntities,
)
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.graph.extract import GraphReader

router = APIRouter()


@router.post(
    "/dataset/{dataset_id}/graph",
    response_model=IPostResponseBase[GraphRead],
)
async def create_graph_for_dataset(
    dataset_id: UUID,
    request_data: GraphBase,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
    current_user: UserData = Depends(get_current_user()),
):
    """
    Create a graph for the given dataset.
    """
    graph = await crud_graph.create_graph(
        obj_in=GraphCreate(
            dataset_id=dataset_id,
            entity_types=[],  # TODO: refactor this better, graphCreate should just not have entity types
        ),
        db_session=db,
    )
    return create_response(data=graph, message="Graph created successfully")


@router.post(
    "/dataset/{dataset_id}/graph/{graph_id}/entities",
    response_model=IPostResponseBase[GraphRead],
)
async def extract_graph_entities(
    dataset_id: UUID,
    graph_id: UUID,
    request_data: IGraphCreateEntities,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
    current_user: UserData = Depends(
        get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
):
    """
    Extract a graph for the given dataset.
    """
    # Update entities status to PENDING before starting extraction
    await crud_graph.update_graph_status(
        graph_id=graph_id,
        new_status="pending",
        db_session=db,
        field="entities",
    )

    # Start celery task for extraction and graph building (Kuzu version)
    celery.signature(
        "tasks.extract_text_from_dataset",
        kwargs={
            "dataset_id": str(dataset_id),
            "graph_id": str(graph_id),
            "entity_types": request_data.entity_types,
        },
    ).apply_async()
    graph = await crud_graph.get_graph(graph_id=graph_id, db_session=db)
    graph.entity_types = request_data.entity_types
    return create_response(
        data=graph, message="Graph entities extraction started (status: PENDING)"
    )


@router.get(
    "/dataset/{dataset_id}/graph/{graph_id}/entity-types",
    response_model=IGetResponseBase[GraphReadEntityTypes],
)
async def get_graph_entities_extraction_status(
    dataset_id: UUID,
    graph_id: UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
):
    """
    Get the entity types for a graph.
    """
    graph = await crud_graph.get_graph(graph_id=graph_id, db_session=db)
    if graph.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Graph not found for this dataset")
    graph_entity_types = await extracted_entity_crud.get_entity_type_counts(
        graph_id=graph_id, db_session=db
    )
    return create_response(
        data=GraphReadEntityTypes(entity_types=graph_entity_types),
        message="Graph entity types fetched successfully",
    )


@router.get(
    "/dataset/{dataset_id}/graph/{graph_id}/entities",
    response_model=IGetResponseBase[list[ExtractedEntityRead]],
)
async def get_graph_entities(
    dataset_id: UUID,
    graph_id: UUID,
    entity_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
):
    """
    Get the entities for a graph.
    """
    entities = await extracted_entity_crud.get_entities_info(
        graph_id=graph_id, db_session=db, entity_type=entity_type
    )
    return create_response(data=entities, message="Graph entities fetched successfully")


@router.post(
    "/dataset/{dataset_id}/graph/{graph_id}/relationships",
    response_model=IPostResponseBase[GraphRead],
)
async def extract_graph_relationships_and_save(
    dataset_id: UUID,
    graph_id: UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
):
    """
    Extract relationships and save them to the kuzu graph.
    """
    celery.signature(
        "tasks.kuzu_relationships_extraction_task",
        kwargs={"dataset_id": str(dataset_id), "graph_id": str(graph_id)},
    ).apply_async()
    await crud_graph.update_graph_status(
        graph_id=graph_id,
        new_status="pending",
        db_session=db,
        field="relationships",
    )
    return create_response(
        data=None, message="Graph relationships extraction started (status: PENDING)"
    )


@router.delete(
    "/dataset/{dataset_id}/graph/{graph_id}/entity/{entity_id}",
    response_model=IDeleteResponseBase[dict],
)
async def delete_graph_entity_by_id(
    dataset_id: UUID,
    graph_id: UUID,
    entity_id: UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
    current_user: UserData = Depends(get_current_user()),
):
    """
    Delete a single extracted entity by its ID for the specified graph and dataset.
    """
    graph = await crud_graph.get_graph(graph_id=graph_id, db_session=db)
    if graph.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Graph not found for this dataset")

    await extracted_entity_crud.delete_entity(entity_id=entity_id, db_session=db)

    return create_response(
        data={"deleted": True, "entity_id": str(entity_id)},
        message="Graph entity deleted successfully",
    )


@router.delete(
    "/dataset/{dataset_id}/graph/{graph_id}/entity",
    response_model=IDeleteResponseBase[dict],
)
async def delete_graph_entities_by_types(
    dataset_id: UUID,
    graph_id: UUID,
    entity_types: list[str] = Query(..., alias="types"),
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
    current_user: UserData = Depends(get_current_user()),
):
    """
    Delete all extracted entities of the given types for the specified graph and dataset.
    """
    graph = await crud_graph.get_graph(graph_id=graph_id, db_session=db)
    if graph.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Graph not found for this dataset")

    total_deleted = 0
    deletion_results = {}

    for entity_type in entity_types:
        deleted_count = await extracted_entity_crud.delete_by_type(
            graph_id=graph_id, entity_type=entity_type, db_session=db
        )
        deletion_results[entity_type] = deleted_count
        total_deleted += deleted_count

    return create_response(
        data={
            "total_deleted": total_deleted,
            "deletion_results": deletion_results,
            "entity_types": entity_types,
        },
        message="Graph entities deleted successfully",
    )


@router.get(
    "/dataset/{dataset_id}/graph/{graph_id}",
    response_model=IGetResponseBase[GraphRead],
)
async def get_graph(
    dataset_id: UUID,
    graph_id: UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
):
    """
    Retrieve a graph by its ID for a specific dataset.
    """
    graph = await crud_graph.get_graph(graph_id=graph_id, db_session=db)
    if graph.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Graph not found for this dataset")
    return create_response(data=graph, message="Graph fetched successfully")


@router.get(
    "/dataset/{dataset_id}/graph",
    response_model=IGetResponseBase[GraphRead],
)
async def get_most_recent_graph(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
):
    """
    Retrieve most recent graph for a specific dataset.
    """
    graph = await crud_graph.get_most_recent_graph(dataset_id=dataset_id, db_session=db)
    return create_response(data=graph, message="Graph fetched successfully")


@router.delete(
    "/dataset/{dataset_id}/graph/{graph_id}",
    response_model=IDeleteResponseBase[GraphRead],
)
async def delete_graph(
    dataset_id: UUID,
    graph_id: UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
    current_user: UserData = Depends(get_current_user()),
):
    """
    Soft delete a graph by setting its deleted_at timestamp.
    """
    # First verify the graph exists and belongs to the dataset
    graph = await crud_graph.get_graph(graph_id=graph_id, db_session=db)
    if graph.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Graph not found for this dataset")

    # Soft delete the graph
    deleted_graph = await crud_graph.soft_delete_graph(graph_id=graph_id, db_session=db)

    return create_response(
        data=deleted_graph, message="Graph soft deleted successfully"
    )


# TODO: tune this based on frontend. Need GraphCheck dep function.
@router.get(
    "/dataset/{dataset_id}/graph/{graph_id}/entities-relationships",
    response_model=IGetResponseBase[GraphEntitiesRelationshipsResponse],
)
async def get_graph_entities_relationships(
    dataset_id: UUID,
    graph_id: UUID,
    limit: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    _=Depends(dataset_check),
):
    """
    Retrieve entities and relationships for a specific graph by its ID.

    Args:
        graph_id: The ID of the graph
        limit: Maximum number of entities and relationships to return (optional, if not provided returns all)
    """
    graph = await crud_graph.get_graph(graph_id=graph_id, db_session=db)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    if graph.relationships_status != GraphStatus.SUCCESS:
        raise HTTPException(status_code=404, detail="Graph not ready")
    try:
        graph = GraphReader(str(graph_id))
        response_data = graph.get_entities_and_relationships(limit=limit)
        return create_response(
            data=response_data,
            message="Graph entities and relationships fetched successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error loading graph: {str(e)}")
