from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.be_core.logger import logger
from app.crud import vanna_training
from app.crud.dataset_crud_v2 import dataset_v2
from app.schemas.response_schema import create_response
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.schemas.vanna_training_schema import IVannaTrainingRead
from app.utils.training_utils import (
    clear_redis_training_locks,
    handle_redis_training_locks,
    prepare_celery_training_task,
    process_question_sql_pairs,
    resolve_database_details,
    setup_vanna_connector_manager,
    store_training_data,
    truncate_embeddings_table,
)

router = APIRouter()


# -------------------------------
# Request/Response Schemas
# -------------------------------
class TrainingResponse(BaseModel):
    message: str
    database_type: str


class QuestionSQLPair(BaseModel):
    question: str
    sql: str


class TrainingRequest(BaseModel):
    documentation: Optional[str] = None
    question_sql_pairs: Optional[List[QuestionSQLPair]] = None


# ===========================================================
# TRAIN API
# ===========================================================
@router.post(
    "/workspace/{workspace_id}/dataset/{dataset_id}/train",
    response_model=TrainingResponse,
)
async def train_model(
    workspace_id: UUID,
    dataset_id: UUID,
    request_data: TrainingRequest,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    _=Depends(deps.dataset_check),
    db_session: AsyncSession = Depends(deps.get_db),
):
    logger.info(
        f"Training request received for dataset {dataset_id} by user {current_user.id}"
    )

    # Update ingestion_status to 'processing' at the start
    try:
        await dataset_v2.update_ingestion_status(
            dataset_id=dataset_id, status="processing", db_session=db_session
        )
        logger.info(f"Updated dataset {dataset_id} ingestion_status to 'processing'")
    except Exception as e:
        logger.warning(f"Failed to update ingestion status to processing: {e}")

    try:
        # Handle Redis-based training locks and duplicate detection
        request_data_hash = hash(str(request_data.dict()))
        await handle_redis_training_locks(
            dataset_id, current_user, str(request_data_hash)
        )

        # Resolve database details from dataset_id
        db_details, db_type, vector_db_name = await resolve_database_details(
            dataset_id, db_session
        )

        # Setup VannaConnectorManager (for any initialization needed)
        await setup_vanna_connector_manager(vector_db_name, "gpt-4o")

        # Process question-SQL pairs
        question_sql_pairs = process_question_sql_pairs(request_data.question_sql_pairs)

        # Store training data with version 1 (default for train API)
        await store_training_data(
            dataset_id=dataset_id,
            documentation=request_data.documentation,
            question_sql_pairs=question_sql_pairs,
            version_id=1,
            db_session=db_session,
        )

        # Prepare and execute Celery training task
        prepare_celery_training_task(
            db_details=db_details,
            db_type=db_type,
            documentation=request_data.documentation,
            question_sql_pairs=question_sql_pairs,
            vector_db_name=vector_db_name,
            user_id=str(current_user.id),
            dataset_id=str(dataset_id),
        )

        return TrainingResponse(
            message=f"Model training initiated successfully for {db_type} database",
            database_type=db_type,
        )

    except HTTPException:
        # Re-raise HTTP exceptions from utilities
        raise
    except Exception as e:
        try:
            await dataset_v2.update_ingestion_status(
                dataset_id=dataset_id, status="failed", db_session=db_session
            )
        except Exception:  # nosec
            pass
        logger.error("Error during training", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================
# RETRAIN API
# ==========================================================
@router.post(
    "/workspace/{workspace_id}/dataset/{dataset_id}/retrain",
)
async def retrain_model(
    workspace_id: UUID,
    dataset_id: UUID,
    request_data: TrainingRequest,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    _=Depends(deps.dataset_check),
    db_session: AsyncSession = Depends(deps.get_db),
):
    logger.info(f"🔄 Retraining requested for dataset {dataset_id}")

    try:
        # 1. Soft delete old entries from vanna_trainings
        try:
            deleted = await vanna_training.soft_delete_by_dataset(
                db_session, dataset_id=dataset_id
            )
            logger.info(
                f"🗑 Soft deleted {deleted} old training entries for dataset {dataset_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to soft delete old training entries: {e}")

        # 2. Get next version for this dataset
        try:
            next_version = await vanna_training.get_next_version_for_dataset(
                db_session, dataset_id=dataset_id
            )
            logger.info(f"📝 Using version {next_version} for dataset {dataset_id}")
        except Exception as e:
            logger.warning(f"Failed to get next version: {e}")
            next_version = 1

        # 3. Resolve database details from dataset_id
        db_details, db_type, vector_db_name = await resolve_database_details(
            dataset_id, db_session
        )

        # 4. Truncate embeddings table
        await truncate_embeddings_table(vector_db_name)

        # 5. Clear Redis training locks to force retrain
        await clear_redis_training_locks(dataset_id, current_user)

        # 6. Process question-SQL pairs
        question_sql_pairs = process_question_sql_pairs(request_data.question_sql_pairs)

        # 7. Store new training data with version
        await store_training_data(
            dataset_id=dataset_id,
            documentation=request_data.documentation,
            question_sql_pairs=question_sql_pairs,
            version_id=next_version,
            db_session=db_session,
        )

        # 8. Prepare and execute Celery training task
        prepare_celery_training_task(
            db_details=db_details,
            db_type=db_type,
            documentation=request_data.documentation,
            question_sql_pairs=question_sql_pairs,
            vector_db_name=vector_db_name,
            user_id=str(current_user.id),
            dataset_id=str(dataset_id),
        )

        logger.info(f"🚀 Started retraining task for dataset {dataset_id}")

        # Return response using create_response
        return create_response(
            data=TrainingResponse(
                message=f"Model retraining initiated successfully for dataset {dataset_id} (version {next_version})",
                database_type=db_type,
            ),
            message="Retraining process started successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions from utilities
        raise
    except Exception as e:
        logger.error(f"Failed to retrain model for dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start retraining: {str(e)}"
        )


# ===========================================================
# GET TRAINING DETAILS API
# ===========================================================
@router.get(
    "/workspace/{workspace_id}/dataset/{dataset_id}/trainings",
)
async def get_training_details(
    workspace_id: UUID,
    dataset_id: UUID,
    current_user: UserData = Depends(deps.get_current_user),
    _=Depends(deps.dataset_check),
    db_session: AsyncSession = Depends(deps.get_db),
):
    try:
        # Get only the latest version training records
        trainings = await vanna_training.get_max_version_by_dataset(
            db_session, dataset_id=dataset_id
        )

        # Convert to Pydantic models for proper serialization
        training_list = [
            IVannaTrainingRead(
                id=training.id,
                dataset_id=training.dataset_id,
                documentation=training.documentation,
                question_sql_pairs=training.question_sql_pairs,
                version_id=training.version_id,
                created_at=training.created_at,
            )
            for training in trainings
        ]

        return create_response(
            data=training_list,
            message=f"Retrieved {len(training_list)} training records for dataset {dataset_id} (latest version)",
        )

    except Exception as e:
        logger.error(f"Error retrieving training details for dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve training details: {str(e)}"
        )


# ===========================================================
# GET ALL VERSIONS API
# ===========================================================
@router.get(
    "/workspace/{workspace_id}/dataset/{dataset_id}/trainings/all-versions",
)
async def get_all_training_versions(
    workspace_id: UUID,
    dataset_id: UUID,
    current_user: UserData = Depends(deps.get_current_user),
    _=Depends(deps.dataset_check),
    db_session: AsyncSession = Depends(deps.get_db),
):
    try:
        # Get ALL training records for the dataset (including soft-deleted ones)
        trainings = await vanna_training.get_all_by_dataset_including_deleted(
            db_session, dataset_id=dataset_id
        )

        # Group by version_id and organize the data
        version_groups = {}
        for training in trainings:
            version_id = training.version_id
            if version_id not in version_groups:
                version_groups[version_id] = []

            version_groups[version_id].append(
                {
                    "id": training.id,
                    "dataset_id": training.dataset_id,
                    "documentation": training.documentation,
                    "question_sql_pairs": training.question_sql_pairs,
                    "version_id": training.version_id,
                    "created_at": training.created_at,
                    "is_deleted": training.deleted_at is not None,
                    "deleted_at": training.deleted_at,
                }
            )

        # Sort versions in descending order (latest first) and simplify
        sorted_versions = []
        for version_id in sorted(version_groups.keys(), reverse=True):
            sorted_versions.append(
                {
                    "version_id": version_id,
                    "training_records": version_groups[version_id],
                }
            )

        return create_response(
            data=sorted_versions,
            message=f"Retrieved all {len(sorted_versions)} versions for dataset {dataset_id}",
        )

    except Exception as e:
        logger.error(
            f"Error retrieving all training versions for dataset {dataset_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve training versions: {str(e)}"
        )
