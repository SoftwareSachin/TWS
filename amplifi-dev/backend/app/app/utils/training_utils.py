"""
Training utilities for common functionality between train and retrain endpoints.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_redis_client
from app.be_core.celery import celery
from app.be_core.logger import logger
from app.be_core.vanna_connector_manager import (
    VannaConnectorManager,
    build_vector_db_connection_string,
)
from app.crud import crud_source, vanna_training
from app.crud.dataset_crud_v2 import dataset_v2
from app.db.connection_handler import DatabaseConnectionHandler
from app.schemas.user_schema import UserData
from app.utils.openai_utils import generate_text2sql_embedding


# -------------------------------
# Embedding class (OpenAI backend)
# -------------------------------
class OpenAIHuggingFaceEmbeddings:
    """OpenAI-based embeddings implementation for compatibility."""

    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text query using OpenAI"""
        return generate_text2sql_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents using OpenAI"""
        return [generate_text2sql_embedding(text) for text in texts]


# Ensure the mock module is available
sys.modules["langchain_huggingface"] = type(
    "MockModule", (), {"HuggingFaceEmbeddings": OpenAIHuggingFaceEmbeddings}
)()


async def resolve_database_details(
    dataset_id: UUID, db_session: AsyncSession
) -> Tuple[Dict[str, Any], str, str]:
    """
    Resolve database details from dataset_id.

    Returns:
        Tuple of (db_details, db_type, vector_db_name)

    Raises:
        HTTPException: If source_id is missing, conflicting DB configs, or no DB details found
    """
    # Get source_id from dataset
    source_id = await dataset_v2.get_source_id_by_dataset_id(dataset_id=dataset_id)
    if not source_id:
        raise HTTPException(status_code=400, detail="source_id missing.")

    # Try to get MySQL and PostgreSQL details
    mysql_details = await crud_source.get_sql_db_details(
        source_id=source_id, db_type="mysql_db", db_session=db_session
    )
    postgres_detail = await crud_source.get_sql_db_details(
        source_id=source_id, db_type="pg_db", db_session=db_session
    )

    # Handle conflicts and missing details
    if mysql_details and postgres_detail:
        raise HTTPException(status_code=409, detail="Conflicting DB config.")
    if not mysql_details and not postgres_detail:
        raise HTTPException(status_code=404, detail="No DB details found.")

    # Determine which details to use
    db_details = mysql_details or postgres_detail
    db_type = DatabaseConnectionHandler.get_database_type_from_details(db_details)
    vector_db_name = f"{db_details['database_name']}_{dataset_id}"

    logger.info(
        f"Resolved database details: type={db_type}, vector_db={vector_db_name}"
    )
    return db_details, db_type, vector_db_name


def process_question_sql_pairs(
    question_sql_pairs: Optional[List[Any]],
) -> List[Dict[str, str]]:
    """
    Process question-SQL pairs from request format to storage format.

    Args:
        question_sql_pairs: List of question-SQL pair objects from request

    Returns:
        List of dictionaries with 'question' and 'sql' keys
    """
    if not question_sql_pairs:
        return []

    processed_pairs = [
        {"question": pair.question, "sql": pair.sql} for pair in question_sql_pairs
    ]
    logger.debug(f"Processed {len(processed_pairs)} question-SQL pairs")
    return processed_pairs


async def store_training_data(
    dataset_id: UUID,
    documentation: Optional[str],
    question_sql_pairs: List[Dict[str, str]],
    version_id: int,
    db_session: AsyncSession,
) -> None:
    """
    Store training data in the vanna_trainings table.

    Args:
        dataset_id: ID of the dataset
        documentation: Documentation text
        question_sql_pairs: Processed question-SQL pairs
        version_id: Version ID for the training data
        db_session: Database session

    Raises:
        HTTPException: If storage fails
    """
    try:
        await vanna_training.create(
            db_session,
            obj_in={
                "dataset_id": dataset_id,
                "documentation": documentation,
                "question_sql_pairs": (
                    question_sql_pairs if question_sql_pairs else None
                ),
                "version_id": version_id,
            },
        )
        logger.info(
            f"Stored training data for dataset {dataset_id} with version {version_id}"
        )
    except Exception as e:
        logger.error(f"Failed to store training data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to store training data: {str(e)}"
        )


def prepare_celery_training_task(
    db_details: Dict[str, Any],
    db_type: str,
    documentation: Optional[str],
    question_sql_pairs: List[Dict[str, str]],
    vector_db_name: str,
    user_id: str,
    dataset_id: str,
) -> None:
    """
    Prepare and execute the Celery training task.

    Args:
        db_details: Database connection details
        db_type: Type of database (mysql/postgresql)
        documentation: Documentation text
        question_sql_pairs: Question-SQL pairs
        vector_db_name: Name of the vector database
        user_id: ID of the user
        dataset_id: ID of the dataset
    """
    try:
        celery.signature(
            "task.vanna_training_task",
            kwargs={
                "db_details": db_details,
                "db_type": db_type,
                "documentation": documentation,
                "question_sql_pairs": question_sql_pairs,
                "database_name": vector_db_name,
                "user_id": user_id,
                "dataset_id": dataset_id,
            },
            queue="training_vanna_queue",
        ).apply_async()

        logger.info(f"Celery training task started for dataset {dataset_id}")
    except Exception as e:
        logger.error(f"Failed to start training task: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start training task: {str(e)}"
        )


async def handle_redis_training_locks(
    dataset_id: UUID, current_user: UserData, request_data_hash: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle Redis-based training locks and duplicate detection.

    Args:
        dataset_id: ID of the dataset
        current_user: Current user data
        request_data_hash: Hash of the request data for duplicate detection

    Returns:
        Tuple of (processing_key, request_key) or (None, None) if Redis unavailable

    Raises:
        HTTPException: If training is already in progress or duplicate request detected
    """
    redis_client = None
    processing_key = None
    request_key = None

    try:
        redis_client = await get_redis_client()
        processing_key = f"training:processing:{dataset_id}:{current_user.id}"
        request_key = (
            f"training:request:{dataset_id}:{current_user.id}:{request_data_hash}"
        )

        # Check if training is already in progress
        processing_exists = await redis_client.exists(processing_key)
        if processing_exists:
            processing_info = await redis_client.get(processing_key)
            processing_info_str = (
                processing_info.decode()
                if isinstance(processing_info, bytes)
                else str(processing_info)
            )
            logger.warning(
                f"Dataset {dataset_id} is already being trained by {processing_info_str}"
            )
            raise HTTPException(
                status_code=409,
                detail="Dataset is currently being trained. Please wait for completion.",
            )

        # Check for duplicate requests
        request_exists = await redis_client.exists(request_key)
        if request_exists:
            logger.warning(
                f"Duplicate training request detected for dataset {dataset_id}"
            )
            raise HTTPException(
                status_code=409,
                detail="Identical training request already submitted recently",
            )

        # Set both flags atomically
        await redis_client.setex(
            processing_key, 3600, f"user:{current_user.id}"
        )  # 1 hour
        await redis_client.setex(request_key, 300, "processing")  # 5 minutes

        logger.debug(
            f"Set training flags - processing: {processing_key}, request: {request_key}"
        )

        # Notify user via WebSocket
        await redis_client.publish(
            f"{current_user.id}:train_status", "🚀 Training started..."
        )

        return processing_key, request_key

    except HTTPException:
        # Re-raise HTTP exceptions (these are business logic errors)
        raise
    except Exception as e:
        logger.warning(f"Redis deduplication not available: {str(e)}")
        return None, None
    finally:
        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")


async def clear_redis_training_locks(dataset_id: UUID, current_user: UserData) -> None:
    """
    Clear Redis training locks for retrain operations.

    Args:
        dataset_id: ID of the dataset
        current_user: Current user data
    """
    redis_client = None
    try:
        redis_client = await get_redis_client()
        processing_key = f"training:processing:{dataset_id}:{current_user.id}"

        # Delete processing lock
        await redis_client.delete(processing_key)
        logger.info(f"🔓 Cleared Redis lock {processing_key}")

    except Exception as e:
        logger.warning(f"Could not clear Redis lock: {e}")
    finally:
        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.warning(f"Could not fetch Redis client: {e}")
                pass


async def truncate_embeddings_table(vector_db_name: str) -> None:
    """
    Truncate the embeddings table for a vector database.

    Args:
        vector_db_name: Name of the vector database
    """
    try:
        engine = create_engine(
            build_vector_db_connection_string(vector_db_name),
            isolation_level="AUTOCOMMIT",
        )
        with engine.connect() as conn:
            conn.execute(
                text(
                    "TRUNCATE TABLE public.langchain_pg_embedding RESTART IDENTITY CASCADE;"
                )
            )
        logger.info(f"🧹 Truncated embeddings table for vector DB {vector_db_name}")
    except Exception as e:
        logger.warning(f"Failed to truncate embeddings table for {vector_db_name}: {e}")


async def setup_vanna_connector_manager(
    vector_db_name: str, llm_model: str = "gpt-4o"
) -> VannaConnectorManager:
    """
    Setup VannaConnectorManager for training operations.

    Args:
        vector_db_name: Name of the vector database
        llm_model: LLM model to use

    Returns:
        VannaConnectorManager instance
    """
    manager = VannaConnectorManager(source_db_name=vector_db_name, llm_model=llm_model)
    logger.debug(f"Created VannaConnectorManager for {vector_db_name}")
    return manager
