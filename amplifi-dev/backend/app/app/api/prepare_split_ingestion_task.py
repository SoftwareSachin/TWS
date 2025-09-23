"""
Celery task for preparing file splitting and deciding whether to split files for ingestion.
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.ingest_crud import ingestion_crud
from app.db.session import SyncSessionLocal
from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType
from app.models.file_split_model import FileSplit, SplitFileStatusType
from app.utils.ingestion_utils import (
    check_file_needs_splitting,
    check_splits_valid,
    clean_existing_splits,
    create_splits_for_file,
    get_splitting_config_hash,
    update_file_ingestion_status,
    update_task_id,
)


def _get_file_ingestion_record(
    db, file_id: str, ingestion_id: str, dataset_id: str
) -> Optional[FileIngestion]:
    """
    Helper function to get and validate the file ingestion record.

    Args:
        db: Database session
        file_id: ID of the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset

    Returns:
        The file ingestion record or None if not found
    """
    # First check if file ingestion record exists
    file_ingestion = (
        db.query(FileIngestion)
        .filter(
            FileIngestion.file_id == UUID(file_id),
            FileIngestion.ingestion_id == UUID(ingestion_id),
            FileIngestion.dataset_id == UUID(dataset_id),
        )
        .first()
    )

    if not file_ingestion:
        error_msg = (
            f"No file ingestion record found for file {file_id} "
            f"in ingestion {ingestion_id} for dataset {dataset_id}"
        )
        logger.error(error_msg)

        # Try to create a record if it doesn't exist
        try:
            logger.info(
                f"Attempting to create missing file ingestion record for file {file_id}"
            )
            file_ingestion = FileIngestion(
                file_id=UUID(file_id),
                ingestion_id=UUID(ingestion_id),
                dataset_id=UUID(dataset_id),
                status=FileIngestionStatusType.Processing,
            )
            db.add(file_ingestion)
            db.commit()
            db.refresh(file_ingestion)
            logger.info(f"Created missing file ingestion record for file {file_id}")
            return file_ingestion
        except Exception as e:
            logger.error(f"Failed to create missing file ingestion record: {str(e)}")
            return None

    return file_ingestion


def _process_direct_ingestion(
    db,
    file_id: str,
    file_path: str,
    doc_id: str,
    file_metadata: dict,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    organization_id: Optional[str],
    chunking_config: Optional[dict],
    skip_successful_files: bool,
    config_hash: str,
    task_id: str,
) -> Dict[str, Any]:
    """
    Helper function to process a file directly without splitting.

    Args:
        db: Database session
        file_id: ID of the file
        file_path: Path to the file
        doc_id: Document ID
        file_metadata: Metadata for the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        organization_id: ID of the organization
        chunking_config: Configuration for chunking
        skip_successful_files: Whether to skip successful files
        user_id: ID of the user who initiated the ingestion
        config_hash: Hash of the current splitting configuration
        task_id: ID of the task

    Returns:
        Dictionary with results of the direct ingestion
    """
    logger.info(f"File {file_id} does not need splitting, processing directly")

    # Check for and clean existing splits
    existing_splits = (
        db.query(FileSplit).filter(FileSplit.original_file_id == UUID(file_id)).all()
    )

    if existing_splits:
        logger.info(
            f"File no longer needs splitting but has {len(existing_splits)} old splits - cleaning up"
        )
        clean_existing_splits(db, UUID(file_id), UUID(dataset_id))

    celery.signature(
        "tasks.ingest_files_task",
        kwargs={
            "file_path": file_path,
            "document_id": doc_id,
            "metadata": file_metadata,
            "chunking_config": chunking_config,
            "ingestion_id": ingestion_id,
            "dataset_id": dataset_id,
            "organization_id": organization_id,
            "skip_successful_files": skip_successful_files,
            "user_id": user_id,
        },
    ).apply_async()

    return {
        "file_id": file_id,
        "doc_id": doc_id,
        "ingestion_id": ingestion_id,
        "task_id": task_id,
        "success": True,
        "needs_splitting": False,
        "config_hash": config_hash,
        "message": "File does not need splitting, using regular ingestion",
    }


def _prepare_splits(
    db,
    file_id: str,
    file_path: str,
    file_ingestion_id: UUID,
    dataset_id: str,
    splits_valid: bool,
) -> List[FileSplit]:
    """
    Helper function to prepare splits for a file.

    Args:
        db: Database session
        file_id: ID of the file
        file_path: Path to the file
        file_ingestion_id: ID of the file ingestion
        dataset_id: ID of the dataset
        splits_valid: Whether existing splits are valid

    Returns:
        List of file splits
    """
    # Check if splits already exist for this specific file+dataset combination
    existing_splits = (
        db.query(FileSplit)
        .filter(
            FileSplit.original_file_id == UUID(file_id),
            FileSplit.dataset_id == UUID(dataset_id),
        )
        .all()
    )

    if existing_splits and splits_valid:
        # Use existing splits for this dataset
        logger.info(
            f"Using existing valid splits for file {file_id} in dataset {dataset_id}"
        )
        return existing_splits

    if existing_splits and not splits_valid:
        # Clean up existing splits for this dataset using clean_existing_splits
        logger.info(
            f"File {file_id} needs splitting and existing splits for dataset {dataset_id} are invalid - cleaning up"
        )
        # Use the clean_existing_splits with dataset_id parameter
        clean_existing_splits(db=db, file_id=UUID(file_id), dataset_id=UUID(dataset_id))

    # Create new splits
    logger.info(
        f"Creating new splits for file {file_id} in dataset {dataset_id} with current configuration"
    )
    return create_splits_for_file(
        db=db,
        file_id=UUID(file_id),
        file_path=file_path,
        file_ingestion_id=file_ingestion_id,
        dataset_id=UUID(dataset_id),
    )


def _schedule_split_ingestions(
    db,
    splits: List[FileSplit],
    original_file_id: str,
    file_metadata: dict,
    ingestion_id: str,
    dataset_id: str,
    organization_id: Optional[str],
    chunking_config: Optional[dict],
    skip_successful_files: bool,
    config_hash: str,
    user_id: UUID,
) -> None:
    """
    Helper function to schedule ingestion tasks for each split.

    Args:
        db: Database session
        splits: List of file splits
        original_file_id: ID of the original file
        file_metadata: Metadata for the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        organization_id: ID of the organization
        chunking_config: Configuration for chunking
        skip_successful_files: Whether to skip successful files
        config_hash: Hash of the current splitting configuration
        user_id: ID of the user who initiated the ingestion
    """

    # Add metadata for original document to each split
    split_metadata = file_metadata.copy() if file_metadata else {}
    split_metadata["is_split"] = True
    split_metadata["original_file_id"] = original_file_id

    # Schedule split ingestion tasks for each split
    for split in splits:
        # Create a deterministic ID for this split based on the original doc_id
        split_namespace = f"{original_file_id}_{split.id}_{dataset_id}"
        split_doc_id = str(uuid5(NAMESPACE_URL, split_namespace))

        # Add split-specific metadata
        this_split_metadata = split_metadata.copy()
        this_split_metadata["split_id"] = str(split.id)
        this_split_metadata["split_index"] = split.split_index
        this_split_metadata["total_splits"] = split.total_splits
        this_split_metadata["config_hash"] = config_hash

        # Schedule the ingest_split task
        celery.signature(
            "tasks.ingest_split_task",
            kwargs={
                "split_id": str(split.id),
                "file_path": split.split_file_path,
                "original_file_id": original_file_id,
                "ingestion_id": ingestion_id,
                "dataset_id": dataset_id,
                "organization_id": organization_id,
                "chunking_config": chunking_config,
                "skip_successful_split": skip_successful_files,
                "doc_id": split_doc_id,
                "metadata": this_split_metadata,
                "retry_count": 0,
                "max_retries": settings.MAX_INGESTION_RETRIES,
                "user_id": user_id,
            },
        ).apply_async()

        # Update the split's status to processing
        split.status = SplitFileStatusType.Processing
        split.task_id = None  # Will be updated by the task
        db.add(split)


def _handle_preparation_error(
    e, db, self_request_id, file_id, ingestion_id, dataset_id
):
    """
    Handle errors in the preparation process.

    Args:
        e: The exception that occurred
        db: Database session
        self_request_id: ID of the current task
        file_id: ID of the file being processed
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset

    Returns:
        Error response dictionary
    """
    db.rollback()
    logger.error(f"Error in prepare_split_ingestion: {str(e)}", exc_info=True)

    # Update ingestion status - with better error handling
    try:
        update_file_ingestion_status(
            task_id=self_request_id, status=FileIngestionStatusType.Exception
        )
    except Exception as status_error:
        logger.error(f"Failed to update status to Exception: {str(status_error)}")

        # Try direct database update as a fallback
        try:
            file_ingestion = _get_file_ingestion_record(
                db, file_id, ingestion_id, dataset_id
            )
            if file_ingestion:
                file_ingestion.status = FileIngestionStatusType.Exception
                file_ingestion.finished_at = datetime.now(UTC)
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed fallback status update: {str(db_error)}")

    return {
        "file_id": file_id,
        "error": str(e),
        "success": False,
    }


def _ensure_file_ingestion_record(db, file_id, ingestion_id, dataset_id, task_id):
    """
    Ensure that a file ingestion record exists and is up to date.

    Args:
        db: Database session
        file_id: ID of the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        task_id: ID of the current task

    Returns:
        Tuple of (success, file_ingestion_record or error_message)
    """
    # Get the file ingestion record
    file_ingestion = _get_file_ingestion_record(db, file_id, ingestion_id, dataset_id)

    if not file_ingestion:
        # If we couldn't get or create a record, try one more approach
        try:
            # Try to create or update the record using the CRUD method
            ingestion_crud.create_or_update_file_ingestion_records(
                ingestion_id=ingestion_id,
                dataset_id=UUID(dataset_id),
                created_time=None,
                status=FileIngestionStatusType.Processing,
                name="Auto-created during task",
                file_ids=[UUID(file_id)],
                task_ids=[task_id],
                db_session=db,
            )
            # Try to get the record again
            file_ingestion = _get_file_ingestion_record(
                db, file_id, ingestion_id, dataset_id
            )
        except Exception as inner_e:
            logger.error(f"Failed final attempt to create record: {str(inner_e)}")

    if not file_ingestion:
        # If we still don't have a record, we can't proceed
        error_msg = "Could not find or create file ingestion record"
        logger.error(error_msg)
        return False, error_msg

    # Update status to Processing
    file_ingestion.status = FileIngestionStatusType.Processing
    file_ingestion.task_id = task_id
    db.commit()

    return True, file_ingestion


@celery.task(name="tasks.prepare_split_ingestion_task", bind=True, acks_late=True)
def prepare_split_ingestion_task(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    organization_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    skip_successful_files: bool = True,
    doc_id: Optional[str] = None,
    file_metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Prepare a file for ingestion, splitting if necessary.

    This task:
    1. Checks if the file needs splitting
    2. If it does, creates splits and schedules individual ingestion tasks for each split
    3. If not, processes the file directly

    Args:
        file_id: ID of the file to ingest
        file_path: Path to the file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        organization_id: ID of the organization
        chunking_config: Configuration for chunking
        skip_successful_files: Whether to skip successful files
        doc_id: Document ID to assign to the ingested file
        file_metadata: Optional metadata for the file
        user_id: ID of the user who initiated the ingestion

    Returns:
        Dictionary with results of the preparation process
    """
    logger.info(f"Preparing split ingestion for file: {file_id}")

    # Initialize metadata if not provided
    if file_metadata is None:
        file_metadata = {}

    # Register task ID with ingestion process
    try:
        update_task_id(
            file_id=UUID(file_id),
            task_id=self.request.id,
            ingestion_id=ingestion_id,
            dataset_id=UUID(dataset_id),
        )
    except Exception as e:
        logger.error(f"Failed to update task ID: {str(e)}")
        # Continue execution as we'll try to get/create the record next

    # Open a database session
    with SyncSessionLocal() as db:
        try:
            # Ensure we have a valid file ingestion record
            success, result = _ensure_file_ingestion_record(
                db, file_id, ingestion_id, dataset_id, self.request.id
            )
            if not success:
                return {"error": result, "success": False}

            file_ingestion = result

            # Check if file needs splitting and get splitting configuration
            needs_splitting = check_file_needs_splitting(file_path)
            current_config_hash = get_splitting_config_hash()
            logger.info(f"Current splitting configuration hash: {current_config_hash}")
            splits_valid = check_splits_valid(db, UUID(file_id), file_ingestion.id)

            # Process based on splitting needs
            if not needs_splitting:
                # Process the file directly without splitting
                return _process_direct_ingestion(
                    db=db,
                    file_id=file_id,
                    file_path=file_path,
                    doc_id=doc_id,
                    file_metadata=file_metadata,
                    ingestion_id=ingestion_id,
                    dataset_id=dataset_id,
                    organization_id=organization_id,
                    chunking_config=chunking_config,
                    skip_successful_files=skip_successful_files,
                    user_id=user_id,
                    config_hash=current_config_hash,
                    task_id=self.request.id,
                )

            # File needs splitting - prepare splits and schedule ingestion tasks
            splits = _prepare_splits(
                db=db,
                file_id=file_id,
                file_path=file_path,
                file_ingestion_id=file_ingestion.id,
                dataset_id=dataset_id,
                splits_valid=splits_valid,
            )

            # Schedule split ingestion tasks
            _schedule_split_ingestions(
                db=db,
                splits=splits,
                original_file_id=file_id,
                file_metadata=file_metadata,
                ingestion_id=ingestion_id,
                dataset_id=dataset_id,
                organization_id=organization_id,
                chunking_config=chunking_config,
                skip_successful_files=skip_successful_files,
                config_hash=current_config_hash,
                user_id=user_id,
            )

            db.commit()

            return {
                "file_id": file_id,
                "ingestion_id": ingestion_id,
                "task_id": self.request.id,
                "success": True,
                "needs_splitting": True,
                "splits_count": len(splits),
                "config_hash": current_config_hash,
                "message": f"File split into {len(splits)} parts with configuration {current_config_hash}",
            }

        except Exception as e:
            return _handle_preparation_error(
                e, db, self.request.id, file_id, ingestion_id, dataset_id
            )
