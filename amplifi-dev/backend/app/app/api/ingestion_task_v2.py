"""
Celery tasks for file ingestion, handling direct/complete file ingestion for documents.
This version uses the Document model directly instead of FileIngestion and R2R.
Supports PDF, DOCX, XLSX, PPTX, Markdown, HTML, and CSV files.
For split-based ingestion, see prepare_split_ingestion_task_v2.py.
For image and audio files, see image_ingestion_task.py and audio_ingestion_task.py.
"""

import os
import secrets
from datetime import UTC, datetime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import attributes

from app.api.deps import get_redis_client_sync
from app.be_core.celery import celery
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.document_chunk_model import DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
)
from app.utils.document_processing_utils import (
    process_document,  # Use the new unified function
)
from app.utils.document_processing_utils import (
    _get_document_type_from_file_path,
)
from app.utils.ingestion_status_propagation import (
    propagate_ingestion_status,
    should_skip_processing_due_to_timeout,
)
from app.utils.ingestion_utils_v2 import (
    acquire_tokens_with_retry,
    file_splitter,
    update_document_status,
)
from app.utils.processing_lock_utils import (
    _cleanup_processing_lock,
    _handle_stale_lock,
)


def check_for_extracted_images(document: Document, result: Dict[str, Any]) -> bool:
    """Checks if images were extracted based on document metadata and processing result."""
    has_extracted_images = False

    # Check document metadata first
    if document and document.document_metadata:
        has_extracted_images = document.document_metadata.get(
            "has_extracted_images", False
        )

        waiting_for_images = document.document_metadata.get("waiting_for_images", False)
        if waiting_for_images:
            has_extracted_images = True

    # Check result metadata as fallback
    if not has_extracted_images and result.get("document", {}).get("metadata"):
        metadata_dict = result["document"]["metadata"]
        has_extracted_images = metadata_dict.get("has_extracted_images", False)
        waiting_for_images = metadata_dict.get("waiting_for_images", False)
        if waiting_for_images:
            has_extracted_images = True

    # Check result flag as final fallback
    if not has_extracted_images:
        has_extracted_images = result.get("has_images", False)

    return has_extracted_images


def _delete_document_chunks(document_id: UUID) -> int:
    """Delete all chunks associated with the document and return count."""
    with SyncSessionLocal() as db:
        try:
            # # Count chunks before deletion for logging
            # chunk_count = (
            #     db.query(DocumentChunk)
            #     .filter(DocumentChunk.document_id == document_id)
            #     .count()
            # )

            # Delete all chunks associated with the document
            deleted_count = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document_id)
                .delete(synchronize_session=False)
            )

            db.commit()

            if deleted_count > 0:
                logger.info(
                    f"Deleted {deleted_count} chunks for document {document_id} before reingestion."
                )
            else:
                logger.info(f"No existing chunks found for document {document_id}")

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {str(e)}")
            db.rollback()
            raise


def _cleanup_document_metadata(db, document: Document) -> None:
    """Clean up document metadata when reprocessing entire document."""
    try:
        if not document.document_metadata:
            return

        # Clean up processing-related metadata that should be regenerated
        metadata_to_clear = [
            "has_extracted_images",
            "waiting_for_images",
            "image_summary",
            "processing_stats",
            "chunk_counts",
            "split_completion",
            "last_split_status_update",
        ]

        for key in metadata_to_clear:
            document.document_metadata.pop(key, None)

        # Reset status flags
        document.document_metadata["reprocessing"] = True
        document.document_metadata["reprocessed_at"] = datetime.now(UTC).isoformat()

        # Mark for database update
        db.add(document)
        attributes.flag_modified(document, "document_metadata")
        db.commit()

        logger.info(
            f"Cleaned up metadata for document {document.id} before reprocessing"
        )

    except Exception as e:
        logger.error(f"Error cleaning up document metadata: {str(e)}")


def _should_clean_document_chunks(
    document_record: Optional[Document], skip_successful_files: bool
) -> tuple[bool, str]:
    """
    Determine if document chunks should be cleaned and why.

    Returns:
        tuple: (should_clean, reason)
    """
    if not document_record:
        return False, "no_existing_document"

    if not skip_successful_files:
        # Force reprocessing mode - always clean
        return True, "force_reprocessing"

    # Check document status for selective cleaning
    if document_record.processing_status == DocumentProcessingStatusEnum.Failed:
        return True, "previous_failure"
    elif document_record.processing_status == DocumentProcessingStatusEnum.Processing:
        return True, "incomplete_processing"
    elif document_record.processing_status == DocumentProcessingStatusEnum.Success:
        # Check if document has issues that warrant reprocessing
        if document_record.document_metadata:
            waiting_for_images = document_record.document_metadata.get(
                "waiting_for_images", False
            )
            if waiting_for_images:
                return True, "stuck_waiting_for_images"
        return False, "already_successful"

    return True, "unknown_status"


def _check_if_document_exists(
    file_id: str, ingestion_id: Optional[str], dataset_id: Optional[str]
) -> Document:
    with SyncSessionLocal() as db:
        document = (
            db.query(Document)
            .filter(
                Document.file_id == UUID(file_id),
                (Document.ingestion_id == UUID(ingestion_id) if ingestion_id else None),
                Document.dataset_id == UUID(dataset_id) if dataset_id else None,
            )
            .order_by(Document.created_at.desc())
            .first()
        )

        return document


def _validate_file_and_get_document(
    file_path: str,
    file_id: str,
    ingestion_id: Optional[str],
    dataset_id: Optional[str],
    task_id: str,
    user_id: Optional[UUID],
) -> tuple[Optional[Document], Optional[Dict[str, Any]]]:
    """
    Validate file existence and retrieve existing document record.

    Returns:
        tuple: (document_record, error_response)
        If error_response is not None, the caller should return it immediately.
    """
    document_record = _check_if_document_exists(
        file_id=file_id, ingestion_id=ingestion_id, dataset_id=dataset_id
    )

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")

        if document_record:
            _handle_exception(
                FileNotFoundError(f"File not found: {file_path}"),
                str(document_record.id),
                task_id,
                ingestion_id,
                user_id,
            )

        error_response = {
            "file_id": file_id,
            "ingestion_id": ingestion_id,
            "task_id": task_id,
            "success": False,
            "status": "exception",
            "error": f"File not found: {file_path}",
        }
        return document_record, error_response

    return document_record, None


def _should_skip_successful_document(
    document_record: Optional[Document], skip_successful_files: bool
) -> bool:
    """Check if document should be skipped due to successful processing."""
    return (
        skip_successful_files
        and document_record
        and document_record.processing_status == DocumentProcessingStatusEnum.Success
    )


def _create_skip_response(
    document_record: Document, file_id: str, ingestion_id: Optional[str], task_id: str
) -> Dict[str, Any]:
    """Create response for skipped documents."""
    return {
        "file_id": file_id,
        "document_id": str(document_record.id),
        "ingestion_id": ingestion_id,
        "task_id": task_id,
        "success": True,
        "status": "success",
        "skipped": True,
        "message": "Document already successfully processed",
    }


def _handle_successful_document_skip(
    document_record: Document,
    file_id: str,
    ingestion_id: Optional[str],
    task_id: str,
    user_id: Optional[UUID],
    db,
) -> tuple[str, Dict[str, Any]]:
    """Handle the case where document is already successfully processed and should be skipped."""
    logger.info(
        f"Document {document_record.id} already successfully processed - skipping"
    )

    # Send success notification via websocket
    if user_id:
        propagate_ingestion_status(db, "document", document_record.id, user_id)

    skip_response = _create_skip_response(
        document_record, file_id, ingestion_id, task_id
    )
    return str(document_record.id), skip_response


def _create_new_document(
    file_id: str,
    dataset_id: Optional[str],
    ingestion_id: Optional[str],
    file_path: str,
    metadata: Optional[dict],
    db,
) -> Document:
    """Create a new document record in the database with automatic document type detection."""
    # Determine document type from file path
    document_type = _get_document_type_from_file_path(file_path)

    document_record = Document(
        file_id=UUID(file_id),
        dataset_id=UUID(dataset_id) if dataset_id else None,
        ingestion_id=UUID(ingestion_id) if ingestion_id else None,
        file_path=file_path,
        document_type=document_type,  # Use detected document type
        processing_status=DocumentProcessingStatusEnum.Queued,
        document_metadata=metadata or {},
    )
    db.add(document_record)
    db.commit()
    db.refresh(document_record)
    return document_record


def _update_document_to_processing(document_record: Document, task_id: str, db) -> None:
    """Update document status to processing."""
    # Do not update documents that are already in failed/exception state
    if document_record.processing_status in [
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        logger.info(
            f"Document {document_record.id} is in {document_record.processing_status} state - not updating status or timestamp"
        )
        return

    document_record.processing_status = DocumentProcessingStatusEnum.Processing
    document_record.task_id = task_id
    document_record.updated_at = datetime.now(UTC)
    db.commit()


def _create_error_response(
    file_id: str, ingestion_id: Optional[str], task_id: str, error_message: str
) -> Dict[str, Any]:
    """Create error response for document record creation failures."""
    return {
        "file_id": file_id,
        "ingestion_id": ingestion_id,
        "task_id": task_id,
        "success": False,
        "status": "exception",
        "error": f"Failed to create/update document record: {error_message}",
    }


def _handle_document_creation_exception(
    exception: Exception,
    document_record: Optional[Document],
    task_id: str,
    ingestion_id: Optional[str],
    user_id: Optional[UUID],
    file_id: str,
    db,
) -> Dict[str, Any]:
    """Handle exceptions during document record creation."""
    db.rollback()
    logger.error(f"Error creating/updating document record: {str(exception)}")

    if (
        user_id
        and document_record
        and hasattr(document_record, "id")
        and document_record.id
    ):
        _handle_exception(
            exception, str(document_record.id), task_id, ingestion_id, user_id
        )

    return _create_error_response(file_id, ingestion_id, task_id, str(exception))


def _handle_document_record_creation(
    document_record: Optional[Document],
    file_id: str,
    dataset_id: Optional[str],
    ingestion_id: Optional[str],
    file_path: str,
    metadata: Optional[dict],
    task_id: str,
    skip_successful_files: bool,
    user_id: Optional[UUID],
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Handle document record creation/validation and check for skip conditions.

    Returns:
        tuple: (document_id, response)
        If response is not None, the caller should return it immediately.
    """
    with SyncSessionLocal() as db:
        try:
            # Check if document should be skipped due to successful processing
            if _should_skip_successful_document(document_record, skip_successful_files):
                return _handle_successful_document_skip(
                    document_record, file_id, ingestion_id, task_id, user_id, db
                )

            # Determine if we should clean chunks
            should_clean, clean_reason = _should_clean_document_chunks(
                document_record, skip_successful_files
            )

            logger.info(
                f"Chunk cleanup decision for document: should_clean={should_clean}, reason={clean_reason}"
            )

            # Handle chunk cleanup if needed
            if should_clean and document_record:
                deleted_count = _delete_document_chunks(document_record.id)
                _cleanup_document_metadata(db, document_record)

                logger.info(
                    f"Cleaned {deleted_count} chunks and metadata for document {document_record.id} "
                    f"(reason: {clean_reason})"
                )

            # Create new document record if it doesn't exist
            if not document_record:
                document_record = _create_new_document(
                    file_id, dataset_id, ingestion_id, file_path, metadata, db
                )
                logger.info(
                    f"Created new document {document_record.id} for file {file_id}"
                )
            else:
                logger.info(
                    f"Reusing existing document {document_record.id} for file {file_id}"
                )

            # Update document status to processing
            _update_document_to_processing(document_record, task_id, db)

            return str(document_record.id), None

        except Exception as e:
            error_response = _handle_document_creation_exception(
                e, document_record, task_id, ingestion_id, user_id, file_id, db
            )
            return None, error_response


def _handle_token_acquisition_with_retry(
    file_path: str,
    file_id: str,
    retry_count: int,
    max_retries: int,
    document_id: Optional[str],
    task_id: str,
    metadata: Optional[dict],
    chunking_config: Optional[dict],
    ingestion_id: Optional[str],
    dataset_id: Optional[str],
    organization_id: Optional[str],
    skip_successful_files: bool,
    user_id: Optional[UUID],
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Handle token acquisition with retry logic.

    Returns:
        tuple: (success, response)
        If response is not None, the caller should return it immediately.
    """
    try:
        token_count = file_splitter.count_file_tokens(file_path)
        logger.info(f"Estimated token count for file {file_id}: {token_count}")

        success, remaining_time = acquire_tokens_with_retry(token_count)

        if not success:
            # If we couldn't get tokens and haven't exceeded max retries,
            # schedule a retry with exponential backoff
            if retry_count < max_retries:
                retry_response = _schedule_ingestion_retry(
                    remaining_time=remaining_time,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    file_path=file_path,
                    file_id=file_id,
                    metadata=metadata,
                    chunking_config=chunking_config,
                    ingestion_id=ingestion_id,
                    dataset_id=dataset_id,
                    organization_id=organization_id,
                    skip_successful_files=skip_successful_files,
                    user_id=user_id,
                    task_id=task_id,
                )
                return False, retry_response
            else:
                # Max retries exceeded, mark as failed
                update_document_status(
                    task_id=task_id,
                    status=DocumentProcessingStatusEnum.Failed,
                    error_message=f"Failed to acquire tokens after {max_retries} retries",
                )

                # Propagate failed status up to file and dataset level
                if document_id:
                    _propagate_token_failure_status(document_id, user_id)

                failure_response = {
                    "file_id": file_id,
                    "document_id": document_id,
                    "ingestion_id": ingestion_id,
                    "task_id": task_id,
                    "success": False,
                    "status": "failed",
                    "error": f"Failed to acquire tokens after {max_retries} retries",
                }
                return False, failure_response

        return True, None

    except Exception as e:
        logger.error(f"Error acquiring tokens: {str(e)}")
        error_response = _handle_exception(
            e, document_id, task_id, ingestion_id, user_id
        )
        return False, error_response


def _propagate_token_failure_status(document_id: str, user_id: Optional[UUID]) -> None:
    """Handle status propagation when token acquisition fails."""
    if not document_id or not user_id:
        return

    with SyncSessionLocal() as db:
        try:
            doc_uuid = UUID(document_id)
            doc_to_notify = db.query(Document).filter(Document.id == doc_uuid).first()

            if doc_to_notify:
                propagate_ingestion_status(db, "document", doc_to_notify.id, user_id)
            else:
                logger.warning(
                    f"Could not find document with id {document_id} to propagate failure status after token acquisition failure."
                )
        except (ValueError, TypeError) as ve:
            logger.error(
                f"Invalid document_id '{document_id}' for UUID conversion: {ve}"
            )


def _process_document_and_update_status(
    file_path: str,
    file_id: str,
    document_id: str,
    dataset_id: Optional[str],
    chunking_config: Optional[dict],
    ingestion_id: Optional[str],
    user_id: Optional[UUID],
) -> Dict[str, Any]:
    """
    Process the document and update its status based on results.
    Supports PDF, DOCX, XLSX, PPTX, Markdown, HTML, and CSV files.

    Returns:
        Dictionary with processing results
    """
    # Detect document type for logging
    document_type = _get_document_type_from_file_path(file_path)
    logger.info(
        f"Starting {document_type.value} document processing for file {file_id}"
    )

    # Process the document using the unified function
    result = process_document(
        file_path=file_path,
        file_id=UUID(file_id),
        document_id=UUID(document_id),
        dataset_id=UUID(dataset_id),
        chunking_config=chunking_config,
        ingestion_id=UUID(ingestion_id) if ingestion_id else None,
        user_id=user_id,
        split_id=None,  # No split for direct ingestion
    )

    with SyncSessionLocal() as db:
        document = (
            db.query(Document)
            .filter(Document.id == UUID(document_id))
            .with_for_update()
            .first()
        )

        if not document:
            logger.error(f"Document {document_id} not found after processing")
            return {
                "file_id": file_id,
                "document_id": document_id,
                "success": False,
                "error": "Document not found after processing",
            }

        if result["success"]:
            _update_successful_document_status(document, result, user_id, db)
        else:
            _update_failed_document_status(document, result)

        db.commit()

        # Handle status propagation
        _handle_status_propagation(document, user_id, db)

    # Extract information from appropriate result structure
    has_extracted_images = (
        result.get("has_images", False) if result["success"] else False
    )

    return {
        "file_id": file_id,
        "document_id": document_id,
        "success": result["success"],
        "has_extracted_images": has_extracted_images,
        "message": f"{document_type.value} document processing completed",
        "processing_type": "direct",  # Indicate this was direct processing
        "document_type": document_type.value,
        "chunk_counts": (
            {
                "text": result.get("text_chunk_count", 0),
                "tables": result.get("table_count", 0),
                "images": result.get("image_count", 0),
            }
            if result["success"]
            else {}
        ),
    }


def _update_successful_document_status(
    document: Document, result: Dict[str, Any], user_id: Optional[UUID], db
) -> None:
    """Update document status for successful processing."""
    has_extracted_images = result.get("has_images", False)

    logger.info(
        f"Document {document.id}: has_extracted_images = {has_extracted_images} "
        f"(direct ingestion, image_count = {result.get('image_count', 0)})"
    )

    if not document:
        logger.error(f"Document {document.id} not found when updating status")
        return

    # If document is already Success, don't downgrade it
    if document.processing_status == DocumentProcessingStatusEnum.Success:
        logger.info(
            f"Document {document.id} already marked as Success - not changing status"
        )
        return

    # If document is Failed/Exception, don't override
    if document.processing_status in [
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        logger.info(
            f"Document {document.id} is in {document.processing_status} state - not changing"
        )
        return

    if has_extracted_images:
        # Document has images - keep as Processing until images complete
        document.processing_status = DocumentProcessingStatusEnum.Processing

        if not document.document_metadata:
            document.document_metadata = {}
        document.document_metadata["waiting_for_images"] = True

        logger.info(
            f"Document {document.id} marked as Processing - extracted {result.get('image_count', 0)} images"
        )
    else:
        # Document has no images - mark as Success immediately
        document.processing_status = DocumentProcessingStatusEnum.Success
        document.processed_at = datetime.now(UTC)

        logger.info(f"Document {document.id} marked as Success - no images extracted")


def _update_failed_document_status(document: Document, result: Dict[str, Any]) -> None:
    """Update document status for failed processing."""
    document.processing_status = DocumentProcessingStatusEnum.Failed
    document.error_message = str(result.get("errors", ["Unknown error"]))
    document.processed_at = datetime.now(UTC)


def _handle_status_propagation(document: Document, user_id: Optional[UUID], db) -> None:
    """Handle status propagation based on document processing status."""
    if (
        document.processing_status == DocumentProcessingStatusEnum.Success
        or document.processing_status == DocumentProcessingStatusEnum.Failed
    ):
        # No images or processing failed - safe to propagate immediately
        if user_id:
            propagate_ingestion_status(db, "document", document.id, user_id)
    else:
        # Document has images processing - waiting for completion
        logger.info(
            f"Document {document.id} has images processing - waiting for completion"
        )


def _acquire_processing_lock_atomic(
    redis_client, processing_key: str, task_id: str, expiry_seconds: int = 3600
) -> tuple[bool, Optional[str]]:
    """
    Atomically acquire a processing lock using Redis SET NX EX.

    Args:
        redis_client: Redis client instance
        processing_key: The key to lock
        task_id: Current task ID to store in the lock
        expiry_seconds: Lock expiry time in seconds

    Returns:
        tuple: (acquired, existing_task_id)
        - acquired: True if lock was acquired, False if already exists
        - existing_task_id: Task ID that currently holds the lock (if any)
    """
    try:
        # Try to atomically set the key only if it doesn't exist (NX) with expiration (EX)
        result = redis_client.set(processing_key, task_id, nx=True, ex=expiry_seconds)

        if result:
            logger.info(f"Successfully acquired file processing lock: {processing_key}")
            return True, None
        else:
            # Lock exists, get the current task ID
            existing_task_id = redis_client.get(processing_key)
            if existing_task_id:
                # Handle both bytes and string types
                if isinstance(existing_task_id, bytes):
                    existing_task_id = existing_task_id.decode("utf-8")
                else:
                    existing_task_id = str(existing_task_id)

            logger.info(
                f"Processing lock already exists for key {processing_key}, held by task: {existing_task_id}"
            )
            return False, existing_task_id

    except Exception as e:
        logger.error(f"Error acquiring processing lock {processing_key}: {str(e)}")
        # If Redis is unavailable, allow processing to continue
        return True, None


@celery.task(name="tasks.ingest_files_task_v2", bind=True, acks_late=True)
def ingest_files_task_v2(
    self,
    file_path: str,
    file_id: str,
    metadata: Optional[dict] = None,
    chunking_config: Optional[dict] = None,
    ingestion_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    skip_successful_files: bool = True,
    user_id: Optional[UUID] = None,
    retry_count: int = 0,
    max_retries: int = 10,
    retry_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Document ingestion task that processes a single file.
    Supports PDF, DOCX, XLSX, PPTX, Markdown, HTML, and CSV files.
    Uses document_processing_utils.process_document() for extraction and processing.
    Uses a non-blocking retry approach for token acquisition.

    Args:
        file_path: Path to the file to be ingested
        file_id: ID of the file
        metadata: Optional metadata for the file
        chunking_config: Configuration for chunking
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        organization_id: ID of the organization
        skip_successful_files: Whether to skip files that are already successfully ingested
        user_id: ID of the user who initiated the ingestion, for WebSocket updates
        retry_count: Current retry attempt count for token acquisition
        max_retries: Maximum number of retries before giving up
        retry_reason: Reason for retry (e.g., "timeout", "rate_limit")

    Returns:
        Dictionary with results of the ingestion process
    """
    # Detect document type for logging
    document_type = _get_document_type_from_file_path(file_path)

    logger.info(
        f"Starting ingestion_task_v2 for {document_type.value} file: {file_path} "
        f"(retry #{retry_count}, reason: {retry_reason or 'initial'})"
    )

    # Robust task-level deduplication with deadlock prevention
    processing_key = None
    redis_client = None

    try:
        redis_client = get_redis_client_sync()

        # Create unique processing key
        processing_key = (
            f"processing:file:{file_id}:ingestion:{ingestion_id}:dataset:{dataset_id}"
        )

        # Try to atomically acquire processing lock
        acquired, existing_task_id = _acquire_processing_lock_atomic(
            redis_client, processing_key, self.request.id
        )

        if not acquired:
            # Lock exists - check if original task is still alive
            if existing_task_id and not _handle_stale_lock(
                redis_client, processing_key, existing_task_id, self.request.id
            ):
                logger.info(
                    f"File {file_id} is being processed by active task {existing_task_id} - skipping"
                )
                try:
                    redis_client.close()
                except Exception as e:
                    logger.warning(f"Failed to close Redis connection: {str(e)}")

                return {
                    "file_id": file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "already_processing",
                    "processing_task_id": existing_task_id,
                    "document_type": document_type.value,
                }
            # If we reach here, either we took over a stale lock or the original task is dead
            logger.info(
                f"Acquired processing lock after stale lock handling: {processing_key}"
            )

        # logger.info(f"Successfully acquired file processing lock: {processing_key}")

    except Exception as e:
        # Don't fail the task if Redis is unavailable, just log it
        logger.warning(f"Redis deduplication not available: {str(e)}")
        processing_key = None
        redis_client = None

    document_id = None

    try:
        # Step 1: Validate file and get existing document record
        document_record, error_response = _validate_file_and_get_document(
            file_path, file_id, ingestion_id, dataset_id, self.request.id, user_id
        )
        if error_response:
            return error_response

        # Step 2: Handle document record creation/validation and check for skip conditions
        document_id, response = _handle_document_record_creation(
            document_record,
            file_id,
            dataset_id,
            ingestion_id,
            file_path,
            metadata,
            self.request.id,
            skip_successful_files,
            user_id,
        )
        if response:
            return response

        # CHECK here to skip document if already marked failed due to timeout just before token acquisition
        logger.info(f"CHECKING if document {document_id} is failed due to timeout")
        if document_id:
            should_skip, skip_reason = should_skip_processing_due_to_timeout(
                document_id
            )
            if should_skip:
                logger.warning(
                    f"⏰ SKIPPING document processing for document {document_id} - {skip_reason}"
                )
                return {
                    "file_id": file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "document_timeout_failed",
                    "message": skip_reason,
                    "document_id": str(document_id),
                    "document_type": document_type.value,
                }

        # Step 3: Handle token acquisition with retry logic
        token_success, token_response = _handle_token_acquisition_with_retry(
            file_path,
            file_id,
            retry_count,
            max_retries,
            document_id,
            self.request.id,
            metadata,
            chunking_config,
            ingestion_id,
            dataset_id,
            organization_id,
            skip_successful_files,
            user_id,
        )
        if token_response:
            return token_response

        # Step 4: Process document and update status
        result = _process_document_and_update_status(
            file_path,
            file_id,
            document_id,
            dataset_id,
            chunking_config,
            ingestion_id,
            user_id,
        )

        # Add task_id and ingestion_id to result
        result.update(
            {
                "ingestion_id": ingestion_id,
                "task_id": self.request.id,
            }
        )

        logger.info(
            f"Successfully completed {document_type.value} document processing for file {file_id}: "
        )

        return result

    except Exception as e:
        logger.error(
            f"Error processing {document_type.value} document: {str(e)}", exc_info=True
        )
        return _handle_exception(e, document_id, self.request.id, ingestion_id, user_id)

    finally:
        # Clean up processing deduplication flag (only if we own it)
        if processing_key and redis_client:
            try:
                _cleanup_processing_lock(redis_client, processing_key, self.request.id)
            except Exception as e:
                logger.warning(f"Failed to clean up processing lock: {str(e)}")

        # Close Redis connection
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")


def _schedule_ingestion_retry(
    remaining_time: float,
    retry_count: int,
    max_retries: int,
    file_path: str,
    file_id: str,
    metadata: Optional[dict],
    chunking_config: Optional[dict],
    ingestion_id: Optional[str],
    dataset_id: Optional[str],
    organization_id: Optional[str],
    skip_successful_files: bool,
    user_id: Optional[UUID],
    task_id: str,
) -> Dict[str, Any]:
    """Schedule a retry of the ingestion task with exponential backoff."""

    # Calculate backoff time with exponential increase and jitter
    from app.be_core.config import settings

    base_delay = remaining_time or settings.RATE_LIMIT_RETRY_SECONDS * (2**retry_count)
    max_backoff = 300  # 5 minutes max
    backoff = min(base_delay, max_backoff)

    # Add jitter (±10%) to prevent all tasks retrying simultaneously
    jitter = (float(secrets.randbelow(20) - 10) / 100.0) * backoff  # ±10% jitter
    countdown = max(1, backoff + jitter)

    logger.info(
        f"Could not acquire tokens for file {file_id}. "
        f"Retry {retry_count+1}/{max_retries} scheduled in {countdown:.1f}s"
    )

    # Schedule a retry for this same file
    celery.signature(
        "tasks.ingest_files_task_v2",
        kwargs={
            "file_path": file_path,
            "file_id": file_id,
            "metadata": metadata,
            "chunking_config": chunking_config,
            "ingestion_id": ingestion_id,
            "dataset_id": dataset_id,
            "organization_id": organization_id,
            "skip_successful_files": skip_successful_files,
            "user_id": user_id,
            "retry_count": retry_count + 1,
            "max_retries": max_retries,
            "retry_reason": "rate_limit",
        },
    ).apply_async(countdown=countdown)

    return {
        "file_id": file_id,
        "ingestion_id": ingestion_id,
        "task_id": task_id,
        "success": False,
        "status": "scheduled_retry",
        "retry_count": retry_count + 1,
        "retry_delay": countdown,
    }


def _handle_exception(
    exception: Exception,
    document_id: str,
    task_id: str,
    ingestion_id: Optional[str],
    user_id: Optional[UUID],
) -> Dict[str, Any]:
    """Handle exceptions during the ingestion process."""
    import traceback

    error_message = f"{str(exception)}\n{traceback.format_exc()}"
    logger.error(f"Error processing document: {error_message}")

    # Update document status
    if document_id:
        update_document_status(
            task_id=task_id,
            status=DocumentProcessingStatusEnum.Failed,
            error_message=str(error_message),
        )

    # Send error notification
    if user_id:
        with SyncSessionLocal() as db:
            if document_id:
                try:
                    doc_uuid = UUID(document_id)
                    propagate_ingestion_status(db, "document", doc_uuid, user_id)
                except ValueError:
                    logger.error(
                        f"Invalid document_id format '{document_id}' in _handle_exception. Cannot propagate status."
                    )
                    # Potentially re-raise or handle as a more critical error if status propagation is vital
    return {
        "document_id": document_id,
        "ingestion_id": ingestion_id,
        "task_id": task_id,
        "success": False,
        "status": "exception",
        "error": str(exception),
    }
