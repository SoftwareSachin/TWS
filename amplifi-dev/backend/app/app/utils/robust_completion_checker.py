"""
Robust document completion checker with Redis primary and PostgreSQL fallback.

This module provides race-condition-free completion checking for documents with images.
It uses Redis distributed locks when available and falls back to PostgreSQL advisory locks
when Redis is unavailable.

"""

import hashlib
import time
import uuid
from datetime import UTC, datetime
from typing import List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import attributes

from app.api.deps import get_redis_client_sync
from app.be_core.celery import celery
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import Document, DocumentProcessingStatusEnum
from app.models.file_split_model import FileSplit, SplitFileStatusType
from app.utils.ingestion_status_propagation import propagate_ingestion_status


def check_document_completion_with_fallback(
    parent_document_id: str, user_id: Optional[uuid.UUID] = None
) -> None:
    """
    Robust completion check with Redis primary and PostgreSQL fallback.

    This is the main entry point for document completion checking.
    It automatically handles Redis failures and falls back to PostgreSQL.

    Args:
        parent_document_id: The ID of the document to check for completion
        user_id: The ID of the user for status propagation
    """
    logger.debug(f"Starting robust completion check for document {parent_document_id}")

    # Try Redis first
    if _try_redis_completion_check(parent_document_id, user_id):
        return

    # Fall back to PostgreSQL advisory locks
    logger.info(
        f"Redis unavailable, using PostgreSQL advisory locks for document {parent_document_id}"
    )
    _check_document_completion_with_advisory_lock(parent_document_id, user_id)


def _try_redis_completion_check(
    parent_document_id: str, user_id: Optional[uuid.UUID]
) -> bool:
    """
    Try Redis-based completion check, return True if successful.

    Args:
        parent_document_id: Document ID to check
        user_id: User ID for notifications

    Returns:
        True if Redis check was successful (or another task is handling it)
        False if Redis is unavailable and we should fall back to PostgreSQL
    """
    redis_client = None
    completion_lock_key = None

    try:
        redis_client = get_redis_client_sync()
        completion_lock_key = f"completion_check:document:{parent_document_id}"

        # Test Redis connectivity with a quick ping
        redis_client.ping()

        # Get current task ID for lock ownership
        current_task_id = _get_current_task_id()

        # Try to acquire Redis lock with retry mechanism
        lock_acquired = redis_client.set(
            completion_lock_key,
            current_task_id,
            nx=True,  # Only set if not exists
            ex=120,  # 2 minute expiry
        )

        if not lock_acquired:
            logger.debug(
                f"Redis completion check for document {parent_document_id} already in progress - retrying"
            )

            # Wait and retry up to 3 times with 5-second intervals
            for attempt in range(1, 4):  # attempts 1, 2, 3
                logger.debug(f"Waiting for completion lock (attempt {attempt}/3)...")
                time.sleep(5)  # Wait 5 seconds

                lock_acquired = redis_client.set(
                    completion_lock_key,
                    current_task_id,
                    nx=True,
                    ex=120,
                )

                if lock_acquired:
                    logger.info(
                        f"Acquired Redis completion lock for document {parent_document_id} on retry attempt {attempt}"
                    )
                    break

            if not lock_acquired:
                logger.warning(
                    f"Could not acquire Redis completion lock for document {parent_document_id} after 3 retries - falling back to PostgreSQL"
                )
                return False  # Fall back to PostgreSQL advisory locks

        logger.debug(
            f"Acquired Redis completion lock for document {parent_document_id}"
        )

        try:
            # Perform the actual completion check
            _perform_robust_completion_check(parent_document_id, user_id, "redis")
            return True

        finally:
            # Always release the Redis lock
            try:
                redis_client.delete(completion_lock_key)
                logger.debug(
                    f"Released Redis completion lock for document {parent_document_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to release Redis lock: {str(e)}")

    except Exception as e:
        logger.warning(
            f"Redis completion check failed for document {parent_document_id}: {str(e)}"
        )
        return False  # Fall back to PostgreSQL
    finally:
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis client: {str(e)}")


def _check_document_completion_with_advisory_lock(
    parent_document_id: str, user_id: Optional[uuid.UUID] = None
) -> None:
    """
    Check document completion using PostgreSQL advisory locks as fallback.

    PostgreSQL advisory locks are application-level locks that don't block tables
    but ensure only one process can work on a specific resource.

    Args:
        parent_document_id: Document ID to check
        user_id: User ID for notifications
    """
    # Convert document_id to a consistent integer for advisory lock
    lock_id = _get_advisory_lock_id(parent_document_id)

    with SyncSessionLocal() as db:
        try:
            # Try to acquire PostgreSQL advisory lock (non-blocking)
            result = db.execute(
                text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": lock_id}
            )
            lock_acquired = result.scalar()

            if not lock_acquired:
                logger.debug(
                    f"Document {parent_document_id} completion check already in progress "
                    f"(advisory lock {lock_id} held by another session)"
                )
                return

            logger.debug(
                f"Acquired advisory lock {lock_id} for document {parent_document_id}"
            )

            try:
                # Perform the completion check
                _perform_robust_completion_check(
                    parent_document_id, user_id, "postgresql"
                )

            finally:
                # Always release the advisory lock
                db.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": lock_id}
                )
                logger.debug(
                    f"Released advisory lock {lock_id} for document {parent_document_id}"
                )

        except OperationalError as e:
            # Handle lock acquisition failures gracefully
            if "could not obtain lock" in str(e).lower():
                logger.debug(
                    f"Could not acquire row lock for document {parent_document_id} - skipping"
                )
                return
            else:
                logger.error(f"Database error in completion check: {str(e)}")
                raise
        except Exception as e:
            logger.error(
                f"Error in PostgreSQL completion check for {parent_document_id}: {str(e)}",
                exc_info=True,
            )
            raise


def _perform_robust_completion_check(
    parent_document_id: str, user_id: Optional[uuid.UUID], method: str
) -> None:
    """
    Perform the actual completion check logic.

    This is the core logic separated from the locking mechanism.
    It uses actual counts instead of expected counts to prevent the race condition bug.

    Args:
        parent_document_id: Document ID to check
        user_id: User ID for notifications
        method: Locking method used ("redis" or "postgresql")
    """
    with SyncSessionLocal() as db:
        try:
            # Start transaction with row-level locking for extra safety
            with db.begin():
                # Get document with SELECT FOR UPDATE (wait for lock if needed)
                parent_doc = (
                    db.query(Document)
                    .filter(Document.id == uuid.UUID(parent_document_id))
                    .with_for_update()  # Wait for lock instead of failing fast
                    .first()
                )

                if not parent_doc:
                    logger.warning(f"Document {parent_document_id} not found")
                    return

                # Skip if already in final state
                if parent_doc.processing_status in [
                    DocumentProcessingStatusEnum.Success,
                    DocumentProcessingStatusEnum.Failed,
                    DocumentProcessingStatusEnum.Exception,
                ]:
                    logger.debug(
                        f"Document {parent_document_id} already in final state: {parent_doc.processing_status}"
                    )
                    return

                # Get ALL image chunks for this document (both direct and from splits)
                image_chunks = (
                    db.query(DocumentChunk)
                    .filter(
                        DocumentChunk.document_id == uuid.UUID(parent_document_id),
                        DocumentChunk.chunk_type == ChunkTypeEnum.ImageDescription,
                    )
                    .all()
                )

                propagation_info = None  # Initialize to None

                if not image_chunks:
                    # No images - mark document as complete
                    logger.info(
                        f"✅ Document {parent_document_id} has no images - marking as complete (method: {method})"
                    )
                    _finalize_document_completion(
                        db, parent_doc, DocumentProcessingStatusEnum.Success, method
                    )

                    # Store propagation info for after commit (no images case)
                    propagation_info = {
                        "document_id": parent_doc.id,
                        "dataset_id": parent_doc.dataset_id,
                        "status": parent_doc.processing_status,
                        "user_id": user_id,
                    }
                else:
                    # Count image statuses - this is the critical part that was racing
                    success_count, failed_count, processing_count = (
                        _count_image_chunk_statuses(image_chunks)
                    )

                    # Get the ORIGINAL total from metadata (what was actually extracted)
                    original_total = _get_original_total_images(parent_doc)
                    chunks_found = len(image_chunks)
                    completed_count = success_count + failed_count

                    # Handle edge case: if no original total found, use chunks found as fallback
                    if original_total == 0 and chunks_found > 0:
                        logger.warning(
                            f"Document {parent_document_id} has no original image count in metadata, "
                            f"using chunks found ({chunks_found}) as fallback"
                        )
                        original_total = chunks_found

                    # Calculate processing count: everything that's not completed yet
                    # Handle edge case where completed > original (the bug we're fixing)
                    if completed_count > original_total:
                        logger.warning(
                            f"Document {parent_document_id} has MORE completed chunks ({completed_count}) "
                            f"than original extracted ({original_total}) - using completed count as new total"
                        )
                        # Use the higher count as the authoritative total
                        actual_total = completed_count
                        actual_processing_count = 0  # All are complete
                    else:
                        actual_total = original_total
                        actual_processing_count = original_total - completed_count

                    # Always log the current status (whether complete or not)
                    logger.info(
                        f"🔍 Document {parent_document_id} IMAGES status ({method}): "
                        f"Total: {actual_total}, "
                        f"Completed: {completed_count} ({success_count} successful, {failed_count} failed), "
                        f"Processing: {actual_processing_count}"
                    )

                    # Update document metadata with ACTUAL counts (this prevents the negative count bug)
                    _update_document_image_metadata_robust(
                        parent_doc, actual_total, success_count, failed_count, method
                    )

                    # Check completion based on actual total vs completed
                    if actual_processing_count == 0:  # All images are complete
                        if failed_count == actual_total:
                            # All images failed
                            final_status = DocumentProcessingStatusEnum.Failed
                        else:
                            # At least some images succeeded
                            final_status = DocumentProcessingStatusEnum.Success

                        logger.info(
                            f"🎉 Document {parent_document_id} COMPLETE - "
                            f"all {actual_total} images finished, marking as {final_status} (method: {method})"
                        )

                        _finalize_document_completion(
                            db, parent_doc, final_status, method
                        )

                        # Store propagation info for after commit
                        propagation_info = {
                            "document_id": parent_doc.id,
                            "dataset_id": parent_doc.dataset_id,
                            "status": parent_doc.processing_status,
                            "user_id": user_id,
                        }
                    else:
                        logger.debug(
                            f"⏳ Document {parent_document_id} waiting - {actual_processing_count} images still processing"
                        )
                        propagation_info = None

            db.commit()

            # Propagate status AFTER commit so separate session sees the changes
            if propagation_info and propagation_info["user_id"]:
                logger.info(
                    f"🔔 Propagating completion status for document {propagation_info['document_id']} to user {propagation_info['user_id']}"
                )
                logger.info(
                    f"📄 Document details: dataset_id={propagation_info['dataset_id']}, status={propagation_info['status']}"
                )
                try:
                    # Create separate session for propagation to avoid transaction conflicts
                    with SyncSessionLocal() as propagation_db:
                        result = propagate_ingestion_status(
                            propagation_db,
                            "document",
                            propagation_info["document_id"],
                            propagation_info["user_id"],
                        )
                        logger.info(f"📡 Propagation function returned: {result}")
                    logger.info(
                        f"✅ Successfully propagated completion status for document {propagation_info['document_id']}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Failed to propagate completion status for document {propagation_info['document_id']}: {str(e)}",
                        exc_info=True,
                    )
            elif propagation_info:
                logger.warning(
                    f"⚠️ No user_id provided for document {propagation_info['document_id']} - skipping propagation"
                )

        except OperationalError as e:
            # Handle row lock failures gracefully
            if "could not obtain lock" in str(e).lower():
                logger.debug(
                    f"Could not acquire row lock for document {parent_document_id} - skipping"
                )
                return
            else:
                raise
        except Exception as e:
            logger.error(
                f"Error in robust completion check for {parent_document_id}: {str(e)}",
                exc_info=True,
            )
            raise


def _get_original_total_images(document: Document) -> int:
    """
    Get the original total number of images extracted from the document.
    This is the authoritative count stored during document processing.
    """
    if not document.document_metadata:
        return 0

    image_summary = document.document_metadata.get("image_summary", {})
    return image_summary.get("total_images", 0)


def _get_split_original_image_count(split_record: FileSplit) -> int:
    """
    Get the original number of images extracted from this specific split.
    This is the authoritative count stored during split processing.
    """
    if not split_record.split_metadata:
        return 0

    return split_record.split_metadata.get("images_total", 0)


def _update_split_status_after_completion(
    db, split_record: FileSplit, split_id: str, success_count: int, failed_count: int
) -> None:
    """
    Update split status to Success/Failed based on image completion results.
    This is called when all images in a split are complete.
    """
    try:
        # Import here to avoid circular imports
        from app.models.file_split_model import SplitFileStatusType

        # Determine split status based on results
        if success_count > 0:
            new_status = SplitFileStatusType.Success
            logger.info(
                f"Updating split {split_id} to Success - {success_count} images successful, {failed_count} failed"
            )
        else:
            new_status = SplitFileStatusType.Failed
            logger.info(
                f"Updating split {split_id} to Failed - all {failed_count} images failed"
            )

        # Update split status (manually to avoid transaction conflicts)
        split_record.status = new_status
        if new_status in [
            SplitFileStatusType.Success,
            SplitFileStatusType.Failed,
            SplitFileStatusType.Exception,
        ]:
            split_record.finished_at = datetime.now()

        db.add(split_record)

        # Update split metadata with completion info (matching old system)
        if not split_record.split_metadata:
            split_record.split_metadata = {}

        split_record.split_metadata.update(
            {
                "waiting_for_images": False,
                "images_completed": True,
                "images_success_count": success_count,
                "images_failed_count": failed_count,
                "completion_timestamp": datetime.now(UTC).isoformat(),
                # Keep our additional fields too
                "image_processing_complete": True,
                "image_completed_at": datetime.now(UTC).isoformat(),
            }
        )

        from sqlalchemy.orm import attributes

        attributes.flag_modified(split_record, "split_metadata")

        logger.info(f"Split {split_id} status updated and metadata saved")

    except Exception as e:
        logger.error(f"Error updating split {split_id} status: {str(e)}", exc_info=True)


def _update_document_level_summary_robust(
    db, parent_doc: Document, split_id: str, success_count: int, failed_count: int
) -> None:
    """
    Update document-level summary with split completion information.
    This matches the functionality of the old system's _update_document_level_summary.
    """
    try:
        from sqlalchemy.orm import attributes

        if not (
            parent_doc
            and parent_doc.document_metadata
            and "image_summary" in parent_doc.document_metadata
        ):
            logger.debug(f"No image summary found for document {parent_doc.id}")
            return

        image_summary = parent_doc.document_metadata["image_summary"]

        # Update split status in document summary
        _update_split_status_in_document_summary_robust(
            image_summary, split_id, success_count, failed_count
        )

        # Check if all splits are complete
        _check_and_update_all_splits_completion_robust(
            image_summary, str(parent_doc.id), parent_doc
        )

        # Store updated summary
        parent_doc.document_metadata["image_summary"] = image_summary
        attributes.flag_modified(parent_doc, "document_metadata")

        logger.debug(
            f"Updated document-level summary for {parent_doc.id}, split {split_id}"
        )

    except Exception as e:
        logger.error(f"Error updating document level summary: {str(e)}", exc_info=True)


def _update_split_status_in_document_summary_robust(
    image_summary: dict, split_id: str, success_count: int, failed_count: int
) -> None:
    """Update split status in document summary."""
    if "images_by_split" not in image_summary:
        image_summary["images_by_split"] = {}

    split_key = split_id
    if split_key in image_summary["images_by_split"]:
        image_summary["images_by_split"][split_key].update(
            {
                "processing_complete": True,
                "success_count": success_count,
                "failed_count": failed_count,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )


def _check_and_update_all_splits_completion_robust(
    image_summary: dict, parent_document_id: str, parent_doc
) -> None:
    """Check if all splits are complete and update document summary accordingly."""
    all_splits_complete = True
    total_success = 0
    total_failed = 0

    for _split_key, split_info in image_summary.get("images_by_split", {}).items():
        if not split_info.get("processing_complete", False):
            all_splits_complete = False
            break
        total_success += split_info.get("success_count", 0)
        total_failed += split_info.get("failed_count", 0)

    if all_splits_complete:
        image_summary.update(
            {
                "processing_status": "completed",
                "total_success_count": total_success,
                "total_failed_count": total_failed,
                "all_completed_at": datetime.now(UTC).isoformat(),
            }
        )

        # Clear waiting flag
        if parent_doc.document_metadata:
            parent_doc.document_metadata.pop("waiting_for_images", None)

        logger.info(
            f"Document {parent_document_id} image processing complete: "
            f"{total_success} successful, {total_failed} failed"
        )


def _count_image_chunk_statuses(
    image_chunks: List[DocumentChunk],
) -> Tuple[int, int, int]:
    """
    Count image chunk statuses and return (success, failed, processing).

    Args:
        image_chunks: List of image chunks to analyze

    Returns:
        Tuple of (success_count, failed_count, processing_count)
    """
    success_count = 0
    failed_count = 0
    processing_count = 0

    for chunk in image_chunks:
        if chunk.chunk_metadata and "status" in chunk.chunk_metadata:
            status = chunk.chunk_metadata["status"]
            if status == DocumentProcessingStatusEnum.Success.value:
                success_count += 1
            elif status in [
                DocumentProcessingStatusEnum.Failed.value,
                DocumentProcessingStatusEnum.Exception.value,
            ]:
                failed_count += 1
            else:
                processing_count += 1
        else:
            # No status yet - still processing
            processing_count += 1

    return success_count, failed_count, processing_count


def _update_document_image_metadata_robust(
    document: Document,
    total_images: int,
    success_count: int,
    failed_count: int,
    method: str,
) -> None:
    """
    Update document metadata with actual image counts to prevent negative counts.

    Args:
        document: Document to update
        total_images: Actual number of image chunks found
        success_count: Number of successful images
        failed_count: Number of failed images
        method: Completion check method used
    """
    if not document.document_metadata:
        document.document_metadata = {}

    if "image_summary" not in document.document_metadata:
        document.document_metadata["image_summary"] = {}

    # Always use actual counts, never expected counts
    document.document_metadata["image_summary"].update(
        {
            "total_images": total_images,  # This is what we actually found
            "successful_images": success_count,
            "failed_images": failed_count,
            "processing_images": total_images - success_count - failed_count,
            "last_checked": datetime.now(UTC).isoformat(),
            "count_source": "actual_chunks",  # Flag to indicate this is actual count
            "completion_method": method,  # Track which method was used
        }
    )

    # Remove waiting flag if all done
    if success_count + failed_count == total_images:
        document.document_metadata["waiting_for_images"] = False
        document.document_metadata["completed_via"] = method

    attributes.flag_modified(document, "document_metadata")


def _finalize_document_completion(
    db, document: Document, final_status: DocumentProcessingStatusEnum, method: str
) -> None:
    """
    Finalize document with completion status.

    Args:
        db: Database session
        document: Document to finalize
        final_status: Final status to set
        method: Completion method used
    """
    document.processing_status = final_status
    document.processed_at = datetime.now(UTC)

    if document.document_metadata:
        document.document_metadata["waiting_for_images"] = False
        document.document_metadata["completed_at"] = datetime.now(UTC).isoformat()
        document.document_metadata["completion_method"] = method

    attributes.flag_modified(document, "document_metadata")
    db.add(document)


def _get_advisory_lock_id(document_id: str) -> int:
    """
    Convert document ID to a consistent 64-bit integer for PostgreSQL advisory locks.

    PostgreSQL advisory locks use bigint (64-bit), so we need to ensure consistent mapping.

    Args:
        document_id: Document ID string

    Returns:
        64-bit integer for advisory lock
    """
    # Use SHA-256 hash to get consistent integer from document_id
    hash_obj = hashlib.sha256(f"doc_completion:{document_id}".encode())
    # Take first 8 bytes and convert to signed 64-bit integer
    hash_bytes = hash_obj.digest()[:8]
    lock_id = int.from_bytes(hash_bytes, byteorder="big", signed=True)
    return lock_id


def _get_current_task_id() -> str:
    """Get current Celery task ID or generate a fallback."""
    try:
        if celery.current_task and hasattr(celery.current_task, "request"):
            return f"task:{celery.current_task.request.id}"
        else:
            logger.warning("No current task ID found, returning fallback task ID")
            return f"task:unknown:{int(time.time())}"
    except Exception as e:
        logger.warning(
            f"Failed to get current task ID: {str(e)}, returning fallback task ID"
        )
        return f"task:fallback:{int(time.time())}"


def check_split_completion_with_fallback(
    parent_document_id: str, split_id: str, user_id: Optional[uuid.UUID] = None
) -> None:
    """
    Robust split completion check with Redis primary and PostgreSQL fallback.

    This checks if a specific split is complete and if all splits for a document are done.

    Args:
        parent_document_id: The ID of the parent document
        split_id: The ID of the split that completed
        user_id: The ID of the user for status propagation
    """
    logger.debug(
        f"Starting robust split completion check for split {split_id} in document {parent_document_id}"
    )

    # Try Redis first
    if _try_redis_split_completion_check(parent_document_id, split_id, user_id):
        return

    # Fall back to PostgreSQL advisory locks
    logger.info(
        f"Redis unavailable, using PostgreSQL advisory locks for split {split_id} completion check"
    )
    _check_split_completion_with_advisory_lock(parent_document_id, split_id, user_id)


def _try_redis_split_completion_check(
    parent_document_id: str, split_id: str, user_id: Optional[uuid.UUID]
) -> bool:
    """
    Try Redis-based split completion check.

    Args:
        parent_document_id: Document ID to check
        split_id: Split ID that completed
        user_id: User ID for notifications

    Returns:
        True if Redis check was successful, False if should fall back to PostgreSQL
    """
    redis_client = None
    completion_lock_key = None

    try:
        redis_client = get_redis_client_sync()
        # Use split-specific lock key to avoid blocking document-level checks
        completion_lock_key = (
            f"split_completion_check:document:{parent_document_id}:split:{split_id}"
        )

        # Test Redis connectivity
        redis_client.ping()

        current_task_id = _get_current_task_id()

        # Try to acquire Redis lock with retry mechanism
        lock_acquired = redis_client.set(
            completion_lock_key,
            current_task_id,
            nx=True,
            ex=120,  # 2 minute expiry
        )

        if not lock_acquired:
            logger.debug(
                f"Redis split completion check for split {split_id} already in progress - retrying"
            )

            # Wait and retry up to 3 times with 5-second intervals
            for attempt in range(1, 4):  # attempts 1, 2, 3
                logger.debug(
                    f"Waiting for split completion lock (attempt {attempt}/3)..."
                )
                time.sleep(5)  # Wait 5 seconds

                lock_acquired = redis_client.set(
                    completion_lock_key,
                    current_task_id,
                    nx=True,
                    ex=120,
                )

                if lock_acquired:
                    logger.info(
                        f"Acquired Redis split completion lock for split {split_id} on retry attempt {attempt}"
                    )
                    break

            if not lock_acquired:
                logger.warning(
                    f"Could not acquire Redis split completion lock for split {split_id} after 3 retries - falling back to PostgreSQL"
                )
                return False  # Fall back to PostgreSQL advisory locks

        logger.debug(f"Acquired Redis split completion lock for split {split_id}")

        try:
            # Perform the actual split completion check
            _perform_split_completion_check(
                parent_document_id, split_id, user_id, "redis"
            )
            return True

        finally:
            # Always release the Redis lock
            try:
                redis_client.delete(completion_lock_key)
                logger.debug(
                    f"Released Redis split completion lock for split {split_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to release Redis split lock: {str(e)}")

    except Exception as e:
        logger.warning(
            f"Redis split completion check failed for split {split_id}: {str(e)}"
        )
        return False  # Fall back to PostgreSQL
    finally:
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis client: {str(e)}")


def _check_split_completion_with_advisory_lock(
    parent_document_id: str, split_id: str, user_id: Optional[uuid.UUID] = None
) -> None:
    """
    Check split completion using PostgreSQL advisory locks as fallback.

    Args:
        parent_document_id: Document ID to check
        split_id: Split ID that completed
        user_id: User ID for notifications
    """
    # Use split-specific lock ID
    lock_id = _get_advisory_lock_id(f"split:{split_id}:doc:{parent_document_id}")

    with SyncSessionLocal() as db:
        try:
            # Try to acquire PostgreSQL advisory lock
            result = db.execute(
                text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": lock_id}
            )
            lock_acquired = result.scalar()

            if not lock_acquired:
                logger.debug(
                    f"Split {split_id} completion check already in progress "
                    f"(advisory lock {lock_id} held by another session)"
                )
                return

            logger.debug(f"Acquired advisory lock {lock_id} for split {split_id}")

            try:
                # Perform the split completion check
                _perform_split_completion_check(
                    parent_document_id, split_id, user_id, "postgresql"
                )

            finally:
                # Always release the advisory lock
                db.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": lock_id}
                )
                logger.debug(f"Released advisory lock {lock_id} for split {split_id}")

        except Exception as e:
            logger.error(
                f"Error in PostgreSQL split completion check for split {split_id}: {str(e)}",
                exc_info=True,
            )
            raise


def _perform_split_completion_check(
    parent_document_id: str, split_id: str, user_id: Optional[uuid.UUID], method: str
) -> None:
    """
    Perform the actual split completion check logic.

    This checks if the specific split is done with its images, and if all splits
    for the document are complete.

    Args:
        parent_document_id: Document ID to check
        split_id: Split ID that completed
        user_id: User ID for notifications
        method: Locking method used ("redis" or "postgresql")
    """
    with SyncSessionLocal() as db:
        try:
            with db.begin():
                # First, check if this specific split is complete
                split_record = (
                    db.query(FileSplit)
                    .filter(FileSplit.id == uuid.UUID(split_id))
                    .with_for_update()
                    .first()
                )

                if not split_record:
                    logger.warning(f"Split {split_id} not found")
                    return

                # Get split image chunks to verify completion
                split_image_chunks = (
                    db.query(DocumentChunk)
                    .filter(
                        DocumentChunk.document_id == uuid.UUID(parent_document_id),
                        DocumentChunk.split_id == uuid.UUID(split_id),
                        DocumentChunk.chunk_type == ChunkTypeEnum.ImageDescription,
                    )
                    .all()
                )

                if split_image_chunks:
                    # Check if all images in this split are complete
                    split_success, split_failed, split_processing = (
                        _count_image_chunk_statuses(split_image_chunks)
                    )

                    # Get original image count for this split
                    split_original_count = _get_split_original_image_count(split_record)
                    split_chunks_found = len(split_image_chunks)
                    split_completed_count = split_success + split_failed

                    # Handle edge case: if no original count found, use chunks found as fallback
                    if split_original_count == 0 and split_chunks_found > 0:
                        logger.warning(
                            f"Split {split_id} has no original image count in metadata, "
                            f"using chunks found ({split_chunks_found}) as fallback"
                        )
                        split_original_count = split_chunks_found

                    # Handle edge case where completed > original (same fix as document level)
                    if split_completed_count > split_original_count:
                        logger.warning(
                            f"Split {split_id} has MORE completed chunks ({split_completed_count}) "
                            f"than original extracted ({split_original_count}) - using completed count as new total"
                        )
                        split_actual_total = split_completed_count
                        split_actual_processing = 0
                    else:
                        split_actual_total = split_original_count
                        split_actual_processing = (
                            split_original_count - split_completed_count
                        )

                    # Always log the current status (whether complete or not)
                    logger.info(
                        f"🔍 Split {split_id} IMAGES status ({method}): "
                        f"Total: {split_actual_total}, "
                        f"Completed: {split_completed_count} ({split_success} successful, {split_failed} failed), "
                        f"Processing: {split_actual_processing}"
                    )

                    if split_actual_processing > 0:
                        logger.debug(
                            f"⏳ Split {split_id} waiting - {split_actual_processing} images still processing"
                        )
                        return  # Split not ready yet

                    logger.info(
                        f"✅ Split {split_id} IMAGES complete ({method}): All images finished!"
                    )

                    # Update split status to Success now that all images are complete
                    _update_split_status_after_completion(
                        db, split_record, split_id, split_success, split_failed
                    )

                # Now check if all splits for the document are complete (get parent document)
                parent_doc = (
                    db.query(Document)
                    .filter(Document.id == uuid.UUID(parent_document_id))
                    .with_for_update()
                    .first()
                )

                if not parent_doc:
                    logger.warning(f"Parent document {parent_document_id} not found")
                    return

                # Update document-level summary (only if split completed)
                if split_actual_processing == 0:
                    _update_document_level_summary_robust(
                        db, parent_doc, split_id, split_success, split_failed
                    )

                # Get all splits for this document
                all_splits = (
                    db.query(FileSplit)
                    .filter(
                        FileSplit.original_file_id == parent_doc.file_id,
                        FileSplit.dataset_id == parent_doc.dataset_id,
                    )
                    .all()
                )

                if not all_splits:
                    logger.debug(f"No splits found for document {parent_document_id}")
                    return

                # Count split completion statuses
                completed_splits = 0
                successful_splits = 0
                failed_splits = 0

                for split in all_splits:
                    if split.status in [
                        SplitFileStatusType.Success,
                        SplitFileStatusType.Failed,
                    ]:
                        completed_splits += 1
                        if split.status == SplitFileStatusType.Success:
                            successful_splits += 1
                        else:
                            failed_splits += 1

                total_splits = len(all_splits)

                logger.info(
                    f"📊 Document {parent_document_id} SPLITS status ({method}): "
                    f"{completed_splits}/{total_splits} splits complete ({successful_splits} successful, {failed_splits} failed)"
                )

                # Check if all splits are complete
                if completed_splits == total_splits:
                    # Determine final status
                    if failed_splits == total_splits:
                        final_status = DocumentProcessingStatusEnum.Failed
                    else:
                        final_status = DocumentProcessingStatusEnum.Success

                    logger.info(
                        f"🎉 Document {parent_document_id} COMPLETE - "
                        f"all {total_splits} splits finished, marking as {final_status} (method: {method})"
                    )

                    # Update document metadata
                    if not parent_doc.document_metadata:
                        parent_doc.document_metadata = {}

                    parent_doc.document_metadata.update(
                        {
                            "split_completion": {
                                "total_splits": total_splits,
                                "successful_splits": successful_splits,
                                "failed_splits": failed_splits,
                                "completed_at": datetime.now(UTC).isoformat(),
                                "completion_method": method,
                            }
                        }
                    )

                    # Finalize document
                    parent_doc.processing_status = final_status
                    parent_doc.processed_at = datetime.now(UTC)
                    parent_doc.successful_splits_count = successful_splits
                    parent_doc.total_splits_count = total_splits

                    attributes.flag_modified(parent_doc, "document_metadata")
                    db.add(parent_doc)

                    # Store propagation info for after commit
                    propagation_info = {
                        "document_id": parent_doc.id,
                        "dataset_id": parent_doc.dataset_id,
                        "status": parent_doc.processing_status,
                        "user_id": user_id,
                    }
                else:
                    remaining = total_splits - completed_splits
                    logger.debug(
                        f"⏳ Document {parent_document_id} waiting - {remaining} splits still processing"
                    )
                    propagation_info = None

            db.commit()

            # Propagate status AFTER commit so separate session sees the changes
            if propagation_info and propagation_info["user_id"]:
                logger.info(
                    f"🔔 Propagating completion status for document {propagation_info['document_id']} to user {propagation_info['user_id']}"
                )
                logger.info(
                    f"📄 Document details: dataset_id={propagation_info['dataset_id']}, status={propagation_info['status']}"
                )
                try:
                    # Create separate session for propagation to avoid transaction conflicts
                    with SyncSessionLocal() as propagation_db:
                        result = propagate_ingestion_status(
                            propagation_db,
                            "document",
                            propagation_info["document_id"],
                            propagation_info["user_id"],
                        )
                        logger.info(f"📡 Propagation function returned: {result}")
                    logger.info(
                        f"✅ Successfully propagated completion status for document {propagation_info['document_id']}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Failed to propagate completion status for document {propagation_info['document_id']}: {str(e)}",
                        exc_info=True,
                    )
            elif propagation_info:
                logger.warning(
                    f"⚠️ No user_id provided for document {propagation_info['document_id']} - skipping propagation"
                )

        except Exception as e:
            logger.error(
                f"Error in split completion check for split {split_id}: {str(e)}",
                exc_info=True,
            )
            raise


# Utility function for manual recovery/debugging
def force_document_completion_check(
    document_id: str, user_id: Optional[uuid.UUID] = None
) -> dict:
    """
    Force a completion check for a document (useful for debugging/recovery).

    Args:
        document_id: Document ID to check
        user_id: User ID for notifications

    Returns:
        Dictionary with check results
    """
    logger.info(f"Force checking completion for document {document_id}")

    try:
        check_document_completion_with_fallback(document_id, user_id)
        return {
            "success": True,
            "document_id": document_id,
            "message": "Completion check executed successfully",
        }
    except Exception as e:
        logger.error(f"Force completion check failed for {document_id}: {str(e)}")
        return {"success": False, "document_id": document_id, "error": str(e)}
