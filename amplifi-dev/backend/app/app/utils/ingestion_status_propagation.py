"""Utility functions for propagating ingestion status from bottom to top in the hierarchy.

This version uses the Document model (v2) for status tracking instead of FileIngestion.
"""

from datetime import UTC, datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.be_core.config import settings
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.dataset_model import Dataset
from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.models.file_split_model import FileSplit, SplitFileStatusType


def update_split_status_from_images(
    db: Session, document_id: UUID, split_id: str
) -> bool:
    """
    Update split status based on completion of its image processing.

    Args:
        db: Database session
        document_id: ID of the parent document
        split_id: ID of the specific split

    Returns:
        bool: True if split status was updated, False otherwise
    """
    try:
        # Get the split record
        split = db.query(FileSplit).filter(FileSplit.id == UUID(split_id)).first()
        if not split:
            logger.error(f"Split {split_id} not found")
            return False

        # Check if all images from this split are processed
        split_image_chunks = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_type == ChunkTypeEnum.ImageDescription,
                DocumentChunk.split_id == split_id,
            )
            .all()
        )

        if not split_image_chunks:
            # No images from this split, mark as success
            split.status = SplitFileStatusType.Success
            db.commit()
            logger.info(f"Split {split_id} marked as Success - no images extracted")
            return True

        # Check if all images from this split are processed
        all_images_processed = True
        for chunk in split_image_chunks:
            if chunk.chunk_metadata and "status" in chunk.chunk_metadata:
                status = chunk.chunk_metadata["status"]
                if status not in [
                    DocumentProcessingStatusEnum.Success.value,
                    DocumentProcessingStatusEnum.Failed.value,
                ]:
                    all_images_processed = False
                    break
            else:
                all_images_processed = False
                break

        if all_images_processed:
            # Update split status to Success
            split.status = SplitFileStatusType.Success
            db.commit()
            logger.info(
                f"Split {split_id} marked as Success after all images processed"
            )
            return True

        return False

    except Exception as e:
        logger.error(f"Error updating split status from images: {str(e)}")
        return False


def update_document_status_from_splits(db: Session, document_id: UUID) -> bool:
    """
    Update document status based on the status of all its splits.
    """
    # Get the document record with FOR UPDATE lock
    document = (
        db.query(Document).filter(Document.id == document_id).with_for_update().first()
    )
    if not document:
        logger.error(f"Document {document_id} not found")
        return False

    # Check if document is ready for status update
    if not _is_document_ready_for_split_status_update(document, document_id):
        return False

    # Get and validate splits
    splits = _get_document_splits(db, document)
    if not splits:
        return False

    # Validate splits belong to this document
    if not _validate_splits_belong_to_document(document, splits):
        logger.error(f"Split validation failed for document {document_id}")
        return False

    # Check if any splits are waiting for images
    if _are_splits_waiting_for_images(splits):
        return False

    # Calculate split status counts
    split_counts = _calculate_split_status_counts(splits)

    # Update document with split information
    _update_document_split_metadata(document, split_counts)

    # Determine and apply new status
    old_status = document.processing_status
    _update_document_status_from_split_counts(document, split_counts, document_id)

    db.commit()

    # Return True if status actually changed
    return old_status != document.processing_status


def _is_document_ready_for_split_status_update(
    document: Document, document_id: UUID
) -> bool:
    """Check if document is ready for split status update (not waiting for images)."""
    if not document.document_metadata:
        return True

    # Check if document is waiting for images
    if document.document_metadata.get("waiting_for_images", False):
        logger.info(
            f"Document {document_id} is waiting for images - not updating status from splits yet"
        )
        return False

    # Check image summary status
    image_summary = document.document_metadata.get("image_summary", {})
    if image_summary.get("processing_status") == "processing":
        logger.info(
            f"Document {document_id} has images still processing - not updating status from splits yet"
        )
        return False

    return True


def _get_document_splits(db: Session, document: Document) -> Optional[list]:
    """Get all splits for the document."""
    splits = (
        db.query(FileSplit)
        .filter(
            FileSplit.document_id == document.id,
            FileSplit.dataset_id == document.dataset_id,
        )
        .all()
    )

    if not splits:
        # No splits means this was direct ingestion, don't change status based on splits
        logger.info(
            f"No splits found for document {document.id} - this was direct ingestion"
        )
        return None

    # Debug logging to identify cross-dataset issues
    logger.info(f"Found {len(splits)} splits for document {document.id}")
    for split in splits:
        logger.info(
            f"  Split {split.id}: dataset={split.dataset_id}, "
            f"original_file={split.original_file_id}, document={split.document_id}"
        )

    return splits


def _are_splits_waiting_for_images(splits: list) -> bool:
    """Check if any splits are waiting for images."""
    for split in splits:
        if split.split_metadata and split.split_metadata.get(
            "waiting_for_images", False
        ):
            logger.info(
                f"Split {split.id} is waiting for images - not updating document status yet"
            )
            return True
    return False


def _calculate_split_status_counts(splits: list) -> dict:
    """Calculate counts of splits by their status."""
    total_splits = len(splits)

    # Log split details for debugging
    logger.info(f"Calculating status for {total_splits} splits:")
    for split in splits:
        logger.info(
            f"  Split {split.id}: status={split.status}, document_id={split.document_id}"
        )

    successful_splits = sum(
        1 for split in splits if split.status == SplitFileStatusType.Success
    )
    failed_splits = sum(
        1
        for split in splits
        if split.status in [SplitFileStatusType.Failed, SplitFileStatusType.Exception]
    )
    processing_splits = total_splits - successful_splits - failed_splits

    counts = {
        "total": total_splits,
        "successful": successful_splits,
        "failed": failed_splits,
        "processing": processing_splits,
    }

    logger.info(f"Split status counts: {counts}")
    return counts


def _update_document_split_metadata(document: Document, split_counts: dict) -> None:
    """Update document model fields and metadata with split counts."""
    # Update document model fields
    old_successful = document.successful_splits_count
    old_total = document.total_splits_count

    document.successful_splits_count = split_counts["successful"]
    document.total_splits_count = split_counts["total"]

    # Log the update for debugging
    logger.info(
        f"Document {document.id} split counts updated: "
        f"successful {old_successful} → {split_counts['successful']}, "
        f"total {old_total} → {split_counts['total']}"
    )

    # Update document metadata counters
    if not document.document_metadata:
        document.document_metadata = {}

    document.document_metadata.update(
        {
            "last_split_status_update": datetime.now(UTC).isoformat(),
        }
    )


def _update_document_status_from_split_counts(
    document: Document, split_counts: dict, document_id: UUID
) -> None:
    """Determine and update document status based on split counts."""
    # Log the counts for better debugging
    logger.info(
        f"Document {document_id} split counts: "
        f"successful={split_counts['successful']}, "
        f"failed={split_counts['failed']}, "
        f"processing={split_counts['processing']}, "
        f"total={split_counts['total']}"
    )

    # Don't overwrite Failed/Exception status
    if document.processing_status in [
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        logger.info(
            f"Document {document_id} already in failed state ({document.processing_status}), not updating"
        )
        return

    if split_counts["failed"] > 0:
        document.processing_status = DocumentProcessingStatusEnum.Failed
        document.processed_at = datetime.now(UTC)
        logger.warning(
            f"Document {document_id} marked as Failed - {split_counts['failed']} out of {split_counts['total']} splits failed"
        )
        return

    elif split_counts["processing"] > 0:
        document.processing_status = DocumentProcessingStatusEnum.Processing
        logger.info(
            f"Document {document_id} still processing - {split_counts['processing']} splits remaining"
        )
        return

    elif split_counts["successful"] == split_counts["total"]:
        document.processing_status = DocumentProcessingStatusEnum.Success
        document.processed_at = datetime.now(UTC)
        logger.info(
            f"Document {document_id} marked as Success - all {split_counts['total']} splits successful"
        )
        return


def update_document_status_from_images(db: Session, document_id: UUID) -> bool:
    """
    Update document status based on its child images' status.
    Refactored to reduce cyclomatic complexity by using helper functions.

    Args:
        db: Database session
        document_id: ID of the document to update

    Returns:
        bool: True if status was updated, False otherwise
    """
    # Get document and validate
    document = _get_document_for_image_update(db, document_id)
    if not document:
        return False

    # Get child image chunks
    child_image_chunks = _get_child_image_chunks(db, document_id)
    if not child_image_chunks:
        return False

    # Check ingestion type and handle accordingly
    if _is_split_based_ingestion(db, document):
        return _handle_split_based_image_status_update(
            db, document_id, child_image_chunks
        )
    else:
        return _handle_direct_image_status_update(db, document, child_image_chunks)


def _get_document_for_image_update(
    db: Session, document_id: UUID
) -> Optional[Document]:
    """Get document for image status update with validation."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        logger.error(f"Document {document_id} not found")
        return None
    return document


def _get_child_image_chunks(db: Session, document_id: UUID) -> Optional[list]:
    """Get all child image chunks for the document."""
    child_image_chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_type == ChunkTypeEnum.ImageDescription,
            DocumentChunk.chunk_metadata.contains(
                {"parent_document_id": str(document_id)}
            ),
        )
        .all()
    )

    if not child_image_chunks:
        # No child images found
        return None

    return child_image_chunks


def _is_split_based_ingestion(db: Session, document: Document) -> bool:
    """Check if this document uses split-based ingestion."""
    splits_exist = (
        db.query(FileSplit)
        .filter(
            FileSplit.original_file_id == document.file_id,
            FileSplit.dataset_id == document.dataset_id,
            FileSplit.document_id == document.id,  # ← Add this filter
        )
        .first()
        is not None
    )

    logger.info(f"Document {document.id} split-based ingestion: {splits_exist}")
    return splits_exist


def _handle_split_based_image_status_update(
    db: Session, document_id: UUID, child_image_chunks: list
) -> bool:
    """Handle image status updates for split-based ingestion."""
    updated_splits = set()

    for chunk in child_image_chunks:
        split_id = chunk.split_id
        if split_id and split_id not in updated_splits:
            if update_split_status_from_images(db, document_id, split_id):
                updated_splits.add(split_id)

    # Then update document status based on all splits
    return update_document_status_from_splits(db, document_id)


def _handle_direct_image_status_update(
    db: Session, document: Document, child_image_chunks: list
) -> bool:
    """Handle image status updates for direct ingestion."""
    # Calculate image status counts
    image_counts = _calculate_image_status_counts(child_image_chunks)

    # Update document status based on counts
    old_status = document.processing_status
    _update_document_status_from_image_counts(document, image_counts)

    db.commit()
    return old_status != document.processing_status


def _calculate_image_status_counts(child_image_chunks: list) -> dict:
    """Calculate counts of images by their status."""
    total_images = len(child_image_chunks)
    successful_images = 0
    failed_images = 0

    for chunk in child_image_chunks:
        if chunk.chunk_metadata and "status" in chunk.chunk_metadata:
            status = chunk.chunk_metadata["status"]
            if status == DocumentProcessingStatusEnum.Success.value:
                successful_images += 1
            elif status in [
                DocumentProcessingStatusEnum.Failed.value,
                DocumentProcessingStatusEnum.Exception.value,
            ]:
                failed_images += 1

    return {
        "total": total_images,
        "successful": successful_images,
        "failed": failed_images,
        "processing": total_images - successful_images - failed_images,
    }


def _update_document_status_from_image_counts(
    document: Document, image_counts: dict
) -> None:
    """Determine and update document status based on image counts."""
    # Don't overwrite Failed/Exception status with Success - preserve failure state
    if document.processing_status in [
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        logger.info(
            f"Document {document.id} already in failed state ({document.processing_status}), not updating to success"
        )
        return

    if image_counts["successful"] == image_counts["total"]:
        # All images processed successfully
        document.processing_status = DocumentProcessingStatusEnum.Success
        document.processed_at = datetime.now(UTC)

    elif image_counts["failed"] == image_counts["total"]:
        # All images failed
        document.processing_status = DocumentProcessingStatusEnum.Failed
        document.processed_at = datetime.now(UTC)

    elif image_counts["successful"] + image_counts["failed"] == image_counts["total"]:
        # Mixed success/failure but all processed
        if image_counts["successful"] > 0:
            document.processing_status = DocumentProcessingStatusEnum.Success
        else:
            document.processing_status = DocumentProcessingStatusEnum.Failed
        document.processed_at = datetime.now(UTC)

    else:
        # Still processing
        document.processing_status = DocumentProcessingStatusEnum.Processing


def update_document_status_from_audio(db: Session, document_id: UUID) -> bool:
    """
    Update document status based on its audio chunks' status.
    Audio files don't have splits, so we check the chunks directly.

    Args:
        db: Database session
        document_id: ID of the document to update

    Returns:
        bool: True if status was updated, False otherwise
    """
    # Get the document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        logger.error(f"Document {document_id} not found")
        return False

    # Only process audio documents
    if document.document_type != DocumentTypeEnum.Audio:
        return False

    # Find all audio-related chunks for this document
    audio_chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_type.in_(
                [ChunkTypeEnum.AudioSegment, ChunkTypeEnum.Speaker]
            ),
        )
        .all()
    )

    if not audio_chunks:
        # No audio chunks found yet, processing might still be in progress
        return False

    # For audio files, if chunks exist and document isn't already failed,
    # we can consider it successful since the audio ingestion task
    # only creates chunks after successful transcription
    if document.processing_status not in [
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        document.processing_status = DocumentProcessingStatusEnum.Success
        document.processed_at = datetime.now(UTC)
        db.commit()
        logger.info(
            f"Updated audio document {document_id} status to Success based on chunks"
        )
        return True

    return False


def update_document_status_from_video(db: Session, document_id: UUID) -> bool:
    """
    Update document status based on its video chunks' status and validate completeness.
    Video files don't have splits, so we check the chunks directly.

    Args:
        db: Database session
        document_id: ID of the document to update

    Returns:
        bool: True if status was updated, False otherwise
    """
    # Get the document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        logger.error(f"Document {document_id} not found")
        return False

    # Only process video documents
    if document.document_type != DocumentTypeEnum.Video:
        return False

    # Video documents are marked as Success by the video_ingestion_task itself
    # after successful processing. This function is mainly for validation and
    # ensuring consistency in case of edge cases.

    # If document is already in a terminal state (Success/Failed/Exception),
    # don't change it - the video ingestion task has already set the final status
    if document.processing_status in [
        DocumentProcessingStatusEnum.Success,
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        logger.debug(
            f"Video document {document_id} already in terminal state: {document.processing_status}"
        )
        return False

    # Find all video segment chunks for this document
    video_chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_type == ChunkTypeEnum.VideoSegment,
        )
        .all()
    )

    if not video_chunks:
        # No video segment chunks found - this indicates processing failed
        # If propagate_ingestion_status is called, processing should be complete
        # The absence of video segment chunks means the video processing failed to create any segments
        if document.processing_status == DocumentProcessingStatusEnum.Processing:
            logger.warning(
                f"Video document {document_id} has no video segment chunks but status is Processing - "
                f"marking as Failed (processing completed without creating video segments)"
            )
            document.processing_status = DocumentProcessingStatusEnum.Failed
            document.processed_at = datetime.now(UTC)
            document.error_message = (
                "Video processing completed but no video segment chunks were created"
            )
            db.commit()
            return True
        else:
            logger.debug(
                f"Video document {document_id} has no video segment chunks and status is {document.processing_status}"
            )
            return False

    # Validate chunk count against expected segments if metadata is available
    expected_segments = None
    if document.document_metadata:
        expected_segments = document.document_metadata.get("total_segments")
        video_duration = document.document_metadata.get("video_duration", 0)
        segment_length = document.document_metadata.get("segment_length", 30)

        if expected_segments:
            actual_video_segments = len(video_chunks)

            if actual_video_segments != expected_segments:
                logger.warning(
                    f"Video document {document_id} chunk count mismatch: "
                    f"expected {expected_segments} segments, found {actual_video_segments}. "
                    f"Duration: {video_duration}s, Segment length: {segment_length}s"
                )
                # Don't mark as failed, but log the discrepancy for monitoring
                # The video ingestion task is the authoritative source for success/failure

    # If we reach here, chunks exist but document is still in Processing state
    # This might happen in edge cases where the video ingestion task was interrupted
    # after creating chunks but before updating document status
    logger.info(
        f"Video document {document_id} has {len(video_chunks)} video segment chunks but status is {document.processing_status}. "
        f"Expected segments: {expected_segments or 'unknown'}"
    )

    # Don't automatically mark as Success - let the video ingestion task handle final status
    # This function is primarily for validation and monitoring
    return False


def update_dataset_status_from_documents(db: Session, dataset_id: UUID) -> bool:
    """
    Update dataset status based on the status of all its documents.
    Refactored to reduce cyclomatic complexity by using helper functions.

    Args:
        db: Database session
        dataset_id: ID of the dataset to update

    Returns:
        bool: True if status was updated, False otherwise
    """
    # Get and validate dataset
    dataset = _get_dataset_for_status_update(db, dataset_id)
    if not dataset:
        return False

    # Get and validate documents
    documents = _get_dataset_documents(db, dataset_id)
    if not documents:
        return False

    # Calculate document status counts
    status_counts = _calculate_document_status_counts(documents)

    # Prepare and update dataset metadata
    metadata = _prepare_dataset_metadata(dataset)
    _update_dataset_ingestion_stats(metadata, status_counts, len(documents))

    # Determine and apply overall dataset status
    dataset_status = _determine_dataset_status(status_counts, len(documents))
    _finalize_dataset_metadata_update(dataset, metadata, dataset_status)

    db.commit()
    return True


def _get_dataset_for_status_update(db: Session, dataset_id: UUID) -> Optional[Dataset]:
    """Get dataset for status update with validation."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        logger.error(f"Dataset {dataset_id} not found")
        return None
    return dataset


def _get_dataset_documents(db: Session, dataset_id: UUID) -> Optional[list]:
    """Get all non-deleted documents for the dataset."""
    documents = (
        db.query(Document)
        .filter(
            Document.dataset_id == dataset_id,
            Document.deleted_at.is_(None),  # Only consider non-deleted documents
        )
        .all()
    )

    if not documents:
        logger.warning(f"No documents found for dataset {dataset_id}")
        return None

    return documents


def _calculate_document_status_counts(documents: list) -> dict:
    """Calculate counts of documents by their processing status."""
    status_counts = {
        DocumentProcessingStatusEnum.Success: 0,
        DocumentProcessingStatusEnum.Failed: 0,
        DocumentProcessingStatusEnum.Exception: 0,
        DocumentProcessingStatusEnum.Processing: 0,
        DocumentProcessingStatusEnum.Queued: 0,
        DocumentProcessingStatusEnum.Not_Started: 0,
        DocumentProcessingStatusEnum.Extracting: 0,
        DocumentProcessingStatusEnum.ExtractionCompleted: 0,
        DocumentProcessingStatusEnum.Splitting: 0,
    }

    for doc in documents:
        if doc.processing_status in status_counts:
            status_counts[doc.processing_status] += 1

    return status_counts


def _prepare_dataset_metadata(dataset: Dataset) -> dict:
    """Parse and prepare dataset metadata for updates."""
    metadata = dataset.description
    if not isinstance(metadata, dict):
        try:
            # If it's a string, try to parse as JSON
            import json

            metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            metadata = {}

    return metadata


def _update_dataset_ingestion_stats(
    metadata: dict, status_counts: dict, total_documents: int
) -> None:
    """Update dataset metadata with ingestion statistics."""
    ingestion_stats = {
        "ingestion_stats": {
            "total_documents": total_documents,
            "successful_documents": status_counts[DocumentProcessingStatusEnum.Success],
            "failed_documents": status_counts[DocumentProcessingStatusEnum.Failed]
            + status_counts[DocumentProcessingStatusEnum.Exception],
            "processing_documents": status_counts[
                DocumentProcessingStatusEnum.Processing
            ]
            + status_counts[DocumentProcessingStatusEnum.Extracting]
            + status_counts[DocumentProcessingStatusEnum.ExtractionCompleted]
            + status_counts[DocumentProcessingStatusEnum.Splitting],
            "not_started_documents": status_counts[
                DocumentProcessingStatusEnum.Not_Started
            ]
            + status_counts[DocumentProcessingStatusEnum.Queued],
            "last_updated": datetime.now(UTC).isoformat(),
        }
    }

    # Add ingestion stats to metadata
    if isinstance(metadata, dict):
        metadata.update(ingestion_stats)


def _determine_dataset_status(status_counts: dict, total_documents: int) -> str:
    """Determine overall dataset ingestion status based on document counts."""
    # All documents not started
    if (
        status_counts[DocumentProcessingStatusEnum.Not_Started]
        + status_counts[DocumentProcessingStatusEnum.Queued]
    ) == total_documents:
        return "Not Started"

    # Count documents in terminal states (Success, Failed, Exception)
    terminal_count = (
        status_counts[DocumentProcessingStatusEnum.Success]
        + status_counts[DocumentProcessingStatusEnum.Failed]
        + status_counts[DocumentProcessingStatusEnum.Exception]
    )

    # Only determine final status when ALL documents are in terminal states
    if terminal_count == total_documents:
        # All documents are in terminal states - now determine final status
        if status_counts[DocumentProcessingStatusEnum.Success] == total_documents:
            return "Success"  # All succeeded
        elif (
            status_counts[DocumentProcessingStatusEnum.Failed]
            + status_counts[DocumentProcessingStatusEnum.Exception]
        ) > 0:
            return "Failed"  # At least one failed
        else:
            return "Partial"  # Mixed success/failure (shouldn't happen but safety)
    else:
        # Some documents are still in non-terminal states (Processing, Extracting, etc.)
        return "Processing"


def _finalize_dataset_metadata_update(
    dataset: Dataset, metadata: dict, dataset_status: str
) -> None:
    """Finalize dataset metadata update with status - using dedicated columns."""
    # Store ingestion status in dedicated column
    dataset.ingestion_status = dataset_status

    # Store ingestion stats in dedicated JSON column
    if "ingestion_stats" in metadata:
        dataset.ingestion_stats = metadata["ingestion_stats"]

    # Update timestamp
    dataset.ingestion_last_updated = datetime.now(UTC)


def send_ingestion_status_notification(
    user_id: UUID,
    entity_id: UUID,
    status: str,
    entity_type: str,
    details: dict = None,
    error_message: str = None,
):
    """
    Send a centralized WebSocket notification for ingestion status updates.

    Args:
        user_id: ID of the user who initiated the ingestion
        entity_id: ID of the entity (document, file, dataset) being processed
        status: Status of the entity (Success, Failed, Processing, etc.)
        entity_type: Type of entity (document, image, audio, split, dataset)
        details: Additional details to include in the notification
        error_message: Error message if status is Failed
    """
    try:
        import json
        from datetime import UTC, datetime

        from app.api.deps import publish_websocket_message
        from app.schemas.long_task_schema import ITaskType

        class UUIDEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, UUID):
                    # Convert UUID to string
                    return str(obj)
                # Let the default method handle everything else
                return json.JSONEncoder.default(self, obj)

        # Build notification data
        notification_data = {
            "entity_id": str(entity_id),
            "entity_type": entity_type,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Add error message if status is failed
        if status.lower() == "failed" and error_message:
            notification_data["error"] = error_message

        # Add any additional details
        if details:
            # Ensure any UUIDs in the details are properly handled
            sanitized_details = {}
            for key, value in details.items():
                if isinstance(value, UUID):
                    sanitized_details[key] = str(value)
                else:
                    sanitized_details[key] = value
            notification_data.update(sanitized_details)

        # Publish to WebSocket using the UUID-aware encoder
        publish_websocket_message(
            f"{user_id}:{ITaskType.ingestion}",
            json.dumps(notification_data, cls=UUIDEncoder),
        )
        logger.info(
            f"Sent {status} notification for {entity_type} {entity_id} via WebSocket"
        )
        return True
    except Exception as e:
        logger.error(f"Error sending WebSocket notification: {str(e)}", exc_info=True)
        return False


def send_success_notification(
    user_id: UUID, entity_id: UUID, entity_type: str, details: dict = None
):
    """
    Send a success notification for an entity.

    Args:
        user_id: ID of the user who initiated the ingestion
        entity_id: ID of the entity that was processed successfully
        entity_type: Type of entity (document, image, audio, split, dataset)
        details: Additional details to include in the notification
    """
    return send_ingestion_status_notification(
        user_id=user_id,
        entity_id=entity_id,
        status="Success",
        entity_type=entity_type,
        details=details,
    )


def send_failure_notification(
    user_id: UUID,
    entity_id: UUID,
    entity_type: str,
    error_message: str,
    details: dict = None,
):
    """
    Send a failure notification for an entity.

    Args:
        user_id: ID of the user who initiated the ingestion
        entity_id: ID of the entity that failed processing
        entity_type: Type of entity (document, image, audio, split, dataset)
        error_message: Error message describing the failure
        details: Additional details to include in the notification
    """
    return send_ingestion_status_notification(
        user_id=user_id,
        entity_id=entity_id,
        status="Failed",
        entity_type=entity_type,
        details=details,
        error_message=error_message,
    )


def propagate_ingestion_status(
    db: Session, level: str, entity_id: UUID, user_id: Optional[UUID] = None
) -> bool:
    """
    Propagate ingestion status upward through the hierarchy using Document model (v2).
    Refactored to reduce cyclomatic complexity by using helper functions.

    Args:
        db: Database session
        level: Level in the hierarchy to start propagation from ('image', 'audio', 'document', 'split', 'dataset')
        entity_id: ID of the entity at the specified level
        user_id: Optional user ID for sending WebSocket notifications

    Returns:
        bool: True if propagation was successful
    """
    try:
        # Define level handlers mapping
        level_handlers = {
            "image": _handle_image_propagation,
            "audio": _handle_audio_propagation,
            "document": _handle_document_propagation,
            "split": _handle_split_propagation,
            "dataset": _handle_dataset_propagation,
        }

        # Get the appropriate handler for the level
        handler = level_handlers.get(level)
        if not handler:
            logger.error(f"Unknown level: {level}")
            return False

        # Execute the handler
        return handler(db, entity_id, user_id)

    except Exception as e:
        logger.error(f"Error propagating ingestion status: {str(e)}", exc_info=True)
        if user_id:
            send_failure_notification(user_id, entity_id, level, str(e))
        return False


def _get_chunk_by_id(
    db: Session, entity_id: UUID, entity_type: str
) -> Optional[DocumentChunk]:
    """Get a document chunk by ID with error logging."""
    chunk = db.query(DocumentChunk).filter(DocumentChunk.id == entity_id).first()
    if not chunk:
        logger.error(f"{entity_type.capitalize()} chunk {entity_id} not found")
    return chunk


def _get_document_by_id(db: Session, entity_id: UUID) -> Optional[Document]:
    """Get a document by ID with error logging."""
    document = db.query(Document).filter(Document.id == entity_id).first()
    if not document:
        logger.error(f"Document {entity_id} not found")
    return document


def _create_chunk_notification_details(
    chunk: DocumentChunk, include_split_id: bool = False
) -> dict:
    """Create notification details for a chunk."""
    details = {
        "chunk_type": (
            chunk.chunk_type.value if hasattr(chunk, "chunk_type") else "unknown"
        ),
        "document_id": str(chunk.document_id) if chunk.document_id else None,
    }
    if include_split_id:
        details["split_id"] = chunk.split_id if chunk.split_id else None
    return details


def _create_document_notification_details(
    document: Document, include_file_id: bool = False
) -> dict:
    """Create notification details for a document."""
    details = {
        "document_type": (
            document.document_type.value
            if hasattr(document, "document_type")
            else "unknown"
        ),
        "processing_status": (
            document.processing_status.value
            if hasattr(document, "processing_status")
            else "unknown"
        ),
        "dataset_id": str(document.dataset_id) if document.dataset_id else None,
    }
    if include_file_id:
        details["file_id"] = str(document.file_id) if document.file_id else None
    return details


def _send_document_status_notification(
    user_id: UUID, document: Document, entity_id: UUID
) -> None:
    """Send appropriate notification based on document status."""
    details = _create_document_notification_details(document, include_file_id=True)

    if document.processing_status in [
        DocumentProcessingStatusEnum.Success,
        # DocumentProcessingStatusEnum.ExtractionCompleted,
    ]:
        send_success_notification(user_id, entity_id, "document", details)
    elif document.processing_status in [
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        send_failure_notification(
            user_id,
            entity_id,
            "document",
            document.error_message or "Unknown error",
            details,
        )


def _propagate_to_dataset_and_notify(
    db: Session, document: Document, user_id: Optional[UUID], updated: bool
) -> None:
    """Propagate to dataset level and send notification if status was updated."""
    if not document.dataset_id:
        return

    # Propagate to dataset level
    propagate_ingestion_status(db, "dataset", document.dataset_id, user_id)

    # Send document notification if status was updated
    # if user_id and updated:
    #     details = _create_document_notification_details(document)
    #     send_success_notification(user_id, document.id, "document", details)


def _handle_image_propagation(
    db: Session, entity_id: UUID, user_id: Optional[UUID]
) -> bool:
    """Handle propagation for image level entities."""
    # Find the parent document
    chunk = _get_chunk_by_id(db, entity_id, "image")
    if not chunk:
        return False

    # Send notification about the image status
    # if user_id:
    #     details = _create_chunk_notification_details(chunk, include_split_id=True)
    #     send_success_notification(user_id, entity_id, "image", details)

    # Handle parent document propagation
    if chunk.chunk_metadata and "parent_document_id" in chunk.chunk_metadata:
        parent_doc_id = UUID(chunk.chunk_metadata["parent_document_id"])

        # Update document status based on its images
        updated = update_document_status_from_images(db, parent_doc_id)

        # Get the document to find its dataset_id
        document = _get_document_by_id(db, parent_doc_id)
        if document:
            _propagate_to_dataset_and_notify(db, document, user_id, updated)

    return True


def _handle_audio_propagation(
    db: Session, entity_id: UUID, user_id: Optional[UUID]
) -> bool:
    """Handle propagation for audio level entities."""
    # Find the audio chunk
    chunk = _get_chunk_by_id(db, entity_id, "audio")
    if not chunk:
        return False

    # Send notification about the audio chunk status
    # if user_id:
    #     details = _create_chunk_notification_details(chunk)
    #     send_success_notification(user_id, entity_id, "audio", details)

    # Update document status based on its audio chunks
    if chunk.document_id:
        updated = update_document_status_from_audio(db, chunk.document_id)

        # Get the document to find its dataset_id
        document = _get_document_by_id(db, chunk.document_id)
        if document:
            _propagate_to_dataset_and_notify(db, document, user_id, updated)

    return True


def _handle_document_propagation(
    db: Session, entity_id: UUID, user_id: Optional[UUID]
) -> bool:
    """Handle propagation for document level entities."""
    # Get the document
    document = _get_document_by_id(db, entity_id)
    if not document:
        return False

    # Check document type and update status accordingly
    if document.document_type == DocumentTypeEnum.Audio:
        update_document_status_from_audio(db, entity_id)
    elif document.document_type == DocumentTypeEnum.Video:
        update_document_status_from_video(db, entity_id)

    # Send notification about the document status
    if user_id:
        _send_document_status_notification(user_id, document, entity_id)

    # Propagate to dataset level
    if document.dataset_id:
        propagate_ingestion_status(db, "dataset", document.dataset_id, user_id)

    return True


def _handle_split_propagation(
    db: Session, entity_id: UUID, user_id: Optional[UUID]
) -> bool:
    """Handle propagation for split level entities."""
    # Handle splits properly with new schema
    split = db.query(FileSplit).filter(FileSplit.id == entity_id).first()
    if not split:
        logger.error(f"Split {entity_id} not found")
        return False

    # Send notification about the split status
    # if user_id:
    #     details = {
    #         "original_file_id": (
    #             str(split.original_file_id) if split.original_file_id else None
    #         ),
    #         "split_index": split.split_index,
    #         "total_splits": split.total_splits,
    #     }
    #     send_success_notification(user_id, entity_id, "split", details)

    # Find the document for this file and update its status
    document = (
        db.query(Document)
        .filter(
            Document.file_id == split.original_file_id,
            Document.dataset_id == split.dataset_id,
        )
        .first()
    )

    if document:
        updated = update_document_status_from_splits(db, document.id)
        _propagate_to_dataset_and_notify(db, document, user_id, updated)

    return True


def _handle_dataset_propagation(
    db: Session, entity_id: UUID, user_id: Optional[UUID]
) -> bool:
    """Handle propagation for dataset level entities."""
    logger.info(
        f"📊 Processing dataset propagation for dataset {entity_id}, user {user_id}"
    )

    # Update dataset status directly from documents
    updated = update_dataset_status_from_documents(db, entity_id)
    logger.info(f"📈 Dataset status update result: {updated}")

    # Send dataset notification if status was updated
    if user_id and updated:
        logger.info(f"🔔 Checking dataset notification for {entity_id}")
        dataset = db.query(Dataset).filter(Dataset.id == entity_id).first()
        if dataset:
            logger.info(
                f"📊 Dataset found: status={dataset.ingestion_status}, name={dataset.name}"
            )
            _send_dataset_notification(user_id, entity_id, dataset)
        else:
            logger.error(f"❌ Dataset {entity_id} not found for notification")
    else:
        if not user_id:
            logger.warning(
                f"⚠️ No user_id provided for dataset {entity_id} - skipping notification"
            )
        if not updated:
            logger.info(
                f"ℹ️ Dataset {entity_id} status not updated - skipping notification"
            )

    return updated


def cleanup_processing_flag(dataset_id: str, user_id: str):
    """
    Clean up the processing flag when dataset ingestion completes (success or failure).

    Args:
        dataset_id: ID of the dataset that completed processing
        user_id: ID of the user who initiated the ingestion
    """
    try:
        from app.api.deps import get_redis_client_sync

        redis_client = get_redis_client_sync()
        processing_key = f"ingestion:processing:{dataset_id}:{user_id}"

        # Check if the flag exists before deleting (for logging purposes)
        if redis_client.exists(processing_key):
            redis_client.delete(processing_key)
            logger.info(f"Cleaned up processing flag: {processing_key}")
        else:
            logger.debug(
                f"Processing flag already cleaned or not found: {processing_key}"
            )

    except Exception as e:
        logger.warning(
            f"Failed to clean up processing flag for dataset {dataset_id}: {str(e)}"
        )


def _send_dataset_notification(
    user_id: UUID, entity_id: UUID, dataset: Dataset
) -> None:
    """Send dataset notification based on its status."""
    import json

    try:
        status = dataset.ingestion_status or "Unknown"
        stats = dataset.ingestion_stats or {}

        details = {
            "name": dataset.name,
            "workspace_id": str(dataset.workspace_id),
            "ingestion_status": status,
            "ingestion_stats": stats,
        }

        # Send notification based on dataset status
        if status == "Success":
            send_success_notification(user_id, entity_id, "dataset", details)
        elif status == "Failed":
            send_failure_notification(
                user_id, entity_id, "dataset", "Dataset ingestion failed", details
            )
        # else:
        #     # Send a general status update notification
        #     send_ingestion_status_notification(
        #         user_id, entity_id, status, "dataset", details
        #     )

        # Clean up processing flag when dataset reaches final state
        if status in ["Success", "Failed"]:
            cleanup_processing_flag(str(entity_id), str(user_id))
            logger.info(
                f"Dataset {entity_id} ingestion completed with status: {status}"
            )

    except json.JSONDecodeError:
        logger.error(
            f"Unable to parse dataset description as JSON: {dataset.description}"
        )
    except Exception as e:
        logger.error(f"Error in dataset notification: {str(e)}")


def _validate_splits_belong_to_document(document: Document, splits: list) -> bool:
    """Validate that all splits belong to the correct document."""
    for split in splits:
        if split.document_id != document.id:
            logger.error(
                f"Split {split.id} has document_id {split.document_id} but should be {document.id}"
            )
            return False
        if split.original_file_id != document.file_id:
            logger.error(
                f"Split {split.id} has original_file_id {split.original_file_id} but should be {document.file_id}"
            )
            return False
        if split.dataset_id != document.dataset_id:
            logger.error(
                f"Split {split.id} has dataset_id {split.dataset_id} but should be {document.dataset_id}"
            )
            return False

    logger.info(f"All {len(splits)} splits validated for document {document.id}")
    return True


def is_document_failed_due_to_timeout(document: Document) -> bool:
    """
    Check if a document was marked as failed due to timeout.

    Args:
        document: Document to check

    Returns:
        bool: True if document was failed due to timeout, False otherwise
    """
    if document.processing_status != DocumentProcessingStatusEnum.Failed:
        return False

    if not document.document_metadata:
        return False

    timeout_info = document.document_metadata.get("timeout_failure")
    if timeout_info and timeout_info.get("reason") == "processing_timeout":
        return True

    return False


def should_skip_processing_due_to_timeout(document_id: UUID) -> tuple[bool, str]:
    """
    Check if processing should be skipped for a document due to timeout failure.

    Args:
        db: Database session
        document_id: Document ID to check

    Returns:
        tuple: (should_skip, reason_message)
    """

    with SyncSessionLocal() as db:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            return False, "Document not found"

        if is_document_failed_due_to_timeout(document):
            timeout_info = document.document_metadata.get("timeout_failure", {})
            failed_at = timeout_info.get("marked_failed_at", "unknown time")
            hours_stuck = timeout_info.get("hours_stuck", 0)

            reason = (
                f"Document was marked as failed due to timeout at {failed_at} "
                f"(stuck for {hours_stuck:.1f} hours). Skipping further processing."
            )
            return True, reason

        return False, "Document can be processed"


def is_split_failed_due_to_timeout(split: FileSplit) -> bool:
    """
    Check if a split was marked as failed due to timeout.

    Args:
        split: FileSplit to check

    Returns:
        bool: True if split was failed due to timeout, False otherwise
    """
    if split.status != SplitFileStatusType.Failed:
        return False

    if not split.split_metadata:
        return False

    timeout_info = split.split_metadata.get("timeout_failure")
    if timeout_info and timeout_info.get("reason") in [
        "processing_timeout",
        "parent_document_timeout",
    ]:
        return True

    return False


def should_skip_split_processing_due_to_timeout(split_id: UUID) -> tuple[bool, str]:
    """
    Check if split processing should be skipped due to timeout failure.

    Args:
        split_id: Split ID to check

    Returns:
        tuple: (should_skip, reason_message)
    """
    from app.models.file_split_model import FileSplit

    with SyncSessionLocal() as db:
        split = db.query(FileSplit).filter(FileSplit.id == split_id).first()

        if not split:
            return False, "Split not found"

        if is_split_failed_due_to_timeout(split):
            timeout_info = split.split_metadata.get("timeout_failure", {})
            failed_at = timeout_info.get("marked_failed_at", "unknown time")
            reason_type = timeout_info.get("reason", "unknown")

            reason = (
                f"Split was marked as failed due to {reason_type} at {failed_at}. "
                f"Skipping further processing."
            )
            return True, reason

        return False, "Split can be processed"


def _get_user_id_from_processing_flag(dataset_id: str) -> Optional[UUID]:
    """
    Extract user_id from existing processing flag for a dataset.

    Args:
        dataset_id: Dataset ID to check

    Returns:
        UUID of the user who set the processing flag, or None if not found
    """
    try:
        from app.api.deps import get_redis_client_sync

        redis_client = get_redis_client_sync()

        # Find processing keys for this dataset (format: ingestion:processing:{dataset_id}:{user_id})
        pattern = f"ingestion:processing:{dataset_id}:*"
        keys = redis_client.keys(pattern)

        if keys:
            # Extract user_id from the first matching key
            key = keys[0].decode() if isinstance(keys[0], bytes) else str(keys[0])
            parts = key.split(":")
            if len(parts) >= 4:
                user_id_str = parts[3]
                try:
                    return UUID(user_id_str)
                except ValueError:
                    logger.warning(
                        f"Invalid user_id format in processing flag: {user_id_str}"
                    )
                    return None

        logger.debug(f"No processing flag found for dataset {dataset_id}")
        return None

    except Exception as e:
        logger.error(
            f"Error getting user_id from processing flag for dataset {dataset_id}: {str(e)}"
        )
        return None


def _get_user_ids_for_datasets_batch(dataset_ids: list) -> dict:
    """
    Batch lookup of user_ids for multiple datasets to optimize Redis calls.

    Args:
        dataset_ids: List of dataset IDs to look up

    Returns:
        Dictionary mapping dataset_id -> user_id (or None if not found)
    """
    try:
        from app.api.deps import get_redis_client_sync

        redis_client = get_redis_client_sync()
        result = {}

        # Use SCAN instead of KEYS for better performance at scale
        for dataset_id in dataset_ids:
            try:
                pattern = f"ingestion:processing:{dataset_id}:*"

                # Use SCAN with pattern (more efficient than KEYS for large datasets)
                cursor = 0
                keys = []
                while True:
                    cursor, partial_keys = redis_client.scan(
                        cursor, match=pattern, count=10
                    )
                    keys.extend(partial_keys)
                    if cursor == 0:
                        break

                if keys:
                    # Extract user_id from the first matching key
                    key = (
                        keys[0].decode() if isinstance(keys[0], bytes) else str(keys[0])
                    )
                    parts = key.split(":")
                    if len(parts) >= 4:
                        user_id_str = parts[3]
                        try:
                            result[dataset_id] = UUID(user_id_str)
                        except ValueError:
                            logger.warning(
                                f"Invalid user_id format in processing flag: {user_id_str}"
                            )
                            result[dataset_id] = None
                    else:
                        result[dataset_id] = None
                else:
                    result[dataset_id] = None

            except Exception as e:
                logger.error(
                    f"Error getting user_id for dataset {dataset_id}: {str(e)}"
                )
                result[dataset_id] = None

        return result

    except Exception as e:
        logger.error(f"Error in batch user_id lookup: {str(e)}")
        return dict.fromkeys(dataset_ids, None)


def _force_cleanup_if_dataset_complete(
    db: Session, dataset_id: str, user_id: UUID
) -> None:
    """
    Force cleanup of processing flag if all documents in dataset are in terminal states.

    Args:
        db: Database session
        dataset_id: Dataset ID to check
        user_id: User ID for cleanup
    """
    try:
        # Check if all documents are in terminal states
        from app.models.document_model import Document, DocumentProcessingStatusEnum

        documents = (
            db.query(Document)
            .filter(
                Document.dataset_id == UUID(dataset_id),
                Document.deleted_at.is_(None),
            )
            .all()
        )

        if not documents:
            logger.info(
                f"No documents found for dataset {dataset_id} - cleaning up processing flag"
            )
            cleanup_processing_flag(dataset_id, str(user_id))
            return

        # Check if all documents are in terminal states
        terminal_states = [
            DocumentProcessingStatusEnum.Success,
            DocumentProcessingStatusEnum.Failed,
            DocumentProcessingStatusEnum.Exception,
        ]

        all_terminal = all(
            doc.processing_status in terminal_states for doc in documents
        )

        if all_terminal:
            logger.info(
                f"All {len(documents)} documents in dataset {dataset_id} are in terminal states - "
                f"forcing processing flag cleanup"
            )
            cleanup_processing_flag(dataset_id, str(user_id))
        else:
            non_terminal_count = sum(
                1 for doc in documents if doc.processing_status not in terminal_states
            )
            logger.info(
                f"Dataset {dataset_id} still has {non_terminal_count} non-terminal documents - "
                f"keeping processing flag"
            )

    except Exception as e:
        logger.error(f"Error in force cleanup check for dataset {dataset_id}: {str(e)}")


def _send_document_timeout_notifications(
    failed_documents: list, user_id_map: dict = None
) -> None:
    """
    Send document-level notifications for documents that were marked as failed due to timeout.
    Optimized for large-scale operations with batching and efficient Redis lookups.

    Args:
        failed_documents: List of document info dictionaries that were marked as failed
                         Each dict contains: document_id, dataset_id, original_status,
                         hours_stuck, file_name
        user_id_map: Optional pre-fetched mapping of dataset_id -> user_id to avoid Redis lookups
                    (useful when processing flags might be cleaned up before this function runs)
    """
    try:
        if not failed_documents:
            return

        # Group documents by dataset to minimize Redis lookups
        dataset_documents = {}
        for doc_info in failed_documents:
            dataset_id = doc_info.get("dataset_id")
            if dataset_id:
                if dataset_id not in dataset_documents:
                    dataset_documents[dataset_id] = []
                dataset_documents[dataset_id].append(doc_info)

        if not dataset_documents:
            logger.warning("No valid dataset_ids found in failed documents")
            return

        # Use pre-fetched user_id_map or batch lookup user_ids for all datasets
        if user_id_map is None:
            dataset_ids = list(dataset_documents.keys())
            user_id_map = _get_user_ids_for_datasets_batch(dataset_ids)
            logger.debug("Performed fresh user_id lookup from Redis")
        else:
            logger.debug("Using pre-fetched user_id mapping")

        # Track notification stats
        total_notifications = 0
        failed_notifications = 0

        # Send notifications grouped by user to optimize WebSocket connections
        user_notifications = {}  # user_id -> list of notifications

        for dataset_id, documents in dataset_documents.items():
            user_id = user_id_map.get(dataset_id)
            if not user_id:
                logger.warning(
                    f"No user_id found for dataset {dataset_id} - skipping {len(documents)} document notifications"
                )
                failed_notifications += len(documents)
                continue

            if user_id not in user_notifications:
                user_notifications[user_id] = []

            # Prepare notifications for this user
            for doc_info in documents:
                document_id = doc_info.get("document_id")

                hours_stuck = doc_info.get("hours_stuck", 0)

                if document_id:
                    notification_data = {
                        "document_id": document_id,
                        "dataset_id": dataset_id,
                        "hours_stuck": hours_stuck,
                        "error_message": f"Document processing timed out after {hours_stuck:.1f} hours",
                        "details": {
                            "processing_status": "Failed",
                            "dataset_id": dataset_id,
                            "timeout_info": {
                                "hours_stuck": hours_stuck,
                                "reason": "processing_timeout",
                            },
                        },
                    }
                    user_notifications[user_id].append(notification_data)

        # Send batched notifications per user
        for user_id, notifications in user_notifications.items():
            try:
                # Option 1: Send individual notifications (current approach)
                for notification in notifications:
                    send_failure_notification(
                        user_id=user_id,
                        entity_id=UUID(notification["document_id"]),
                        entity_type="document",
                        error_message=notification["error_message"],
                        details=notification["details"],
                    )
                    total_notifications += 1

                logger.info(
                    f"📧 Sent {len(notifications)} timeout notifications to user {user_id}"
                )

                # Option 2: Send batch notification (uncomment if you want to batch)
                # _send_batch_timeout_notification(user_id, notifications)

            except Exception as e:
                logger.error(f"Error sending notifications to user {user_id}: {str(e)}")
                failed_notifications += len(notifications)

        logger.info(
            f"📊 Timeout notification summary: {total_notifications} sent, "
            f"{failed_notifications} failed, {len(dataset_documents)} datasets processed"
        )

    except Exception as e:
        logger.error(f"Error in document timeout notifications: {str(e)}")


def mark_stuck_documents_as_failed(db: Session) -> dict:
    """
    Find and mark documents that have been stuck in processing for too long as failed.

    Args:
        db: Database session

    Returns:
        Dictionary with results of the cleanup operation
    """
    timeout_seconds = settings.DOCUMENT_PROCESSING_TIMEOUT_SECONDS
    cutoff_time = datetime.now(UTC) - timedelta(seconds=timeout_seconds)

    logger.info(
        f"🔍 Checking for documents stuck in processing for more than {timeout_seconds} seconds (last updated before {cutoff_time})"
    )

    # Define intermediate states that should timeout
    intermediate_states = [
        DocumentProcessingStatusEnum.Processing,
        DocumentProcessingStatusEnum.Extracting,
        DocumentProcessingStatusEnum.ExtractionCompleted,
    ]

    # Find stuck documents
    stuck_documents = (
        db.query(Document)
        .filter(
            Document.processing_status.in_(intermediate_states),
            Document.updated_at
            < cutoff_time,  # Document last updated before cutoff (handles re-ingestion)
            Document.deleted_at.is_(None),  # Not deleted
        )
        .all()
    )

    if not stuck_documents:
        logger.info("✅ No stuck documents found")
        return {
            "success": True,
            "message": "No stuck documents found",
            "total_checked": 0,
            "marked_failed": 0,
            "documents": [],
        }

    logger.warning(f"⚠️ Found {len(stuck_documents)} stuck documents")

    marked_failed = []
    errors = []

    for document in stuck_documents:
        try:
            # Calculate how long it's been stuck (based on last update)
            time_stuck = datetime.now(UTC) - document.updated_at.replace(tzinfo=UTC)
            hours_stuck = time_stuck.total_seconds() / 3600

            logger.warning(
                f"📄 Marking stuck document as failed: {document.id} "
                f"(stuck for {hours_stuck:.1f} hours in status {document.processing_status})"
            )

            # Mark as failed
            old_status = document.processing_status
            document.processing_status = DocumentProcessingStatusEnum.Failed
            document.processed_at = datetime.now(UTC)
            document.error_message = f"Document marked as failed due to timeout after {hours_stuck:.1f} hours in {old_status} status"

            # Update metadata to track the timeout
            if not document.document_metadata:
                document.document_metadata = {}

            document.document_metadata.update(
                {
                    "timeout_failure": {
                        "original_status": old_status.value,
                        "hours_stuck": hours_stuck,
                        "timeout_threshold_seconds": timeout_seconds,
                        "marked_failed_at": datetime.now(UTC).isoformat(),
                        "reason": "processing_timeout",
                    }
                }
            )

            from sqlalchemy.orm import attributes

            attributes.flag_modified(document, "document_metadata")
            db.add(document)

            # Mark all non-successful splits as failed when document is failed due to timeout
            from app.models.file_split_model import FileSplit, SplitFileStatusType

            non_successful_splits = (
                db.query(FileSplit)
                .filter(
                    FileSplit.document_id == document.id,
                    FileSplit.status != SplitFileStatusType.Success,
                )
                .all()
            )

            if non_successful_splits:
                logger.info(
                    f"📄 Marking {len(non_successful_splits)} non-successful splits as failed for document {document.id}"
                )

                for split in non_successful_splits:
                    # Store original status before changing it
                    original_status = split.status.value

                    split.status = SplitFileStatusType.Failed

                    # Update split metadata to track timeout failure
                    if not split.split_metadata:
                        split.split_metadata = {}

                    split.split_metadata.update(
                        {
                            "timeout_failure": {
                                "original_status": original_status,
                                "parent_document_timeout": True,
                                "marked_failed_at": datetime.now(UTC).isoformat(),
                                "reason": "parent_document_timeout",
                            }
                        }
                    )

                    attributes.flag_modified(split, "split_metadata")
                    db.add(split)

            marked_failed.append(
                {
                    "document_id": str(document.id),
                    "dataset_id": str(document.dataset_id),
                    "original_status": old_status.value,
                    "hours_stuck": round(hours_stuck, 2),
                    "file_name": getattr(document, "file_name", "unknown"),
                }
            )

        except Exception as e:
            error_msg = f"Error marking document {document.id} as failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append({"document_id": str(document.id), "error": error_msg})

    try:
        db.commit()
        db.refresh(document)
        logger.info(f"✅ Successfully marked {len(marked_failed)} documents as failed")

        # Get user_ids BEFORE any notifications (to avoid processing flag cleanup issues)
        dataset_ids = list(
            {doc["dataset_id"] for doc in marked_failed if doc.get("dataset_id")}
        )
        user_id_map = (
            _get_user_ids_for_datasets_batch(dataset_ids) if dataset_ids else {}
        )

        # Send document-level notifications FIRST (for each individual failed document)
        _send_document_timeout_notifications(marked_failed, user_id_map)

        # Then trigger dataset status propagation and notifications (when all docs are terminal)
        _propagate_dataset_updates_for_failed_documents(db, marked_failed)

    except Exception as e:
        db.rollback()
        logger.error(
            f"❌ Failed to commit stuck document updates: {str(e)}", exc_info=True
        )
        return {
            "success": False,
            "message": f"Failed to commit updates: {str(e)}",
            "total_checked": len(stuck_documents),
            "marked_failed": 0,
            "errors": [str(e)],
        }

    return {
        "success": True,
        "message": f"Marked {len(marked_failed)} stuck documents as failed",
        "total_checked": len(stuck_documents),
        "marked_failed": len(marked_failed),
        "documents": marked_failed,
        "errors": errors,
    }


def _propagate_dataset_updates_for_failed_documents(
    db: Session, failed_documents: list
) -> None:
    """
    Trigger dataset status updates for documents that were marked as failed.

    Args:
        db: Database session
        failed_documents: List of document info that were marked as failed
    """
    # Get unique dataset IDs
    dataset_ids = set()
    for doc_info in failed_documents:
        if doc_info.get("dataset_id"):
            dataset_ids.add(doc_info["dataset_id"])

    logger.info(
        f"📊 Propagating status updates for {len(dataset_ids)} affected datasets"
    )

    for dataset_id in dataset_ids:
        try:
            # Try to get user_id from existing processing flags to ensure cleanup
            user_id = _get_user_id_from_processing_flag(dataset_id)

            # Trigger dataset status update with user_id for proper cleanup
            updated = propagate_ingestion_status(
                db, "dataset", UUID(dataset_id), user_id
            )
            if updated:
                logger.info(
                    f"📈 Updated dataset {dataset_id} status after timeout failures"
                )
            elif user_id:
                # Even if status wasn't updated, force cleanup if all documents are in terminal state
                _force_cleanup_if_dataset_complete(db, dataset_id, user_id)
        except Exception as e:
            logger.error(
                "❌ Failed to update dataset %s status: %s",
                dataset_id,
                str(e),
                exc_info=True,
            )


def get_stuck_documents_report(db: Session) -> dict:
    """
    Generate a report of potentially stuck documents without marking them as failed.

    Args:
        db: Database session

    Returns:
        Dictionary with stuck documents report
    """
    timeout_seconds = settings.DOCUMENT_PROCESSING_TIMEOUT_SECONDS
    cutoff_time = datetime.now(UTC) - timedelta(seconds=timeout_seconds)
    warning_cutoff = datetime.now(UTC) - timedelta(
        seconds=timeout_seconds * 0.75
    )  # 75% of timeout

    # Define intermediate states
    intermediate_states = [
        DocumentProcessingStatusEnum.Processing,
        DocumentProcessingStatusEnum.Extracting,
        DocumentProcessingStatusEnum.ExtractionCompleted,
    ]

    # Find documents approaching timeout (warning zone)
    warning_documents = (
        db.query(Document)
        .filter(
            Document.processing_status.in_(intermediate_states),
            Document.updated_at < warning_cutoff,
            Document.updated_at >= cutoff_time,  # Not yet timed out
            Document.deleted_at.is_(None),
        )
        .all()
    )

    # Find already timed out documents
    stuck_documents = (
        db.query(Document)
        .filter(
            Document.processing_status.in_(intermediate_states),
            Document.updated_at < cutoff_time,
            Document.deleted_at.is_(None),
        )
        .all()
    )

    def _document_info(doc):
        time_stuck = datetime.now(UTC) - doc.updated_at.replace(tzinfo=UTC)
        return {
            "document_id": str(doc.id),
            "dataset_id": str(doc.dataset_id),
            "status": doc.processing_status.value,
            "created_at": doc.created_at.isoformat(),
            "updated_at": doc.updated_at.isoformat(),
            "hours_stuck": round(time_stuck.total_seconds() / 3600, 2),
            "file_name": getattr(doc, "file_name", "unknown"),
        }

    return {
        "timeout_threshold_seconds": timeout_seconds,
        "current_time": datetime.now(UTC).isoformat(),
        "cutoff_time": cutoff_time.isoformat(),
        "warning_cutoff": warning_cutoff.isoformat(),
        "stuck_documents": {
            "count": len(stuck_documents),
            "documents": [_document_info(doc) for doc in stuck_documents],
        },
        "warning_documents": {
            "count": len(warning_documents),
            "documents": [_document_info(doc) for doc in warning_documents],
        },
    }
