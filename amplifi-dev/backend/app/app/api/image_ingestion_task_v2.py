"""
Celery task for image document ingestion using Document model (v2).
Uses the Document model directly instead of FileIngestion model.
Enhanced with memory optimization for processing large images.

COMPLETION CHECKING:
This module now uses the robust completion checker (app.utils.robust_completion_checker)
for all completion checking logic. The old completion functions have been removed
to maintain a clean and robust architecture.

The completion checking is handled via:
- check_split_completion_with_fallback() for split-based image processing
- check_document_completion_with_fallback() for direct image processing

These functions provide Redis-based distributed locking with PostgreSQL fallback
to prevent race conditions and ensure consistent completion state checking.
"""

import gc
import json
import os
import re
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import attributes

from app.api.deps import get_redis_client_sync
from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.utils.image_processing_utils import (
    extract_image_features,
    extract_ocr_text_with_chunks,
    optimize_image_for_api,
    process_image_in_tiles,
)
from app.utils.ingestion_benchmark import IngestionBenchmark
from app.utils.ingestion_status_propagation import (
    propagate_ingestion_status,
    should_skip_processing_due_to_timeout,
)
from app.utils.openai_utils import (
    generate_embedding_with_retry,
    get_openai_client,
    retry_openai_call,
)
from app.utils.processing_lock_utils import (
    _cleanup_processing_lock,
    _handle_stale_lock,
)


def _generate_embedding_with_retry(text: str) -> List[float]:
    """Generate embedding with retry logic for rate limit errors"""
    return generate_embedding_with_retry(text)


def _validate_file_path(file_path: str) -> bool:
    """
    Validate that a file path exists and is within allowed directories.

    Args:
        file_path: Path to validate

    Returns:
        True if valid, False if not
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    return True


def memory_efficient_image_loading(
    file_path: str, max_dimension: int = 1024
) -> Tuple[Image.Image, int, int]:
    """
    Load an image with memory optimization.

    Args:
        file_path: Path to the image file
        max_dimension: Maximum dimension for thumbnailing

    Returns:
        Tuple of (PIL Image object, original width, original height)
    """
    try:
        # Validate file exists first
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Use context manager for proper cleanup
        with Image.open(file_path) as img:
            original_width, original_height = img.size

            # Create a copy to avoid reference issues
            if max(img.size) > max_dimension:
                # Create thumbnail for more memory-efficient processing
                img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
                processed_img = img.copy()
            else:
                processed_img = img.copy()

        return processed_img, original_width, original_height

    except Exception as e:
        logger.error(f"Error loading image {file_path}: {str(e)}")
        raise


def _update_document_status(
    db,
    document: Document,
    status: DocumentProcessingStatusEnum,
    error_message: str = None,
) -> None:
    """Centralized function for document status updates"""

    # Get the document from the database to get the latest state
    document = db.query(Document).filter(Document.id == document.id).first()

    if document.processing_status in [
        DocumentProcessingStatusEnum.Success,
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        logger.info(
            f"Document {document.id} is in {document.processing_status} state - not updating status or timestamp"
        )
        return

    document.processing_status = status
    if error_message:
        document.error_message = error_message
    if status in [
        DocumentProcessingStatusEnum.Success,
        DocumentProcessingStatusEnum.Failed,
        DocumentProcessingStatusEnum.Exception,
    ]:
        document.processed_at = datetime.now(UTC)
    attributes.flag_modified(document, "processing_status")
    db.commit()


def _initialize_document_record(
    db,
    file_id: str,
    file_path: str,
    dataset_id: str,
    ingestion_id: str,
    parent_document_id: Optional[str],
    metadata: Dict[str, Any],
    skip_successful_files: bool,
) -> Tuple[Optional[Document], bool]:
    """Initialize document record and handle preliminary checks"""
    # Determine if this is a child image (part of PDF)
    is_child_image = parent_document_id is not None

    document = None

    if not is_child_image:
        # Check if the file is already processed
        existing_doc = (
            db.query(Document)
            .filter(
                Document.file_id == uuid.UUID(file_id),
                Document.dataset_id == uuid.UUID(dataset_id),
                Document.deleted_at.is_(None),
            )
            .first()
        )
        record_id = existing_doc.id if existing_doc else None

        if not skip_successful_files:
            _delete_existing_document_chunks(record_id)

        if (
            existing_doc
            and skip_successful_files
            and existing_doc.processing_status == DocumentProcessingStatusEnum.Success
        ):
            logger.info(f"File {file_path} already processed successfully. Skipping.")
            return existing_doc, is_child_image

        # This is a standalone image, create document record as usual
        document = _get_or_create_document(
            db=db,
            file_id=file_id,
            dataset_id=dataset_id,
            file_path=file_path,
            task_id=None,  # Will be updated with task.request.id
            ingestion_id=ingestion_id,
            metadata=metadata,
        )
        # Update document status to Processing
        _update_document_status(db, document, DocumentProcessingStatusEnum.Processing)

        # Update document metadata
        image_metadata = _extract_image_metadata(file_path)
        merged_metadata = {
            **(document.document_metadata or {}),
            **image_metadata,
        }
        document.document_metadata = merged_metadata
        db.commit()
        db.refresh(document)

        # Update to extraction status
        _update_document_status(db, document, DocumentProcessingStatusEnum.Extracting)
    else:
        # This is a child image (part of PDF), log this info
        logger.info(f"Processing child image from parent document {parent_document_id}")

        # Get parent document for reference
        parent_doc = (
            db.query(Document)
            .filter(Document.id == uuid.UUID(parent_document_id))
            .first()
        )
        if not parent_doc:
            raise ValueError(f"Parent document {parent_document_id} not found")

    return document, is_child_image


def _process_image_content(
    db,
    document: Optional[Document],
    is_child_image: bool,
    file_path: str,
    parent_document_id: Optional[str],
) -> Dict[str, Any]:
    """Process image content and extract information"""
    # Use tiled processing for OCR to improve memory efficiency
    if os.path.getsize(file_path) > 5 * 1024 * 1024:  # 5MB threshold
        ocr_text, ocr_chunks = process_image_in_tiles(file_path)
        logger.info(f"Used tiled processing for large image: {file_path}")
    else:
        ocr_text, ocr_chunks = extract_ocr_text_with_chunks(file_path)

    description, detected_objects = _analyze_image(
        file_path, Path(file_path).name, ocr_text
    )
    description_embedding = _generate_embedding_with_retry(description)

    current_status = DocumentProcessingStatusEnum.ExtractionCompleted

    if is_child_image:
        # Do NOT overwrite parent document's description and embedding
        # The parent document should keep its PDF summary
        parent_doc = (
            db.query(Document)
            .filter(Document.id == uuid.UUID(parent_document_id))
            .first()
        )

        # Only update processing status, NOT description/embedding
        parent_doc.processing_status = current_status

        # Log for debugging
        logger.info(
            f"Child image processed, keeping parent PDF summary intact for document {parent_document_id}"
        )
    else:
        # For standalone images, update normally
        document.description = description
        document.description_embedding = description_embedding
        document.processing_status = current_status

    db.commit()

    # Free memory after processing
    gc.collect()

    return {
        "ocr_text": ocr_text,
        "ocr_chunks": ocr_chunks,
        "description": description,
        "description_embedding": description_embedding,
        "detected_objects": detected_objects,
        "current_status": current_status,
    }


def _build_chunk_metadata_for_child_image(
    base_metadata: Dict[str, Any],
    file_path: str,
    parent_document_id: str,
    split_id: Optional[str],
) -> Dict[str, Any]:
    """Build metadata for child image chunks with common properties."""
    metadata = base_metadata.copy()
    metadata.update(
        {
            "status": DocumentProcessingStatusEnum.Success.value,
            "processed_at": datetime.now(UTC).isoformat(),
            "file_path": file_path,
            "parent_document_id": parent_document_id,
        }
    )

    if split_id:
        metadata["split_id"] = split_id

    return metadata


def _get_document_id_for_chunk(
    is_child_image: bool, parent_document_id: str, document: Document
) -> uuid.UUID:
    """Get the appropriate document ID for chunk creation."""
    return uuid.UUID(parent_document_id) if is_child_image else document.id


def _create_description_chunk(
    db,
    description: str,
    description_embedding: List[float],
    file_path: str,
    processing_metadata: Dict[str, Any],
    is_child_image: bool,
    parent_document_id: Optional[str],
    document: Optional[Document],
    split_id: Optional[str],
) -> Optional[DocumentChunk]:
    """Create and save the description chunk."""
    # Skip creating chunk if description is an error message
    if description.startswith("Error analyzing image:"):
        logger.warning(f"Skipping chunk creation due to error: {description}")
        return None

    base_metadata = {
        "chunk_order": 0,
        "chunked_by_engine": "gpt-4o",
        "file_path": file_path,
    }

    if is_child_image:
        chunk_metadata = _build_chunk_metadata_for_child_image(
            {**base_metadata, **processing_metadata},
            file_path,
            parent_document_id,
            split_id,
        )
    else:
        chunk_metadata = base_metadata

    desc_chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=_get_document_id_for_chunk(
            is_child_image, parent_document_id, document
        ),
        chunk_type=ChunkTypeEnum.ImageDescription,
        chunk_text=description,
        chunk_embedding=description_embedding,
        chunk_metadata=chunk_metadata,
        split_id=split_id,
    )
    attributes.flag_modified(desc_chunk, "chunk_metadata")
    db.add(desc_chunk)
    db.commit()
    return desc_chunk


def _create_single_ocr_chunk(
    db,
    ocr_chunks: List[Dict[str, Any]],
    is_child_image: bool,
    parent_document_id: Optional[str],
    document: Optional[Document],
    file_path: str,
    split_id: Optional[str],
) -> Optional[uuid.UUID]:
    """Create a single OCR chunk with all text combined."""
    if not ocr_chunks:
        return None

    # Combine all OCR text into a single string
    combined_text = " ".join(
        chunk_info["text"]
        for chunk_info in ocr_chunks
        if chunk_info.get("text", "").strip()
    )

    if not combined_text.strip():
        return None

    # Skip creating chunk if OCR text contains error messages
    if (
        combined_text.startswith("Error analyzing image:")
        or "Error code:" in combined_text
    ):
        logger.warning(
            f"Skipping OCR chunk creation due to error in text: {combined_text[:100]}..."
        )
        return None

    # Use single values for confidence and coordinates
    base_metadata = {
        "confidence": 1,  # Single confidence value
        "coordinates": [0, 0, 0, 0],  # Single coordinates value
        "chunk_order": 0,
        "chunked_by_engine": "pytesseract",
        "original_chunk_count": len(
            ocr_chunks
        ),  # Keep track of how many original chunks were combined
    }

    if is_child_image:
        chunk_metadata = _build_chunk_metadata_for_child_image(
            base_metadata, file_path, parent_document_id, split_id
        )
    else:
        chunk_metadata = base_metadata

    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=_get_document_id_for_chunk(
            is_child_image, parent_document_id, document
        ),
        chunk_type=ChunkTypeEnum.ImageText,
        chunk_text=combined_text,
        chunk_embedding=_generate_embedding_with_retry(combined_text),
        chunk_metadata=chunk_metadata,
        split_id=split_id,
    )
    attributes.flag_modified(chunk, "chunk_metadata")
    db.add(chunk)
    db.commit()
    return chunk.id


def _create_object_detection_chunk(
    db,
    detected_objects: List[Dict[str, Any]],
    is_child_image: bool,
    parent_document_id: Optional[str],
    document: Optional[Document],
    file_path: str,
    split_id: Optional[str],
) -> Optional[uuid.UUID]:
    """Create object detection chunk if objects were detected."""
    if not detected_objects:
        return None

    # Join all object names with semicolons
    object_names = "; ".join(
        obj.get("object_name") if isinstance(obj, dict) else str(obj)
        for obj in detected_objects
    )

    # Skip creating chunk if object names contain error messages
    if (
        object_names.startswith("Error analyzing image:")
        or "Error code:" in object_names
    ):
        logger.warning(
            f"Skipping object detection chunk creation due to error in object names: {object_names[:100]}..."
        )
        return None

    base_metadata = {
        "chunk_order": 0,
        "chunked_by_engine": "gpt-4o",
    }

    if is_child_image:
        chunk_metadata = _build_chunk_metadata_for_child_image(
            base_metadata, file_path, parent_document_id, split_id
        )
    else:
        chunk_metadata = base_metadata

    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=_get_document_id_for_chunk(
            is_child_image, parent_document_id, document
        ),
        chunk_type=ChunkTypeEnum.ImageObject,
        chunk_text=object_names,
        chunk_embedding=_generate_embedding_with_retry(object_names),
        chunk_metadata=chunk_metadata,
        split_id=split_id,
    )
    attributes.flag_modified(chunk, "chunk_metadata")
    db.add(chunk)
    db.commit()
    return chunk.id


def _update_final_statuses(
    db,
    is_child_image: bool,
    document: Optional[Document],
    desc_chunk: Optional[DocumentChunk],
    user_id: Optional[uuid.UUID],
    parent_document_id: Optional[str],
) -> None:
    """Update final processing statuses based on image type."""
    if not is_child_image:
        # Update standalone document status
        _update_document_status(db, document, DocumentProcessingStatusEnum.Success)

        # Propagate success status up to dataset level
        if user_id:
            propagate_ingestion_status(db, "document", document.id, user_id)
    else:
        # Update status in description chunk metadata for child images (if chunk was created)
        if desc_chunk:
            desc_chunk.chunk_metadata["status"] = (
                DocumentProcessingStatusEnum.Success.value
            )
            desc_chunk.chunk_metadata["processed_at"] = datetime.now(UTC).isoformat()
            attributes.flag_modified(desc_chunk, "chunk_metadata")
            db.commit()
        else:
            logger.warning(
                "No description chunk created for child image due to error in processing"
            )


def _handle_completion_checks(
    metadata: Optional[Dict[str, Any]],
    user_id: Optional[uuid.UUID],
) -> None:
    """Handle split and document completion checks with robust fallback mechanism."""
    if not metadata:
        return

    # Use the new robust completion checker for split images
    if metadata.get("parent_document_id") and metadata.get("split_id"):
        from app.utils.robust_completion_checker import (
            check_split_completion_with_fallback,
        )

        check_split_completion_with_fallback(
            parent_document_id=metadata["parent_document_id"],
            split_id=metadata["split_id"],
            user_id=user_id,
        )
    # Check document completion for direct image ingestion using robust checker
    elif metadata.get("parent_document_id"):
        # Use the new robust completion check with Redis/PostgreSQL fallback
        from app.utils.robust_completion_checker import (
            check_document_completion_with_fallback,
        )

        check_document_completion_with_fallback(
            parent_document_id=metadata["parent_document_id"], user_id=user_id
        )


def _store_chunks_and_update_status(
    db,
    document: Optional[Document],
    is_child_image: bool,
    processing_result: Dict[str, Any],
    user_id: Optional[uuid.UUID],
    parent_document_id: Optional[str],
    file_path: str,
    file_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store chunks and update document status"""
    # Extract processing results
    ocr_chunks = processing_result["ocr_chunks"]
    description = processing_result["description"]
    description_embedding = processing_result["description_embedding"]
    detected_objects = processing_result["detected_objects"]
    current_status = processing_result["current_status"]

    # Prepare common metadata
    processing_metadata = {
        "status": current_status.value,
        "started_at": datetime.now(UTC).isoformat(),
        "file_id": file_id,
    }

    # Extract split_id for reuse
    split_id = metadata.get("split_id") if is_child_image and metadata else None
    chunk_ids = []

    # Create description chunk
    desc_chunk = _create_description_chunk(
        db,
        description,
        description_embedding,
        file_path,
        processing_metadata,
        is_child_image,
        parent_document_id,
        document,
        split_id,
    )
    if desc_chunk:
        chunk_ids.append(desc_chunk.id)

    # Create OCR chunk
    ocr_chunk_id = _create_single_ocr_chunk(
        db,
        ocr_chunks,
        is_child_image,
        parent_document_id,
        document,
        file_path,
        split_id,
    )
    if ocr_chunk_id:
        chunk_ids.append(ocr_chunk_id)

    # Create object detection chunk if needed
    object_chunk_id = _create_object_detection_chunk(
        db,
        detected_objects,
        is_child_image,
        parent_document_id,
        document,
        file_path,
        split_id,
    )
    if object_chunk_id:
        chunk_ids.append(object_chunk_id)

    # Update final statuses
    _update_final_statuses(
        db, is_child_image, document, desc_chunk, user_id, parent_document_id
    )

    # Handle completion checks
    _handle_completion_checks(metadata, user_id)

    return {
        "success": True,
        "document_id": str(document.id) if document else None,
        "file_id": file_id,
        "dataset_id": document.dataset_id if document else None,
        "chunk_count": len(chunk_ids),
        "chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
        "parent_document_id": parent_document_id,
        "is_child_image": is_child_image,
    }


def _handle_image_processing_error(
    db,
    e: Exception,
    is_child_image: bool,
    document: Optional[Document],
    file_path: str,
    file_id: str,
    parent_document_id: Optional[str],
    user_id: Optional[uuid.UUID],
) -> Dict[str, Any]:
    error_msg = str(e)

    try:
        if not is_child_image and document:
            _update_document_status(
                db, document, DocumentProcessingStatusEnum.Failed, error_msg
            )

            # Propagate failed status up to dataset level
            if user_id:
                propagate_ingestion_status(db, "document", document.id, user_id)

        elif is_child_image and parent_document_id:
            # For child image, update the description chunk with error info
            try:
                parent_uuid = uuid.UUID(parent_document_id)

                from sqlalchemy import text

                desc_chunk = (
                    db.query(DocumentChunk)
                    .filter(
                        DocumentChunk.document_id == parent_uuid,
                        DocumentChunk.chunk_type == ChunkTypeEnum.ImageDescription,
                        text("chunk_metadata->>'file_path' = :file_path"),
                    )
                    .params(file_path=file_path)
                    .first()
                )

                if desc_chunk:
                    # Update the status in the chunk metadata
                    desc_chunk.chunk_metadata["status"] = (
                        DocumentProcessingStatusEnum.Failed.value
                    )
                    desc_chunk.chunk_metadata["error"] = error_msg
                    db.commit()

                    # Update parent document status based on all images
                    from app.utils.robust_completion_checker import (
                        check_document_completion_with_fallback,
                    )

                    check_document_completion_with_fallback(
                        parent_document_id=parent_document_id, user_id=user_id
                    )
                else:
                    logger.error(
                        f"Could not find description chunk for failed image: {file_path}"
                    )

            except Exception as chunk_err:
                logger.error(
                    f"Error updating chunk status for failed image: {str(chunk_err)}"
                )

    except Exception as update_err:
        logger.error(f"Error in error handling: {str(update_err)}")

    return {
        "success": False,
        "error": error_msg,
        "file_id": file_id,
        "file_path": file_path,
    }


@celery.task(name="tasks.image_ingestion_task_v2", bind=True, acks_late=True)
def image_ingestion_task_v2(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: uuid.UUID,
    metadata: Dict[str, Any] = None,
    skip_successful_files: bool = True,
    parent_document_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process an image file and extract text using OCR, analyze content using AI.

    Args:
        file_id: ID of the file being processed
        file_path: Path to the image file
        ingestion_id: ID of the current ingestion batch
        dataset_id: ID of the dataset
        user_id: ID of the user who initiated the ingestion (for WebSocket notifications)
        metadata: Additional metadata to store with the document
        parent_document_id: Optional ID of parent document (for images extracted from PDFs)

    Returns:
        Dictionary with processing results"""

    # Robust image-level deduplication with deadlock prevention
    processing_key = None
    redis_client = None

    try:
        redis_client = get_redis_client_sync()

        # Create unique processing key for image
        processing_key = (
            f"processing:image:{file_id}:ingestion:{ingestion_id}:dataset:{dataset_id}"
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
                    f"Image {file_id} is being processed by active task {existing_task_id} - skipping"
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
                }
            # If we reach here, either we took over a stale lock or the original task is dead
            logger.info(
                f"Acquired image processing lock after stale lock handling: {processing_key}"
            )

        logger.info(f"Successfully acquired image processing lock: {processing_key}")

    except Exception as e:
        # Don't fail the task if Redis is unavailable, just log it
        logger.warning(f"Redis deduplication not available: {str(e)}")
        processing_key = None
        redis_client = None

    try:
        benchmark = IngestionBenchmark(
            file_id=file_id,
            ingestion_type="image",
            parent_document_id=parent_document_id,
        )
        benchmark.start("total_image_processing")
        logger.info(f"Processing image document: {file_path}")
        logger.info(f"Metadata: {metadata}")

        # Initialize variables used in exception handling
        document = None
        is_child_image = False

        with SyncSessionLocal() as db:

            benchmark.start("document_initialization")
            # 1. Initialize document record (this determines standalone vs child)
            document, is_child_image = _initialize_document_record(
                db,
                file_id,
                file_path,
                dataset_id,
                ingestion_id,
                parent_document_id,
                metadata,
                skip_successful_files,
            )
            benchmark.end("document_initialization")

            # If initialize returns document with Success status, return early
            if (
                document
                and document.processing_status == DocumentProcessingStatusEnum.Success
            ):
                return {
                    "success": True,
                    "file_id": file_id,
                    "dataset_id": dataset_id,
                    "chunk_count": 0,
                    "document_id": str(document.id),
                }

            # Update task_id in document if it's a standalone image
            if not is_child_image and document:
                document.task_id = self.request.id
                document.updated_at = datetime.now(UTC)
                db.commit()

            # 2. now validate file path (after document record exists)
            benchmark.start("file_validation")
            if not _validate_file_path(file_path):
                # For child images, create failed chunk
                if is_child_image:
                    failed_chunk = DocumentChunk(
                        id=uuid.uuid4(),
                        document_id=uuid.UUID(parent_document_id),
                        chunk_type=ChunkTypeEnum.ImageDescription,
                        chunk_text="Failed to process image: Invalid file path",
                        chunk_embedding=[],
                        chunk_metadata={
                            "status": DocumentProcessingStatusEnum.Failed.value,
                            "error": f"Invalid file path: {file_path}",
                            "file_path": file_path,
                            "parent_document_id": parent_document_id,
                            "file_id": file_id,
                            "failed_at": datetime.now(UTC).isoformat(),
                            **(
                                {"split_id": metadata.get("split_id")}
                                if metadata and metadata.get("split_id")
                                else {}
                            ),
                        },
                        split_id=metadata.get("split_id") if metadata else None,
                    )
                    db.add(failed_chunk)
                    db.commit()

                    # Trigger completion checks for the parent using robust checker
                    if metadata and metadata.get("split_id"):
                        from app.utils.robust_completion_checker import (
                            check_split_completion_with_fallback,
                        )

                        check_split_completion_with_fallback(
                            parent_document_id=parent_document_id,
                            split_id=metadata["split_id"],
                            user_id=user_id,
                        )
                    else:
                        from app.utils.robust_completion_checker import (
                            check_document_completion_with_fallback,
                        )

                        check_document_completion_with_fallback(
                            parent_document_id=parent_document_id, user_id=user_id
                        )

                # For standalone images, raise exception to be caught by error handler
                raise ValueError(f"Invalid file path: {file_path}")
            benchmark.end("file_validation")

            # CRITICAL: Check timeout status BEFORE any processing starts
            if parent_document_id:
                # For child images, check if parent document was failed due to timeout
                logger.info(
                    f"🔍 CHECKING timeout status for parent document {parent_document_id}"
                )
                should_skip, skip_reason = should_skip_processing_due_to_timeout(
                    UUID(parent_document_id)
                )
                logger.info(
                    f"🎯 Timeout check result: should_skip={should_skip}, reason='{skip_reason}'"
                )

                if should_skip:
                    logger.warning(
                        f"⏰ SKIPPING image processing for {file_id} - parent document {parent_document_id} was failed due to timeout: {skip_reason}"
                    )
                    return {
                        "file_id": file_id,
                        "success": True,
                        "skipped": True,
                        "reason": "parent_document_timeout_failed",
                        "message": skip_reason,
                        "parent_document_id": parent_document_id,
                    }
            else:
                # For standalone images, check if this file already has a failed document
                existing_doc = (
                    db.query(Document)
                    .filter(
                        Document.file_id == UUID(file_id),
                        Document.dataset_id == UUID(dataset_id),
                    )
                    .first()
                )
                if existing_doc:
                    logger.info(
                        f"CHECKING if document {existing_doc.id} is failed due to timeout"
                    )
                    should_skip, skip_reason = should_skip_processing_due_to_timeout(
                        existing_doc.id
                    )
                    if should_skip:
                        logger.warning(
                            f"⏰ SKIPPING standalone image processing for {file_id} - document {existing_doc.id} was failed due to timeout: {skip_reason}"
                        )
                        return {
                            "file_id": file_id,
                            "success": True,
                            "skipped": True,
                            "reason": "document_timeout_failed",
                            "message": skip_reason,
                            "document_id": str(existing_doc.id),
                        }

            # 3. Process image content
            benchmark.start("image_analysis")
            processing_result = _process_image_content(
                db, document, is_child_image, file_path, parent_document_id
            )
            benchmark.end("image_analysis")

            # 4. Store chunks and update statuses
            benchmark.start("chunk_storage")
            result = _store_chunks_and_update_status(
                db,
                document,
                is_child_image,
                processing_result,
                user_id,
                parent_document_id,
                file_path,
                file_id,
                metadata,
            )
            benchmark.end("chunk_storage")
            benchmark.end("total_image_processing")
            return result
    except Exception as e:
        benchmark.end("total_image_processing")
        return _handle_image_processing_error(
            db,
            e,
            is_child_image,
            document,
            file_path,
            file_id,
            parent_document_id,
            user_id,
        )

    finally:
        # Clean up processing deduplication flag (only if we own it)
        if processing_key and redis_client:
            try:
                _cleanup_processing_lock(redis_client, processing_key, self.request.id)
            except Exception as e:
                logger.warning(f"Failed to clean up image processing lock: {str(e)}")

        # Close Redis connection
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")


def _delete_existing_document_chunks(record_id: UUID) -> None:
    """Delete existing document chunks if they exist"""
    if not record_id:
        logger.warning("No record ID provided for deletion")
        return
    try:
        with SyncSessionLocal() as db:
            existing_doc = db.query(Document).filter(Document.id == record_id).first()
            if existing_doc:
                # Delete chunks in batches to avoid memory issues
                batch_size = 100
                offset = 0

                while True:
                    # Get batch of chunk IDs to delete
                    chunk_ids = (
                        db.query(DocumentChunk.id)
                        .filter(DocumentChunk.document_id == record_id)
                        .limit(batch_size)
                        .offset(offset)
                        .all()
                    )

                    if not chunk_ids:
                        break

                    # Delete this batch
                    db.query(DocumentChunk).filter(
                        DocumentChunk.id.in_([c.id for c in chunk_ids])
                    ).delete(synchronize_session=False)

                    # Commit and update offset
                    db.commit()
                    offset += len(chunk_ids)

                    # Force garbage collection after each batch
                    gc.collect()

                logger.info(f"Deleted existing document chunks with ID {record_id}")
    except Exception as e:
        logger.error(
            f"Error deleting existing document chunks: {str(e)}", exc_info=True
        )


def _get_or_create_document(
    db,
    file_id: str,
    dataset_id: str,
    file_path: str,
    task_id: str,
    ingestion_id: str,
    metadata: Dict[str, Any] = None,
    parent_document_id: Optional[str] = None,
) -> Document:
    """Get existing document or create a new one"""
    # Check if document already exists
    existing_doc = (
        db.query(Document)
        .filter(
            Document.file_id == uuid.UUID(file_id),
            Document.dataset_id == uuid.UUID(dataset_id),
            Document.ingestion_id
            == (uuid.UUID(ingestion_id) if ingestion_id else None),
            Document.deleted_at.is_(None),
        )
        .first()
    )

    if existing_doc:
        # Update existing document
        existing_doc.task_id = task_id
        existing_doc.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(existing_doc)
        logger.info(f"Using existing document record with ID {existing_doc.id}")
        return existing_doc

    # Create new document
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
    mime_type = _get_mime_type(file_path)

    document = Document(
        file_id=uuid.UUID(file_id),
        dataset_id=uuid.UUID(dataset_id),
        document_type=DocumentTypeEnum.Image,
        processing_status=DocumentProcessingStatusEnum.Processing,
        file_path=file_path,
        file_size=file_size,
        mime_type=mime_type,
        document_metadata=metadata or {},
    )

    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(f"Created new document record with ID {document.id}")
    return document


def _get_mime_type(file_path: str) -> str:
    """Determine MIME type from file extension"""
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".jpg" or extension == ".jpeg":
        return "image/jpeg"
    elif extension == ".png":
        return "image/png"
    else:
        return "application/octet-stream"


def _extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract detailed metadata and features from image file using PIL and OpenCV"""
    try:
        # Basic metadata using memory-efficient loading
        with Image.open(image_path) as img:
            width, height = img.size
            format = img.format
            mode = img.mode

            # Get additional metadata if available
            info = {}
            for key, value in img.info.items():
                if isinstance(value, (str, int, float, bool)):
                    info[key] = value

        # Enhanced feature extraction with OpenCV - memory optimized
        cv_img = cv2.imread(image_path)
        if cv_img is not None:
            # Calculate average color using sampling for large images
            if cv_img.shape[0] * cv_img.shape[1] > 1000000:  # 1M pixels threshold
                # Downsample for large images to save memory
                resize_factor = 0.1  # Use 10% of pixels
                small_img = cv2.resize(
                    cv_img, (0, 0), fx=resize_factor, fy=resize_factor
                )
                avg_color = np.average(np.average(small_img, axis=0), axis=0).tolist()
                # Free memory
                del small_img
            else:
                avg_color = np.average(np.average(cv_img, axis=0), axis=0).tolist()

            # Detect edges and lines with memory optimization
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Free up memory by releasing the large arrays we don't need anymore
            del cv_img

            # Detect lines
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
            )
            line_count = 0 if lines is None else len(lines)

            # Free memory
            del edges
            del gray
            if lines is not None:
                del lines

            gc.collect()

            # Prepare feature summary
            feature_summary = f"""
            Image dimensions: {width}x{height}
            Average RGB color: {avg_color}
            Number of straight lines detected: {line_count}
            """

            # Add interpretation hints based on image features
            if line_count > 10:
                feature_summary += "This image likely contains a chart, diagram or structured content.\n"

            # Add file size information
            file_size = os.path.getsize(image_path)
            feature_summary += f"File size: {file_size} bytes\n"

            return {
                "width": width,
                "height": height,
                "format": format,
                "mode": mode,
                "info": info,
                "avg_color": avg_color,
                "line_count": line_count,
                "feature_summary": feature_summary.strip(),
                "file_size": file_size,
            }
        else:
            # Return basic metadata if OpenCV processing fails
            return {
                "width": width,
                "height": height,
                "format": format,
                "mode": mode,
                "info": info,
            }
    except Exception as e:
        logger.error(f"Error extracting image metadata: {str(e)}")
        return {}


@lru_cache(maxsize=10)
def _get_api_config():
    """Get API configuration with caching to avoid repeated lookups"""
    return {
        "model": getattr(settings, "OPENAI_MODEL", "gpt-4o"),
        "max_tokens": getattr(settings, "OPENAI_MAX_TOKENS", 1000),
        "temperature": getattr(settings, "OPENAI_TEMPERATURE", 0.0),
    }


def _analyze_image(
    image_path: str, filename: str, ocr_text: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """Analyze image using AI to get description and detect objects"""
    try:
        # Extract image features to enhance analysis
        feature_text = extract_image_features(image_path)
        feature_text = f"OCR Text extracted: {ocr_text}\n{feature_text}"

        # Get image description using separated function - memory optimized
        description = _analyze_image_description(image_path, filename, feature_text)

        # Detect objects using separated function - memory optimized
        detected_objects = _detect_objects_in_image(image_path, filename, feature_text)

        return description, detected_objects

    except Exception as e:
        logger.error(f"Error in image analysis process: {str(e)}")
        return f"Error analyzing image: {str(e)}", []


def _analyze_image_description(
    image_path: str, filename: str, feature_text: str
) -> str:
    """Analyze image using AI to get a comprehensive description with retry logic"""

    @retry_openai_call()
    def _make_openai_call():
        # Use optimized encoding for memory efficiency
        encoded_image = optimize_image_for_api(image_path)

        # Get OpenAI client
        client = get_openai_client()

        # Create a prompt focused on image analysis
        analysis_prompt = f"""
        Filename: {filename}
        {feature_text}

        Based on the extracted features, please provide a comprehensive analysis of what this image likely contains.

        If it appears to be a chart, graph, or other data visualization:
        - Identify the probable chart type (bar chart, line graph, pie chart, scatter plot, etc.).
        - Describe the likely subject of the chart: what is being measured or compared.
        - Specify what the X-axis and Y-axis represent (including units if available).
        - Extract **all data points** shown in the chart:
        - **If there are multiple values in a single bar/point/segment (e.g., stacked bars, grouped bars, or subcategories), extract them separately.**
        - **If exact values are not labeled, estimate based on axis markers.**
        - **Provide the extracted data in a structured table format.**
        - Table columns should dynamically match the information available (e.g., Category, Value 1, Value 2, etc.).
        - **Avoid using qualitative terms** like "high", "low", "increase", "spike", etc. Always provide numeric or factual information.

        If it appears to be a photograph or non-chart image:
        - Identify the probable subject matter or scene.
        - List key objects or elements present.
        - Mention any relevant contextual information inferred from the image features.

        Additionally, for all image types:
        - **Count and report** the number of each distinct object detected (e.g., 3 cars, 2 trees, 5 people).
        - **Accuracy in counting is important.**

        Finally, if any values are estimated rather than explicitly shown, note that clearly at the end.
        """

        # Get API configuration
        api_config = _get_api_config()

        # Call GPT-4o with vision capabilities
        response = client.chat.completions.create(
            model=api_config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing images and providing detailed descriptions.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=api_config["max_tokens"],
            temperature=api_config["temperature"],
        )

        analysis = response.choices[0].message.content.strip()

        # Free memory after API call
        encoded_image = None
        gc.collect()

        return analysis

    try:
        result = _make_openai_call()
        logger.info(f"Successfully generated image analysis ({len(result)} chars)")
        return result

    except Exception as e:
        logger.error(f"Error analyzing image description after retries: {str(e)}")
        return f"Error analyzing image: {str(e)}"


def _create_object_detection_prompt(filename: str, feature_text: str) -> str:
    """Create the prompt for object detection"""
    return f"""
    Filename: {filename}
    {feature_text}

    Based on the visual features extracted from this image, perform an object detection analysis.

    Try to identify and list prominent objects in the image (such as axes, legends, bars, labels, people, tools, charts, etc.).

    IMPORTANT: Please return ONLY a valid JSON array. Do not include any explanatory text before or after the JSON.

    Please return a JSON array where each object follows this structure:
    {{
        "object_name": "string",
        "confidence": float (between 0 and 1),
        "coordinates": [x_min, y_min, x_max, y_max]
    }}

    Example:
    [
        {{
        "object_name": "Person",
        "confidence": 0.94,
        "coordinates": [32, 45, 180, 300]
        }},
        ...
    ]

    If no object is detected, return an empty array: []

    Make sure:
    - All property names are in double quotes
    - All string values are in double quotes
    - Confidence is a number between 0 and 1
    - Coordinates are an array of 4 integers [x_min, y_min, x_max, y_max]
    - No trailing commas
    - No comments in the JSON"""


def _clean_markdown_content(content: str) -> str:
    """Clean markdown code blocks from content."""
    if "```" in content:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            return match.group(1)
    return content


def _extract_json_array(content: str) -> str:
    """Extract JSON array from content, removing extra text."""
    # Try to find a complete JSON array first
    array_match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
    if array_match:
        return array_match.group(0)

    # Try to find a JSON object that might contain an array
    object_match = re.search(r'\{\s*"[^"]*"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
    if object_match:
        return object_match.group(0)

    # Fallback to looking for just the array content between brackets
    bracket_match = re.search(r"\[(.*?)\]", content, re.DOTALL)
    if bracket_match:
        return f"[{bracket_match.group(1)}]"

    return content


def _normalize_detected_objects(detected_objects: Any) -> List[Dict[str, Any]]:
    """Normalize the parsed JSON response to a list format."""
    # Handle case where response might be wrapped in an object
    if isinstance(detected_objects, dict) and "objects" in detected_objects:
        return detected_objects["objects"]
    elif isinstance(detected_objects, list):
        return detected_objects
    else:
        return []


def _get_valid_confidence(confidence_value: Any) -> float:
    """Get a valid confidence value, defaulting to 0.5 if invalid."""
    if isinstance(confidence_value, (int, float)):
        return float(confidence_value)
    return 0.5


def _get_valid_coordinates(coords: Any) -> List[int]:
    """Get valid coordinates, defaulting to [0, 0, 0, 0] if invalid."""
    if coords and isinstance(coords, list) and len(coords) == 4:
        return coords
    return [0, 0, 0, 0]


def _validate_and_format_object(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and format a single detected object."""
    if not isinstance(obj, dict) or "object_name" not in obj:
        return None

    return {
        "object_name": str(obj.get("object_name", "")),
        "confidence": _get_valid_confidence(obj.get("confidence")),
        "coordinates": _get_valid_coordinates(obj.get("coordinates")),
    }


def _validate_detected_objects(
    detected_objects: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Validate and format all detected objects."""
    valid_objects = []
    for obj in detected_objects:
        valid_obj = _validate_and_format_object(obj)
        if valid_obj:
            valid_objects.append(valid_obj)
    return valid_objects


def _is_valid_text_line(line: str) -> bool:
    """Check if a line contains valid object name text (not JSON syntax)."""
    line = line.strip()
    return (
        line
        and not line.startswith("{")
        and not line.startswith("[")
        and not line.startswith("}")
        and not line.startswith("]")
    )


def _extract_text_objects_fallback(content: str) -> List[Dict[str, Any]]:
    """Fallback method to extract object names from plain text or malformed JSON."""
    text_objects = []

    # Try to extract object names from JSON-like patterns even if malformed
    object_name_patterns = [
        r'"object_name"\s*:\s*"([^"]+)"',  # Standard JSON format
        r"'object_name'\s*:\s*'([^']+)'",  # Single quotes
        r"object_name\s*:\s*[\"']([^\"']+)[\"']",  # Unquoted property name
        r'"([^"]+)"\s*,\s*"confidence"',  # Object name before confidence
    ]

    for pattern in object_name_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match and len(match.strip()) > 0:
                text_objects.append({"object_name": match.strip()})

    # If no patterns matched, try line-by-line extraction
    if not text_objects:
        lines = content.split("\n")
        for line in lines:
            if _is_valid_text_line(line):
                text_objects.append({"object_name": line.strip()})

    return text_objects


def _fix_json_issues(content: str) -> str:
    """Attempt to fix common JSON formatting issues"""
    try:
        # Remove any trailing commas before closing brackets/braces
        content = re.sub(r",(\s*[}\]])", r"\1", content)

        # Fix unescaped quotes in strings - be more careful about this
        # Look for quotes that appear to be inside string values
        def fix_quotes_in_strings(match):
            full_match = match.group(0)
            # If it's a property name (has colon after), leave it alone
            if ":" in full_match.split('"')[-1]:
                return full_match
            # Otherwise, escape the quote
            return full_match.replace('"', '\\"')

        # More careful quote fixing - only fix quotes that appear to be in string values
        content = re.sub(r'"([^"]*)"([^:,}\]]*)"', r'"\1\"\2"', content)

        # Ensure property names are properly quoted
        # Look for unquoted property names and quote them
        content = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', content)

        # Fix single quotes to double quotes (JSON standard)
        content = re.sub(r"(?<!\\)'", '"', content)

        # Remove any control characters that might cause issues
        content = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", content)

        # Try to fix missing commas between objects
        content = re.sub(r"}\s*{", "},{", content)

        # Fix missing commas between array elements
        content = re.sub(r"}\s*\n\s*{", "},{", content)

        # Remove any extra commas at the end
        content = re.sub(r",(\s*[}\]])", r"\1", content)

        return content

    except Exception as e:
        logger.warning(f"Error in _fix_json_issues: {str(e)}")
        return content


def _parse_object_detection_response(content: str) -> List[Dict[str, Any]]:
    """Parse the GPT response for object detection with improved reliability"""
    # Clean up content
    content = _clean_markdown_content(content)
    content = _extract_json_array(content)

    try:
        # Try parsing the JSON response
        detected_objects = json.loads(content)

        # Normalize to list format
        detected_objects = _normalize_detected_objects(detected_objects)

        # Validate and format objects
        return _validate_detected_objects(detected_objects)

    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Failed to parse object detection response as JSON: {str(e)}")
        logger.debug(
            f"Raw content that failed to parse (first 1000 chars): {content[:1000]}..."
        )

        # Try to fix common JSON issues before fallback
        try:
            # Fix common JSON issues
            fixed_content = _fix_json_issues(content)
            logger.debug(f"Attempting to parse fixed JSON: {fixed_content[:500]}...")
            detected_objects = json.loads(fixed_content)
            detected_objects = _normalize_detected_objects(detected_objects)
            logger.info("Successfully parsed JSON after fixing issues")
            return _validate_detected_objects(detected_objects)
        except (json.JSONDecodeError, AttributeError) as e2:
            logger.warning(f"JSON fix attempt also failed: {str(e2)}")
            logger.debug(
                f"Fixed content that still failed (first 1000 chars): {fixed_content[:1000] if 'fixed_content' in locals() else 'N/A'}..."
            )

        # Fallback: Try to extract plain text object names
        logger.info("Falling back to pattern-based object extraction")
        text_objects = _extract_text_objects_fallback(content)
        logger.info(f"Extracted {len(text_objects)} objects using fallback method")
        return text_objects if text_objects else []


def _detect_objects_in_image(
    image_path: str, filename: str, feature_text: str
) -> List[Dict[str, Any]]:
    """Detect objects in an image using AI with memory optimization and retry logic"""

    @retry_openai_call()
    def _make_openai_call():
        # Use optimized encoding for memory efficiency
        encoded_image = optimize_image_for_api(image_path)

        # Get OpenAI client
        client = get_openai_client()

        # Create detection prompt
        detection_prompt = _create_object_detection_prompt(filename, feature_text)

        # Get API configuration
        api_config = _get_api_config()

        # Call GPT-4o with vision capabilities
        response = client.chat.completions.create(
            model=api_config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at detecting and identifying objects in images. Return a JSON array of object names.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=800,
            temperature=0.0,  # Use 0 temperature for more consistent JSON output
        )

        content = response.choices[0].message.content.strip()

        # Parse and return the results
        detected_objects = _parse_object_detection_response(content)

        # Free memory after API call
        encoded_image = None
        gc.collect()

        return detected_objects

    try:
        result = _make_openai_call()
        logger.info(f"Successfully detected {len(result)} objects")
        return result

    except Exception as e:
        logger.error(f"Error detecting objects in image after retries: {str(e)}")
        return []


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
            logger.info(
                f"Successfully acquired image processing lock: {processing_key}"
            )
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
                f"Image processing lock already exists for key {processing_key}, held by task: {existing_task_id}"
            )
            return False, existing_task_id

    except Exception as e:
        logger.error(
            f"Error acquiring image processing lock {processing_key}: {str(e)}"
        )
        # If Redis is unavailable, allow processing to continue
        return True, None
