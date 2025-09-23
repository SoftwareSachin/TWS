"""
Celery task for audio document ingestion using Document model (v2).
Uses the Document model directly instead of FileIngestion model.
Enhanced with rate limiting, retry logic, and consistent error handling.
"""

import os
import secrets
import time
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import azure.cognitiveservices.speech as speechsdk
from pydub.utils import mediainfo
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
from app.utils.ingestion_status_propagation import (
    propagate_ingestion_status,
    should_skip_processing_due_to_timeout,
)
from app.utils.ingestion_utils_v2 import acquire_tokens_with_retry
from app.utils.openai_utils import (
    chat_completion_with_retry,
    generate_embedding_with_retry,
)
from app.utils.processing_lock_utils import (
    _cleanup_processing_lock,
    _handle_stale_lock,
)
from app.utils.uuid6 import uuid7


class BinaryFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
    """Callback class to handle audio stream reading"""

    def __init__(self, filename: str):
        super().__init__()
        self._file_h = open(filename, "rb")
        logger.info(f"Initialized BinaryFileReaderCallback for file: {filename}")

    def read(self, buffer: memoryview) -> int:
        try:
            size = buffer.nbytes
            frames = self._file_h.read(size)
            buffer[: len(frames)] = frames
            return len(frames)
        except Exception as ex:
            logger.error(f"Exception in audio stream read: {ex}")
            raise

    def close(self) -> None:
        try:
            self._file_h.close()
            logger.info("Closed BinaryFileReaderCallback for file.")
        except Exception as ex:
            logger.error(f"Exception in audio stream close: {ex}")
            raise


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
                f"Successfully acquired audio processing lock: {processing_key}"
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
                f"Audio processing lock already exists for key {processing_key}, held by task: {existing_task_id}"
            )
            return False, existing_task_id

    except Exception as e:
        logger.error(
            f"Error acquiring audio processing lock {processing_key}: {str(e)}"
        )
        # If Redis is unavailable, allow processing to continue
        return True, None


def _log_task_start(
    file_path: str, retry_count: int, max_retries: int, retry_reason: Optional[str]
) -> None:
    """Log the task start information."""
    logger.info(
        f"Starting audio_ingestion_task_v2 for file: {file_path} (retry {retry_count}/{max_retries})"
    )
    if retry_reason:
        logger.info(f"Retry reason: {retry_reason}")


def _validate_file_exists(
    file_id: str, ingestion_id: str, task_id: str, file_path: str
) -> Optional[Dict[str, Any]]:
    """Validate that the audio file exists. Returns error response if not found, None if valid."""
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return {
            "file_id": file_id,
            "ingestion_id": ingestion_id,
            "task_id": task_id,
            "success": False,
            "status": "exception",
            "error": f"Audio file not found: {file_path}",
        }
    return None


def _handle_existing_document(
    db,
    file_id: str,
    dataset_id: str,
    ingestion_id: str,
    skip_successful_files: bool,
    user_id: UUID,
) -> Optional[Dict[str, Any]]:
    """Handle existing document logic. Returns skip response if applicable, None otherwise."""
    existing_doc = (
        db.query(Document)
        .filter(
            Document.file_id == UUID(file_id),
            Document.dataset_id == UUID(dataset_id),
            (Document.ingestion_id == UUID(ingestion_id) if ingestion_id else None),
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
        logger.info(f"Document already exists and is successful for file {file_id}")

        if user_id:
            propagate_ingestion_status(db, "audio", existing_doc.id, user_id)

        return {
            "success": True,
            "document_id": str(existing_doc.id),
            "file_id": file_id,
            "dataset_id": dataset_id,
            "status": "success",
            "skipped": True,
        }

    return None


def _handle_rate_limiting(
    file_path: str,
    file_id: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    metadata: Optional[dict],
    chunk_size: int,
    chunk_overlap: int,
    skip_successful_files: bool,
    retry_count: int,
    max_retries: int,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    """Handle rate limiting logic. Returns retry response if needed, None if processing can continue."""
    # Extract audio metadata for token estimation
    audio_metadata = _extract_audio_metadata(file_path)
    duration = audio_metadata.get("duration", 0)

    # Estimate processing tokens
    token_count = _estimate_audio_processing_tokens(file_path, duration)

    # Rate limiting - acquire tokens before processing
    try:
        success, remaining_time = acquire_tokens_with_retry(token_count)
        if not success:
            logger.warning(
                f"RateLimit: Denied {token_count} tokens | Retry in: {remaining_time}s | Used: 0"
            )

            if retry_count >= max_retries:
                logger.error(
                    f"Max retries ({max_retries}) exceeded for audio ingestion"
                )
                return {
                    "file_id": file_id,
                    "ingestion_id": ingestion_id,
                    "task_id": task_id,
                    "success": False,
                    "status": "max_retries_exceeded",
                    "error": f"Rate limit exceeded after {max_retries} retries",
                }

            return _schedule_audio_retry(
                remaining_time=remaining_time,
                retry_count=retry_count,
                max_retries=max_retries,
                file_path=file_path,
                file_id=file_id,
                metadata=metadata,
                ingestion_id=ingestion_id,
                dataset_id=dataset_id,
                user_id=user_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                skip_successful_files=skip_successful_files,
                task_id=task_id,
            )

    except Exception as e:
        logger.error(f"Error in token acquisition: {str(e)}")
        # Continue processing if rate limiting fails (fallback)

    return None


def _update_document_metadata(
    db, document: Document, file_path: str, document_id: str
) -> None:
    """Update document with audio metadata and set processing status."""
    audio_metadata = _extract_audio_metadata(file_path)
    duration = audio_metadata.get("duration", 0)
    token_count = _estimate_audio_processing_tokens(file_path, duration)

    # Merge metadata
    merged_metadata = {
        **(document.document_metadata or {}),
        **audio_metadata,
        "estimated_tokens": token_count,
        "processing_started_at": datetime.now(UTC).isoformat(),
    }
    document.document_metadata = merged_metadata
    attributes.flag_modified(document, "document_metadata")
    db.commit()
    logger.info("Updated document with audio metadata.")

    # Update status to Extracting
    document.processing_status = DocumentProcessingStatusEnum.Extracting
    db.commit()
    logger.info(f"Set document status to Extracting for document ID: {document_id}")


def _process_audio_transcription(file_path: str) -> Tuple[str, List[float]]:
    """Process audio transcription and generate embedding."""
    # Transcribe audio
    speech_config = _get_speech_config()
    transcription = _transcribe_audio(speech_config, file_path)

    logger.info("Successfully transcribed audio file")

    # Generate embedding for full transcription
    try:
        transcription_embedding = generate_embedding_with_retry(transcription)
        logger.info("Generated embedding for transcription")
        return transcription, transcription_embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise ValueError(f"Embedding generation failed: {str(e)}")


def _create_chunks_and_extract_speakers(
    db, transcription: str, document_id: str, chunk_size: int, chunk_overlap: int
) -> Tuple[List[str], List[str]]:
    """Create text chunks and extract speaker information."""
    chunk_ids = []
    chunks = _chunk_text(transcription, chunk_size, chunk_overlap)

    # Process each chunk
    for i, (chunk_text, start_idx, end_idx) in enumerate(chunks):
        chunk = DocumentChunk(
            id=uuid7(),
            document_id=UUID(document_id),
            chunk_type=ChunkTypeEnum.AudioSegment,
            chunk_text=chunk_text,
            chunk_embedding=generate_embedding_with_retry(chunk_text),
            chunk_metadata={
                "chunk_order": i,
                "start_index": start_idx,
                "end_index": end_idx,
                "chunked_by_engine": "AudioChunkerV2",
                "status": DocumentProcessingStatusEnum.Success.value,
                "processed_at": datetime.now(UTC).isoformat(),
            },
        )
        db.add(chunk)
        chunk_ids.append(str(chunk.id))

    logger.info(f"Created {len(chunks)} transcription chunks")

    # Extract speaker names
    speaker_names = _extract_speaker_names(transcription)

    if speaker_names:
        combined_speakers = "; ".join(speaker_names)
        chunk = DocumentChunk(
            id=uuid7(),
            document_id=UUID(document_id),
            chunk_type=ChunkTypeEnum.Speaker,
            chunk_text=combined_speakers,
            chunk_embedding=generate_embedding_with_retry(combined_speakers),
            chunk_metadata={
                "speaker_count": len(speaker_names),
                "speakers": speaker_names,
                "chunked_by_engine": "GPT4SpeakerExtractorV2",
                "status": DocumentProcessingStatusEnum.Success.value,
                "processed_at": datetime.now(UTC).isoformat(),
            },
        )
        db.add(chunk)
        chunk_ids.append(str(chunk.id))
        logger.info(f"Added speaker chunk with {len(speaker_names)} speakers")

    return chunk_ids, speaker_names


def _finalize_document_processing(
    db,
    document: Document,
    transcription: str,
    transcription_embedding: List[float],
    chunk_ids: List[str],
    speaker_names: List[str],
    user_id: UUID,
) -> None:
    """Finalize document processing with transcription and metadata updates."""
    # Update document with transcription and embedding
    document.description = transcription
    document.description_embedding = transcription_embedding
    document.processing_status = DocumentProcessingStatusEnum.Success
    document.processed_at = datetime.now(UTC)

    # Update metadata with final processing info
    document.document_metadata.update(
        {
            "processing_completed_at": datetime.now(UTC).isoformat(),
            "transcription_length": len(transcription),
            "chunk_count": len(chunk_ids),
            "speaker_count": len(speaker_names) if speaker_names else 0,
        }
    )
    attributes.flag_modified(document, "document_metadata")

    db.commit()
    logger.info(f"Marked document {document.id} as successfully processed")

    # Propagate success status up to file and dataset level
    if user_id:
        propagate_ingestion_status(db, "document", document.id, user_id)


def _create_success_response(
    document_id: str,
    file_id: str,
    dataset_id: str,
    chunk_ids: List[str],
    speaker_names: List[str],
    transcription: str,
    retry_count: int,
) -> Dict[str, Any]:
    """Create the success response dictionary."""
    return {
        "success": True,
        "document_id": document_id,
        "file_id": file_id,
        "dataset_id": dataset_id,
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "speaker_count": len(speaker_names) if speaker_names else 0,
        "transcription_length": len(transcription),
        "retry_count": retry_count,
    }


@celery.task(name="tasks.audio_ingestion_task_v2", bind=True, acks_late=True)
def audio_ingestion_task_v2(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    metadata: Optional[dict] = None,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    skip_successful_files: bool = True,
    retry_count: int = 0,
    max_retries: int = 10,
    retry_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an audio file, transcribe it, and extract metadata using Document model (v2).

    Args:
        file_id: ID of the file being processed
        file_path: Path to the audio file
        ingestion_id: ID of the current ingestion batch
        dataset_id: ID of the dataset
        user_id: ID of the user who initiated the ingestion
        metadata: Additional metadata to store with the document
        chunk_size: Size of text chunks (in words)
        chunk_overlap: Number of overlapping words between chunks
        skip_successful_files: Whether to skip already processed files
        retry_count: Current retry attempt count
        max_retries: Maximum number of retries before giving up
        retry_reason: Reason for the retry

    Returns:
        Dictionary with processing results"""

    # Robust deduplication with deadlock prevention for audio processing
    processing_key = None
    redis_client = None
    task_id = self.request.id

    try:
        redis_client = get_redis_client_sync()

        # Create unique processing key for audio
        processing_key = (
            f"processing:audio:{file_id}:ingestion:{ingestion_id}:dataset:{dataset_id}"
        )

        # Try to acquire processing lock with stale detection
        acquired, existing_task_id = _acquire_processing_lock_atomic(
            redis_client, processing_key, task_id, expiry_seconds=3600
        )

        if not acquired:
            # Lock exists - check if original task is still alive
            if existing_task_id and not _handle_stale_lock(
                redis_client, processing_key, existing_task_id, self.request.id
            ):
                logger.info(
                    f"Image {file_id} is being processed by active task {existing_task_id} - skipping"
                )
                # Close Redis connection before returning
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
                f"Acquired audio processing lock after stale lock handling: {processing_key}"
            )

        logger.info(f"Acquired audio processing lock: {processing_key}")

    except Exception as e:
        # Don't fail the task if Redis is unavailable, just log it
        logger.warning(f"Redis deduplication not available: {str(e)}")
        processing_key = None
        redis_client = None

    try:
        _log_task_start(
            file_path, retry_count, max_retries, retry_reason
        )  # Early validation
        file_check_result = _validate_file_exists(
            file_id, ingestion_id, self.request.id, file_path
        )
        if file_check_result:
            # Clean up processing lock before early return
            if processing_key and redis_client:
                try:
                    _cleanup_processing_lock(redis_client, processing_key, task_id)
                except Exception as e:
                    logger.warning(f"Failed to clean up lock on early return: {str(e)}")
            return file_check_result

        document_id = None

        # Check for existing document and handle skipping
        with SyncSessionLocal() as db:
            skip_result = _handle_existing_document(
                db, file_id, dataset_id, ingestion_id, skip_successful_files, user_id
            )
            if skip_result:
                # Clean up processing lock before early return
                if processing_key and redis_client:
                    try:
                        _cleanup_processing_lock(redis_client, processing_key, task_id)
                    except Exception as e:
                        logger.warning(
                            f"Failed to clean up lock on early return: {str(e)}"
                        )
                return skip_result

            # Create/get document and prepare for processing
            document = _get_or_create_document(
                db,
                file_id=file_id,
                dataset_id=dataset_id,
                file_path=file_path,
                task_id=self.request.id,
                ingestion_id=ingestion_id,
                metadata=metadata,
                skip_successful_files=skip_successful_files,
            )

            # Check if document was created and if it's been failed due to timeout
            logger.info(f"CHECKING if document {document.id} is failed due to timeout")
            if skip_result and skip_result.get("document_id"):
                should_skip, skip_reason = should_skip_processing_due_to_timeout(
                    UUID(skip_result["document_id"])
                )
                if should_skip:
                    logger.warning(
                        f"⏰ SKIPPING audio processing for {file_id}: {skip_reason}"
                    )
                    return {
                        "file_id": file_id,
                        "success": True,
                        "skipped": True,
                        "reason": "document_timeout_failed",
                        "message": skip_reason,
                        "document_id": skip_result["document_id"],
                    }

            document_id = str(document.id)

            # Handle rate limiting
            rate_limit_result = _handle_rate_limiting(
                file_path,
                file_id,
                ingestion_id,
                dataset_id,
                user_id,
                metadata,
                chunk_size,
                chunk_overlap,
                skip_successful_files,
                retry_count,
                max_retries,
                self.request.id,
            )
            if rate_limit_result:
                # Clean up processing lock before early return
                if processing_key and redis_client:
                    try:
                        _cleanup_processing_lock(redis_client, processing_key, task_id)
                    except Exception as e:
                        logger.warning(
                            f"Failed to clean up lock on early return: {str(e)}"
                        )
                return rate_limit_result

            # Update document with metadata and set status to Extracting
            _update_document_metadata(db, document, file_path, document_id)

            # Process audio transcription
            transcription, transcription_embedding = _process_audio_transcription(
                file_path
            )

            # Create chunks and process speakers
            chunk_ids, speaker_names = _create_chunks_and_extract_speakers(
                db, transcription, document_id, chunk_size, chunk_overlap
            )

            # Finalize document processing
            _finalize_document_processing(
                db,
                document,
                transcription,
                transcription_embedding,
                chunk_ids,
                speaker_names,
                user_id,
            )

            return _create_success_response(
                document_id,
                file_id,
                dataset_id,
                chunk_ids,
                speaker_names,
                transcription,
                retry_count,
            )

    except Exception as e:
        return _handle_audio_exception(
            exception=e,
            document_id=document_id,
            task_id=self.request.id,
            ingestion_id=ingestion_id,
            user_id=user_id,
            file_id=file_id,
            retry_count=retry_count,
            max_retries=max_retries,
            file_path=file_path,
            metadata=metadata,
            dataset_id=dataset_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_successful_files=skip_successful_files,
        )

    finally:
        # Clean up processing lock safely
        if processing_key and redis_client:
            try:
                _cleanup_processing_lock(redis_client, processing_key, task_id)
            except Exception as e:
                logger.warning(f"Failed to clean up audio processing lock: {str(e)}")

        # Close Redis connection
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")


def _estimate_audio_processing_tokens(file_path: str, duration: float) -> int:
    """Estimate processing tokens for audio file based on duration and size."""
    try:
        # Estimate based on typical transcription output: ~150 words per minute
        # Average ~1.3 tokens per word
        estimated_words = duration * 150 / 60  # duration in minutes
        estimated_tokens = int(estimated_words * 1.3)

        # Add overhead for speaker extraction and metadata processing
        overhead_tokens = min(1000, estimated_tokens * 0.1)
        total_tokens = int(estimated_tokens + overhead_tokens)

        logger.info(
            f"Estimated audio processing tokens: {total_tokens} (duration: {duration}s)"
        )
        return max(100, total_tokens)  # Minimum 100 tokens
    except Exception as e:
        logger.error(f"Error estimating audio tokens: {str(e)}")
        # Fallback: estimate based on file size (1 token per 4 bytes roughly)
        try:
            file_size = os.path.getsize(file_path)
            return max(500, file_size // 4)
        except (OSError, IOError) as file_error:
            logger.error(f"Error getting file size for {file_path}: {str(file_error)}")
            return 1000  # Default fallback


def _schedule_audio_retry(
    remaining_time: float,
    retry_count: int,
    max_retries: int,
    file_path: str,
    file_id: str,
    metadata: Optional[dict],
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    chunk_size: int,
    chunk_overlap: int,
    skip_successful_files: bool,
    task_id: str,
) -> Dict[str, Any]:
    """Schedule a retry of the audio ingestion task."""
    base_delay = remaining_time or settings.RATE_LIMIT_RETRY_SECONDS * (2**retry_count)
    max_backoff = 300  # 5 minutes max
    backoff = min(base_delay, max_backoff)
    jitter = (float(secrets.randbelow(20) - 10) / 100.0) * backoff
    countdown = max(1, backoff + jitter)

    logger.info(
        f"Could not acquire tokens for audio {file_id}. Retry {retry_count+1}/{max_retries} scheduled in {countdown:.1f}s"
    )

    celery.signature(
        "tasks.audio_ingestion_task_v2",
        kwargs={
            "file_path": file_path,
            "file_id": file_id,
            "metadata": metadata,
            "ingestion_id": ingestion_id,
            "dataset_id": dataset_id,
            "user_id": user_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "skip_successful_files": skip_successful_files,
            "retry_count": retry_count + 1,
            "max_retries": max_retries,
            "retry_reason": "rate_limit_exceeded",
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


def _is_retryable_error(exception: Exception) -> bool:
    """Check if the exception is a retryable error."""
    retryable_errors = [
        "transcription timeout",
        "azure speech service",
        "rate limit",
        "connection error",
        "timeout",
        "temporary failure",
    ]

    return any(
        error_phrase in str(exception).lower() for error_phrase in retryable_errors
    )


def _handle_retryable_error(
    retry_count: int,
    max_retries: int,
    file_path: str,
    file_id: str,
    metadata: Optional[dict],
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    chunk_size: int,
    chunk_overlap: int,
    skip_successful_files: bool,
    task_id: str,
) -> Dict[str, Any]:
    """Handle retryable error by scheduling a retry."""
    logger.info(
        f"Retryable error detected. Scheduling retry {retry_count + 1}/{max_retries}"
    )

    return _schedule_audio_retry(
        remaining_time=60,  # Default 1-minute delay for errors
        retry_count=retry_count,
        max_retries=max_retries,
        file_path=file_path,
        file_id=file_id,
        metadata=metadata,
        ingestion_id=ingestion_id,
        dataset_id=dataset_id,
        user_id=user_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        skip_successful_files=skip_successful_files,
        task_id=task_id,
    )


def _update_document_failure_status(
    document_id: str, exception: Exception, retry_count: int, user_id: Optional[UUID]
) -> None:
    """Update document status to Failed and propagate status."""
    try:
        with SyncSessionLocal() as db:
            document = (
                db.query(Document).filter(Document.id == UUID(document_id)).first()
            )
            if document:
                document.processing_status = DocumentProcessingStatusEnum.Failed
                document.error_message = str(exception)
                document.processed_at = datetime.now(UTC)

                if document.document_metadata:
                    document.document_metadata.update(
                        {
                            "error_at": datetime.now(UTC).isoformat(),
                            "retry_count": retry_count,
                        }
                    )
                    attributes.flag_modified(document, "document_metadata")

                db.commit()
                logger.info(f"Updated document {document_id} status to Failed")

                # Propagate failed status
                if user_id:
                    propagate_ingestion_status(db, "document", document.id, user_id)
    except Exception as db_error:
        logger.error(f"Error updating document status: {str(db_error)}")


def _send_failure_notification(user_id: UUID, document_id: str) -> None:
    """Send failure notification to user."""
    try:
        with SyncSessionLocal() as db:
            propagate_ingestion_status(db, "audio", UUID(document_id), user_id)
    except Exception as notify_error:
        logger.error(f"Error sending failure notification: {str(notify_error)}")


def _create_failure_response(
    file_id: str,
    document_id: Optional[str],
    ingestion_id: Optional[str],
    task_id: str,
    exception: Exception,
    retry_count: int,
) -> Dict[str, Any]:
    """Create the failure response dictionary."""
    return {
        "success": False,
        "file_id": file_id,
        "document_id": document_id,
        "ingestion_id": ingestion_id,
        "task_id": task_id,
        "error": str(exception),
        "status": "exception",
        "retry_count": retry_count,
    }


def _handle_audio_exception(
    exception: Exception,
    document_id: Optional[str],
    task_id: str,
    ingestion_id: Optional[str],
    user_id: Optional[UUID],
    file_id: str,
    retry_count: int,
    max_retries: int,
    file_path: str,
    metadata: Optional[dict],
    dataset_id: str,
    chunk_size: int,
    chunk_overlap: int,
    skip_successful_files: bool,
) -> Dict[str, Any]:
    """Handle exceptions during the audio ingestion process."""
    error_message = f"{str(exception)}\n{traceback.format_exc()}"
    logger.error(f"Error processing audio document: {error_message}")

    # Check for retryable errors and handle retry if applicable
    if _is_retryable_error(exception) and retry_count < max_retries:
        return _handle_retryable_error(
            retry_count,
            max_retries,
            file_path,
            file_id,
            metadata,
            ingestion_id,
            dataset_id,
            user_id,
            chunk_size,
            chunk_overlap,
            skip_successful_files,
            task_id,
        )

    # Update document status if available
    if document_id:
        _update_document_failure_status(document_id, exception, retry_count, user_id)

    # Send failure notification
    if user_id and document_id:
        _send_failure_notification(user_id, document_id)

    return _create_failure_response(
        file_id, document_id, ingestion_id, task_id, exception, retry_count
    )


def _get_or_create_document(
    db,
    file_id: str,
    dataset_id: str,
    file_path: str,
    task_id: str,
    ingestion_id: str,
    metadata: Dict[str, Any] = None,
    skip_successful_files: bool = True,
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
        # For re-ingestion, merge metadata properly
        if not skip_successful_files and metadata:
            merged_metadata = existing_doc.document_metadata or {}
            merged_metadata.update(metadata)
            existing_doc.document_metadata = merged_metadata
            attributes.flag_modified(existing_doc, "document_metadata")

        # Do not update documents that are already in failed/exception state
        if existing_doc.processing_status in [
            DocumentProcessingStatusEnum.Failed,
            DocumentProcessingStatusEnum.Exception,
        ]:
            logger.info(
                f"Document {existing_doc.id} is in {existing_doc.processing_status} state - not updating task_id, timestamp, or status"
            )
        else:
            existing_doc.task_id = task_id
            existing_doc.updated_at = datetime.now(UTC)
            existing_doc.processing_status = DocumentProcessingStatusEnum.Processing
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
        document_type=DocumentTypeEnum.Audio,
        processing_status=DocumentProcessingStatusEnum.Processing,
        file_path=file_path,
        file_size=file_size,
        mime_type=mime_type,
        document_metadata=metadata or {},
        task_id=task_id,
        ingestion_id=uuid.UUID(ingestion_id) if ingestion_id else None,
    )

    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(f"Created new document record with ID {document.id}")
    return document


def _delete_existing_document_chunks(record_id: UUID) -> None:
    """Delete existing document chunks for a given record ID"""
    if not record_id:
        logger.warning("No record ID provided for deletion")
        return
    try:
        with SyncSessionLocal() as db:
            existing_doc = db.query(Document).filter(Document.id == record_id).first()
            if existing_doc:
                deleted_count = (
                    db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == record_id)
                    .delete()
                )
                db.commit()
                logger.info(
                    f"Deleted {deleted_count} existing document chunks with ID {record_id}"
                )
    except Exception as e:
        logger.error(
            f"Error deleting existing document chunks: {str(e)}", exc_info=True
        )


def _get_mime_type(file_path: str) -> str:
    """Determine MIME type from file extension"""
    extension = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    return mime_types.get(extension, "application/octet-stream")


def _get_speech_config() -> speechsdk.SpeechConfig:
    """Get Azure Speech Config"""
    logger.info("Fetching Azure Speech configuration")
    speech_config = speechsdk.SpeechConfig(
        subscription=settings.AZURE_SPEECH_KEY, region=settings.AZURE_SPEECH_REGION
    )
    # Set language and other configurations
    speech_config.speech_recognition_language = "en-US"
    return speech_config


def _extract_audio_metadata(audio_file_path: str) -> Dict[str, Any]:
    """Extract metadata from audio file"""
    logger.info(f"Extracting metadata for audio file: {audio_file_path}")
    try:
        info = mediainfo(audio_file_path)
        logger.debug(f"Audio metadata: {info}")
        return {
            "duration": float(info.get("duration", 0.0)),
            "sample_rate": int(info.get("sample_rate", 0)),
            "bitrate": int(info.get("bit_rate", 0)),
            "channels": int(info.get("channels", 0)),
            "format": info.get("format_name", "unknown"),
            "file_size": (
                os.path.getsize(audio_file_path)
                if os.path.exists(audio_file_path)
                else 0
            ),
        }
    except Exception as e:
        logger.error(f"Error extracting audio metadata: {str(e)}")
        return {}


def _transcribe_audio(
    speech_config: speechsdk.SpeechConfig, audio_file_path: str
) -> str:
    """Transcribe audio file using Azure Speech Services - simplified version like v1"""
    logger.info(f"Transcribing audio file: {audio_file_path}")

    file_extension = os.path.splitext(audio_file_path)[1].lower()

    try:
        if file_extension == ".wav":
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )
        elif file_extension in [".mp3", ".aac", ".m4a"]:
            callback = BinaryFileReaderCallback(audio_file_path)

            compressed_format = speechsdk.audio.AudioStreamFormat(
                compressed_stream_format=(
                    speechsdk.AudioStreamContainerFormat.MP3
                    if file_extension == ".mp3"
                    else speechsdk.AudioStreamContainerFormat.ANY
                )
            )

            stream = speechsdk.audio.PullAudioInputStream(
                stream_format=compressed_format, pull_stream_callback=callback
            )

            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. Supported formats are .wav, .mp3, .aac, and .m4a"
            )

        all_results = []
        done = False

        def handle_result(evt):
            all_results.append(evt.result.text)

        def handle_stop(evt):
            nonlocal done
            done = True

        recognizer.recognized.connect(handle_result)
        recognizer.session_stopped.connect(handle_stop)
        recognizer.canceled.connect(handle_stop)

        recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.5)
        recognizer.stop_continuous_recognition()

        logger.info(f"Transcription completed for file: {audio_file_path}")
        transcription = " ".join(all_results)

        if not transcription.strip():
            logger.debug(transcription)
            logger.warning(
                "Transcription resulted in empty text - audio may be silent or in unsupported language"
            )

        return transcription

    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise


def _chunk_text(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[Tuple[str, int, int]]:
    """Split text into chunks with overlap"""
    logger.info(
        f"Chunking text into chunks of size {chunk_size} with overlap {chunk_overlap}"
    )
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        # Calculate character indices for metadata
        start_char = len(" ".join(words[:start]))
        end_char = len(" ".join(words[:end]))

        chunks.append((chunk, start_char, end_char))

        # Move to next chunk with overlap
        if end >= len(words):
            break
        start += chunk_size - chunk_overlap

    logger.info(f"Generated {len(chunks)} chunks from text")
    return chunks


def _extract_speaker_names(transcription: str) -> List[str]:
    """Extract speaker names from transcription using GPT-4"""
    logger.info("Extracting speaker names from transcription")
    try:
        # Limit transcription size to avoid token limits
        limited_transcription = transcription[:4000]

        prompt = f"""Extract unique speaker names from the following transcription.
        Look for patterns like "Dr. Smith:", "Patient:", "Speaker 1:", etc.
        Respond with ONLY a JSON array of strings like: ["Dr. Smith", "Patient", "Nurse"]
        If no speakers are found, respond with: []

        Transcription:
        {limited_transcription}
        """

        response = chat_completion_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts speaker names from transcriptions. Respond only with valid JSON arrays.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for consistent output
            max_tokens=500,  # Limit response size for speaker names
        )

        content = response.strip()

        # Remove markdown code blocks if present
        import re

        content = re.sub(
            r"^```(?:json)?\s*([\s\S]+?)\s*```$", r"\1", content, flags=re.MULTILINE
        )

        try:
            import json

            speaker_names = json.loads(content)

            # Validate that it's a list of strings
            if isinstance(speaker_names, list) and all(
                isinstance(name, str) for name in speaker_names
            ):
                logger.info(
                    f"Successfully extracted {len(speaker_names)} speaker names"
                )
                return speaker_names
            else:
                logger.warning(f"Invalid speaker names format: {speaker_names}")
                return []

        except json.JSONDecodeError:
            # Try with ast.literal_eval as fallback
            try:
                import ast

                speaker_names = ast.literal_eval(content)
                if isinstance(speaker_names, list):
                    return [str(name) for name in speaker_names]
                else:
                    logger.error(
                        f"Speaker extraction returned non-list: {speaker_names}"
                    )
                    return []
            except Exception as e:
                logger.error(f"Speaker name extraction failed: {e}")
                logger.error(f"Raw cleaned response was: {repr(content)}")
                return []

    except Exception as e:
        logger.error(f"Error extracting speaker names: {str(e)}")
        return []
