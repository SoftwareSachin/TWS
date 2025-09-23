"""
Celery task for video file ingestion using Document model (v2).
Supports video processing with segmentation, transcription, and captioning.
Follows the established patterns from other ingestion tasks.
"""

import json
import os
import secrets
import shutil
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

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
from app.utils.feature_flags import is_video_ingestion_enabled
from app.utils.ingestion_status_propagation import (
    propagate_ingestion_status,
    should_skip_processing_due_to_timeout,
)
from app.utils.ingestion_utils_v2 import (
    acquire_tokens_with_retry,
)
from app.utils.openai_utils import (
    chat_completion_with_retry,
    generate_embedding_with_retry,
)
from app.utils.processing_lock_utils import (
    _cleanup_processing_lock,
    _handle_stale_lock,
)
from app.utils.video_ingestion.utils import (
    merge_segment_information,
    segment_caption,
    speech_to_text,
    split_video,
)
from app.utils.video_model_manager import (
    cleanup_video_gpu_memory,
    get_caption_model,
    get_whisper_model,
)


def _get_workspace_id_from_file(file_id: str) -> Optional[str]:
    """Get workspace_id for a given file_id"""
    try:
        from app.db.session import SyncSessionLocal
        from app.models.file_model import File

        with SyncSessionLocal() as db:
            file_record = (
                db.query(File)
                .filter(File.id == UUID(file_id), File.deleted_at.is_(None))
                .first()
            )
            if file_record:
                return str(file_record.workspace_id)
            return None
    except Exception as e:
        logger.error(f"Error getting workspace_id for file {file_id}: {e}")
        return None


def _wait_for_processing_completion(
    processing_key: str, timeout_seconds: int = 7200
) -> bool:
    """
    Wait for another task to complete video processing.
    Uses exponential backoff with jitter to avoid thundering herd.
    """
    try:
        redis_client = get_redis_client_sync()
        start_time = time.time()
        wait_time = 1  # Start with 1 second
        max_wait = 60  # Max 60 seconds between checks

        while time.time() - start_time < timeout_seconds:
            # Check if lock still exists
            if not redis_client.exists(processing_key):
                logger.info(
                    "Video processing completed by another task, proceeding with segment reuse"
                )
                return True

            # Exponential backoff with jitter (using secrets for better randomness)
            jitter = 0.8 + (secrets.randbelow(40) / 100.0)  # 0.8 to 1.2
            actual_wait = min(wait_time * jitter, max_wait)

            logger.debug(
                f"Waiting {actual_wait:.1f}s for video processing to complete..."
            )
            time.sleep(actual_wait)

            wait_time = min(wait_time * 1.5, max_wait)  # Exponential backoff

        logger.warning("Timeout waiting for video processing completion")
        return False

    except Exception as e:
        logger.error(f"Error waiting for processing completion: {e}")
        return False


def _validate_existing_segments(segment_dir: str) -> bool:
    """
    Validate that existing segments are complete and usable.
    """
    try:
        # Check for metadata file
        metadata_file = os.path.join(segment_dir, "segments_metadata.json")
        if not os.path.exists(metadata_file):
            logger.warning(f"Missing metadata file: {metadata_file}")
            return False

        # Load and validate metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        required_fields = ["processing_completed", "segment_count", "total_duration"]
        if not all(field in metadata for field in required_fields):
            logger.warning(f"Invalid metadata structure in {metadata_file}")
            return False

        if not metadata.get("processing_completed", False):
            logger.warning("Processing not completed according to metadata")
            return False

        # Validate segment files exist using configured video output format
        config = _get_video_config()
        video_output_format = config["video_output_format"]
        segment_count = metadata.get("segment_count", 0)
        actual_segments = len(
            [
                f
                for f in os.listdir(segment_dir)
                if f.endswith(f".{video_output_format}")
            ]
        )

        if actual_segments < segment_count:
            logger.warning(
                f"Segment count mismatch: expected {segment_count}, found {actual_segments}"
            )
            return False

        logger.info(f"Validated {segment_count} segments in {segment_dir}")
        return True

    except Exception as e:
        logger.error(f"Error validating segments: {e}")
        return False


def _load_existing_segments(segment_dir: str) -> Dict[str, Any]:
    """
    Load existing segment data from workspace storage.
    """
    try:
        metadata_file = os.path.join(segment_dir, "segments_metadata.json")
        with open(metadata_file, "r") as f:
            segments_data = json.load(f)

        logger.info(f"Loaded existing segments from {segment_dir}")
        return segments_data

    except Exception as e:
        logger.error(f"Error loading existing segments: {e}")
        raise


def _save_segments_metadata(segment_dir: str, segments_data: Dict[str, Any]) -> None:
    """
    Save segment metadata to workspace storage.
    """
    try:
        metadata_file = os.path.join(segment_dir, "segments_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(segments_data, f, indent=2)

        logger.info(f"Saved segments metadata to {metadata_file}")

    except Exception as e:
        logger.error(f"Error saving segments metadata: {e}")
        raise


def _handle_workspace_video_processing(
    workspace_id: str,
    file_id: str,
    file_path: str,
    document_id: str,
    dataset_id: str,
    task_id: str,
    original_file_deleted: bool = False,
) -> Dict[str, Any]:
    """
    Handle workspace-level video processing with concurrent task coordination.

    Returns:
        Dict containing processing results and segment information
    """
    redis_client = get_redis_client_sync()

    # Workspace-level processing lock
    workspace_key = f"processing:video:workspace:{workspace_id}:file:{file_id}"

    acquired, existing_task = _acquire_processing_lock_atomic(
        redis_client, workspace_key, task_id, expiry_seconds=7200
    )

    try:
        segment_dir = os.path.join(settings.VIDEO_SEGMENTS_DIR, workspace_id, file_id)

        if acquired:
            # We got the lock - check if segments already exist
            if os.path.exists(segment_dir) and _validate_existing_segments(segment_dir):
                logger.info(
                    f"Segments already exist for file {file_id} in workspace {workspace_id}, reusing"
                )
                segments_data = _load_existing_segments(segment_dir)
                segments_created = False
            elif original_file_deleted:
                # Original file was deleted but segments don't exist - this is an error state
                raise Exception(
                    f"Original video file was deleted but no valid segments exist for file {file_id}. "
                    f"Cannot process video without either original file or existing segments."
                )
            else:
                logger.info(
                    f"Creating new segments for file {file_id} in workspace {workspace_id}"
                )
                segments_data = _process_video_and_create_workspace_segments(
                    file_path, segment_dir, document_id, dataset_id
                )
                segments_created = True

                # Delete original video immediately after successful segment creation
                if getattr(settings, "VIDEO_DELETE_ORIGINAL_ENABLED", False):
                    _delete_original_video_file(file_path, file_id)
                    logger.info(
                        f"Deleted original video after successful workspace segment creation: {file_id}"
                    )
        else:
            # Another task is processing - wait and then reuse
            logger.info(
                f"Another task is processing file {file_id}, waiting for completion..."
            )

            if _wait_for_processing_completion(workspace_key, timeout_seconds=7200):
                # Load the segments created by the other task
                if os.path.exists(segment_dir) and _validate_existing_segments(
                    segment_dir
                ):
                    segments_data = _load_existing_segments(segment_dir)
                    segments_created = False
                elif original_file_deleted:
                    # Original file was deleted and no segments after waiting - error state
                    raise Exception(
                        f"Original video file was deleted and no valid segments exist after waiting for processing completion for file {file_id}"
                    )
                else:
                    raise Exception(
                        "Expected segments not found after waiting for processing"
                    )
            else:
                raise Exception("Timeout waiting for video processing completion")

        # Create document chunks from segments (always needed per dataset)
        processed_chunks = _create_document_chunks_from_workspace_segments(
            segments_data, document_id, workspace_id, file_id
        )

        return {
            "success": True,
            "segments_created": segments_created,
            "original_video_deleted": segments_created
            and getattr(settings, "VIDEO_DELETE_ORIGINAL_ENABLED", False),
            "document": segments_data.get("document_data", {}),
            "chunks": processed_chunks,
            "message": "Video processed successfully using workspace segments",
        }

    finally:
        if acquired:
            _cleanup_processing_lock(redis_client, workspace_key, task_id)


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
                f"Successfully acquired video processing lock: {processing_key}"
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
                f"Video processing lock already exists for key {processing_key}, held by task: {existing_task_id}"
            )
            return False, existing_task_id

    except Exception as e:
        logger.error(
            f"Error acquiring video processing lock {processing_key}: {str(e)}"
        )
        # If Redis is unavailable, allow processing to continue
        return True, None


def _estimate_video_processing_tokens(
    file_path: str, metadata: Optional[dict] = None, file_id: Optional[str] = None
) -> int:
    """
    Estimate token count for video processing based on duration and frames.

    Uses research-based estimation for accuracy:
    - Audio transcription: ~160 words/min → ~213 tokens/min
    - Vision processing: ~100 tokens per frame processed
    - Description generation: ~500 tokens total
    """
    try:
        # Check if original file was deleted and we should use alternative estimation
        original_file_deleted = metadata and metadata.get(
            "original_file_deleted", False
        )

        if original_file_deleted:
            logger.info(
                f"Original video file deleted, using alternative token estimation for file {file_id}"
            )
            # Try to estimate from existing segments or stored metadata
            duration_minutes, total_frames = _estimate_from_segments_or_metadata(
                file_id, metadata
            )
        else:
            # Try to get video duration using moviepy for accurate estimation
            try:
                from moviepy.video.io.VideoFileClip import VideoFileClip

                with VideoFileClip(file_path) as video:
                    duration_minutes = video.duration / 60.0
                    # Estimate frames based on segment config
                    config = _get_video_config()
                    segment_length = config.get("video_segment_length", 30)
                    frames_per_segment = config.get("rough_num_frames_per_segment", 5)
                    num_segments = max(1, int(duration_minutes * 60 / segment_length))
                    total_frames = num_segments * frames_per_segment
            except Exception as video_error:
                logger.warning(f"Could not analyze video duration: {video_error}")
                # Fallback to file size estimation (only if file exists)
                try:
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    duration_minutes = max(
                        1, file_size_mb / 10
                    )  # Rough estimate: 10MB per minute
                    total_frames = max(
                        5, int(duration_minutes * 2)
                    )  # Conservative frame estimate
                except (OSError, IOError) as file_error:
                    logger.warning(
                        f"Could not access file for size estimation: {file_error}"
                    )
                    # Use stored file size from metadata if available
                    file_size = metadata.get("file_size", 0) if metadata else 0
                    if file_size > 0:
                        file_size_mb = file_size / (1024 * 1024)
                        duration_minutes = max(1, file_size_mb / 10)
                        total_frames = max(5, int(duration_minutes * 2))
                        logger.info(
                            f"Using stored file size for estimation: {file_size_mb:.1f}MB"
                        )
                    else:
                        # Final fallback - use conservative defaults
                        duration_minutes = 5.0  # Assume 5 minute video
                        total_frames = 50  # Conservative frame estimate
                        logger.warning(
                            "Using conservative default estimates for token calculation"
                        )

        # Token estimation based on research
        # 1. Transcription tokens: 160 words/min → ~213 tokens/min
        transcription_tokens = int(duration_minutes * 213)

        # 2. Vision processing tokens: ~100 tokens per frame
        vision_tokens = total_frames * 100

        # 3. Description generation tokens: ~500 tokens
        description_tokens = 500

        # 4. Safety buffer (20%)
        total_tokens = int(
            (transcription_tokens + vision_tokens + description_tokens) * 1.2
        )

        logger.info(
            f"Estimated video tokens: duration={duration_minutes:.1f}min, "
            f"frames={total_frames}, transcription={transcription_tokens}, "
            f"vision={vision_tokens}, total={total_tokens}"
        )

        return max(200, total_tokens)  # Minimum 200 tokens

    except Exception as e:
        logger.warning(f"Failed to estimate video tokens: {e}")
        return 1500  # Higher conservative default for video processing


def _estimate_from_segments_or_metadata(
    file_id: Optional[str], metadata: Optional[dict]
) -> Tuple[float, int]:
    """
    Estimate video duration and frames from existing segments or metadata.

    Returns:
        Tuple of (duration_minutes, total_frames)
    """
    try:
        if file_id:
            # Try to get workspace_id and check for existing segments
            workspace_id = _get_workspace_id_from_file(file_id)
            if workspace_id:
                segment_dir = os.path.join(
                    settings.VIDEO_SEGMENTS_DIR, workspace_id, file_id
                )
                if os.path.exists(segment_dir):
                    # Load existing segments metadata
                    try:
                        segments_data = _load_existing_segments(segment_dir)
                        if segments_data and "segments" in segments_data:
                            # Calculate duration from segments
                            total_duration = 0
                            segment_count = 0
                            for _segment_name, segment_data in segments_data[
                                "segments"
                            ].items():
                                start_time = segment_data.get("start_time", 0)
                                end_time = segment_data.get("end_time", 0)
                                if end_time > start_time:
                                    total_duration = max(total_duration, end_time)
                                    segment_count += 1

                            if total_duration > 0:
                                duration_minutes = total_duration / 60.0
                                # Estimate frames based on segment count and config
                                config = _get_video_config()
                                frames_per_segment = config.get(
                                    "rough_num_frames_per_segment", 5
                                )
                                total_frames = segment_count * frames_per_segment

                                logger.info(
                                    f"Estimated from existing segments: duration={duration_minutes:.1f}min, "
                                    f"segments={segment_count}, frames={total_frames}"
                                )
                                return duration_minutes, total_frames
                    except Exception as e:
                        logger.warning(
                            f"Could not load existing segments for estimation: {e}"
                        )

        # Fallback to stored file size from metadata
        if metadata:
            file_size = metadata.get("file_size", 0)
            if file_size > 0:
                file_size_mb = file_size / (1024 * 1024)
                duration_minutes = max(1, file_size_mb / 10)  # 10MB per minute estimate
                total_frames = max(
                    5, int(duration_minutes * 2)
                )  # Conservative frame estimate

                logger.info(
                    f"Estimated from stored file size: {file_size_mb:.1f}MB -> "
                    f"duration={duration_minutes:.1f}min, frames={total_frames}"
                )
                return duration_minutes, total_frames

        # Final fallback
        logger.warning("No segments or metadata available, using conservative defaults")
        return 5.0, 50  # 5 minutes, 50 frames

    except Exception as e:
        logger.error(f"Error in segment-based estimation: {e}")
        return 5.0, 50  # Conservative defaults


def _get_video_config() -> Dict[str, Any]:
    """Get video processing configuration from settings"""
    return {
        "video_segment_length": getattr(settings, "VIDEO_SEGMENT_LENGTH", 30),
        "rough_num_frames_per_segment": getattr(
            settings, "VIDEO_FRAMES_PER_SEGMENT", 5
        ),
        "audio_output_format": getattr(settings, "VIDEO_AUDIO_FORMAT", "mp3"),
        "video_output_format": getattr(settings, "VIDEO_OUTPUT_FORMAT", "mp4"),
        "embedding_model": getattr(
            settings, "EMBEDDING_MODEL_NAME", "text-embedding-3-large"
        ),
        "embedding_dim": getattr(settings, "EMBEDDING_DIMENSIONS", 1536),
        "enable_captioning": getattr(settings, "VIDEO_ENABLE_CAPTIONING", True),
        "enable_transcription": getattr(settings, "VIDEO_ENABLE_TRANSCRIPTION", True),
        # Parallel processing configuration
        "transcription_max_workers": getattr(
            settings, "VIDEO_TRANSCRIPTION_MAX_WORKERS", 5
        ),
        "captioning_batch_size": getattr(settings, "VIDEO_CAPTIONING_BATCH_SIZE", 5),
    }


def _validate_video_file(file_path: str, file_id: str) -> Optional[Dict[str, Any]]:
    """Validate video file exists and is accessible"""
    if not os.path.exists(file_path):
        logger.error(f"Video file not found: {file_path}")
        return {
            "file_id": file_id,
            "success": False,
            "status": "exception",
            "error": f"Video file not found: {file_path}",
        }

    # Check if file is actually a video file
    video_extensions = {".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mkv"}
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in video_extensions:
        logger.error(f"File is not a supported video format: {file_path}")
        return {
            "file_id": file_id,
            "success": False,
            "status": "exception",
            "error": f"Unsupported video format: {file_ext}",
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
    """Handle existing document logic"""
    document = (
        db.query(Document)
        .filter(
            Document.file_id == UUID(file_id),
            Document.dataset_id == UUID(dataset_id),
            Document.ingestion_id == UUID(ingestion_id),
        )
        .first()
    )

    if (
        document
        and skip_successful_files
        and document.processing_status == DocumentProcessingStatusEnum.Success
    ):
        logger.info(
            f"Video document {document.id} already successfully processed - skipping"
        )

        # Send success notification
        if user_id:
            propagate_ingestion_status(db, "document", document.id, user_id)

        return {
            "file_id": file_id,
            "document_id": str(document.id),
            "ingestion_id": ingestion_id,
            "success": True,
            "status": "success",
            "skipped": True,
            "message": "Video already successfully processed",
        }

    return None


def _handle_token_acquisition_with_retry(
    file_path: str,
    file_id: str,
    retry_count: int,
    max_retries: int,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    metadata: Optional[dict],
    skip_successful_files: bool,
    retry_reason: Optional[str],
) -> bool:
    """Handle token acquisition with retry logic for video processing"""
    try:
        token_count = _estimate_video_processing_tokens(file_path, metadata, file_id)
        success, remaining_time = acquire_tokens_with_retry(token_count)

        if not success:
            if retry_count < max_retries:
                # Calculate backoff with jitter
                base_delay = remaining_time or settings.RATE_LIMIT_RETRY_SECONDS * (
                    2**retry_count
                )
                max_backoff = 600  # 10 minutes max for video (longer than other files)
                backoff = min(base_delay, max_backoff)
                jitter = (float(secrets.randbelow(20) - 10) / 100.0) * backoff
                countdown = max(1, backoff + jitter)

                logger.info(
                    f"Could not acquire tokens for video {file_id}. "
                    f"Retry {retry_count+1}/{max_retries} scheduled in {countdown:.1f}s"
                )

                celery.signature(
                    "tasks.video_ingestion_task",
                    kwargs={
                        "file_id": file_id,
                        "file_path": file_path,
                        "ingestion_id": ingestion_id,
                        "dataset_id": dataset_id,
                        "user_id": user_id,
                        "metadata": metadata,
                        "skip_successful_files": skip_successful_files,
                        "retry_count": retry_count + 1,
                        "max_retries": max_retries,
                        "retry_reason": "rate_limit",
                    },
                ).apply_async(countdown=countdown)

                return False  # Current task should exit
            else:
                logger.error(
                    f"Failed to acquire tokens for video {file_id} after {max_retries} retries"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"Error acquiring tokens for video {file_id}: {str(e)}")
        return False


def _create_or_update_document(
    db,
    file_id: str,
    dataset_id: str,
    ingestion_id: str,
    file_path: str,
    task_id: str,
    metadata: Optional[dict] = None,
) -> Document:
    """Create or update document record for video"""
    # Try to get existing document
    document = (
        db.query(Document)
        .filter(
            Document.file_id == UUID(file_id),
            Document.dataset_id == UUID(dataset_id),
            Document.ingestion_id == UUID(ingestion_id),
        )
        .first()
    )

    if document:
        # Update existing document
        document.processing_status = DocumentProcessingStatusEnum.Processing
        document.task_id = task_id
        document.updated_at = datetime.now(UTC)

        # Clean existing chunks if reprocessing
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete(
            synchronize_session=False
        )
        logger.info(f"Cleaned existing chunks for video document {document.id}")
    else:
        # Create new document
        from app.models.file_model import File

        file_record = db.query(File).filter(File.id == UUID(file_id)).first()
        if not file_record:
            raise ValueError(f"File record {file_id} not found")

        document = Document(
            file_id=UUID(file_id),
            dataset_id=UUID(dataset_id),
            ingestion_id=UUID(ingestion_id),
            document_type=DocumentTypeEnum.Video,
            processing_status=DocumentProcessingStatusEnum.Processing,
            file_path=file_path,
            file_size=file_record.size,
            mime_type=file_record.mimetype,
            document_metadata=metadata or {},
            task_id=task_id,
        )
        db.add(document)
        logger.info(f"Created new video document record for file {file_id}")

    db.commit()
    db.refresh(document)
    return document


def _process_video_and_create_workspace_segments(
    file_path: str,
    segment_dir: str,
    document_id: str,
    dataset_id: str,
) -> Dict[str, Any]:
    """
    Process video and create segments directly in workspace storage.
    This is the core video processing that happens once per file per workspace.
    """
    try:
        # Create workspace segment directory
        os.makedirs(segment_dir, exist_ok=True)
        logger.info(f"Created workspace segment directory: {segment_dir}")

        # Create temporary working directory for processing
        temp_session_id = str(uuid.uuid4())
        working_dir = os.path.join(
            settings.VIDEO_TEMP_DIR, f"session_{temp_session_id}"
        )
        os.makedirs(working_dir, exist_ok=True)

        try:
            # Process video directly into workspace storage
            workspace_segments_data = _process_video_directly_to_workspace(
                file_path, segment_dir, working_dir
            )

            # Save metadata for future reuse
            _save_segments_metadata(segment_dir, workspace_segments_data)

            return workspace_segments_data

        finally:
            # Clean up temporary working directory
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
                logger.debug(f"Cleaned up temporary working directory: {working_dir}")

    except Exception as e:
        logger.error(f"Error processing video and creating workspace segments: {e}")
        # Clean up partial segment directory on failure
        if os.path.exists(segment_dir):
            try:
                shutil.rmtree(segment_dir)
                logger.info(f"Cleaned up partial segment directory: {segment_dir}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to cleanup partial segment directory: {cleanup_error}"
                )
        raise


def _process_video_directly_to_workspace(
    file_path: str,
    segment_dir: str,
    working_dir: str,
) -> Dict[str, Any]:
    """
    Process video and save segments directly to workspace storage.
    Clean, single-pass processing without intermediate storage.
    """
    try:
        # Change to working directory for temporary files
        original_cwd = os.getcwd()
        os.chdir(working_dir)

        # Set environment variables for temporary processing
        os.environ["IMAGEIO_TEMP_DIR"] = working_dir
        os.environ["MOVIEPY_TEMP_DIR"] = working_dir
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        config = _get_video_config()
        video_name = os.path.basename(file_path).split(".")[0]
        current_time = datetime.now(UTC)

        logger.info(f"Processing video directly to workspace: {file_path}")
        logger.info(f"Workspace segment directory: {segment_dir}")

        # Step 1: Split video into segments
        logger.info("Step 1: Splitting video into segments...")
        segment_index2name, segment_times_info = split_video(
            file_path,
            working_dir,
            config["video_segment_length"],
            config["rough_num_frames_per_segment"],
            config["audio_output_format"],
        )

        # Step 2: Generate transcripts (if enabled)
        transcripts = {}
        if config["enable_transcription"]:
            logger.info("Step 2: Generating transcripts...")
            whisper_model = get_whisper_model()
            transcripts = speech_to_text(
                video_name,
                working_dir,
                segment_index2name,
                config["audio_output_format"],
                whisper_model=whisper_model,
                max_workers=config.get("transcription_max_workers", 4),
            )
        else:
            logger.info("Step 2: Transcription disabled, skipping...")
            for index in segment_index2name:
                transcripts[index] = ""

        # Step 3: Generate captions and save segments directly to workspace
        logger.info("Step 3: Processing captions and saving segments to workspace...")
        import queue
        import threading

        captions = {}
        error_queue = queue.Queue()

        # Initialize captions dict
        for index in segment_index2name:
            captions[index] = ""

        # Thread for saving video segments directly to workspace
        def save_segments_to_workspace():
            try:
                _save_video_segments_to_workspace(
                    video_name,
                    file_path,
                    working_dir,
                    segment_dir,
                    segment_index2name,
                    segment_times_info,
                    error_queue,
                    config["video_output_format"],
                )
            except Exception as e:
                error_queue.put(f"Video saving error: {str(e)}")

        # Thread for generating captions (if enabled)
        def generate_captions():
            try:
                if config["enable_captioning"]:
                    caption_model, caption_tokenizer = get_caption_model()
                    segment_caption(
                        video_name,
                        file_path,
                        segment_index2name,
                        transcripts,
                        segment_times_info,
                        captions,
                        error_queue,
                        caption_model,
                        caption_tokenizer,
                        batch_size=config.get("captioning_batch_size", 3),
                    )
            except Exception as e:
                error_queue.put(f"Caption generation error: {str(e)}")

        # Start threads
        thread_saving = threading.Thread(target=save_segments_to_workspace)
        thread_saving.start()

        thread_caption = None
        if config["enable_captioning"]:
            thread_caption = threading.Thread(target=generate_captions)
            thread_caption.start()

        # Wait for completion
        thread_saving.join()
        if thread_caption:
            thread_caption.join()

        # Check for errors
        if not error_queue.empty():
            error_message = error_queue.get()
            logger.error(f"Error processing video {video_name}: {error_message}")
            raise Exception(error_message)

        # Step 4: Merge segment information
        logger.info("Step 4: Merging segment information...")
        segments_information = merge_segment_information(
            segment_index2name,
            segment_times_info,
            transcripts,
            captions,
        )

        # Step 5: Generate video description
        logger.info("Step 5: Generating video description...")
        video_description = _generate_video_description(segments_information, config)

        # Generate description embedding
        if video_description:
            description_embedding = generate_embedding_with_retry(video_description)
        else:
            logger.warning("No video description found, using empty embeddings")
            description_embedding = [0.0] * settings.EMBEDDING_DIMENSIONS

        # Get file info
        file_size = os.path.getsize(file_path)
        segment_count = len(segments_information)
        total_duration = (
            max(
                segment_data.get("end_time", 0)
                for segment_data in segments_information.values()
            )
            if segments_information
            else 0
        )

        # Create comprehensive workspace segments data
        workspace_segments_data = {
            "processing_completed": True,
            "segment_count": segment_count,
            "total_duration": total_duration,
            "processing_completed_at": current_time.isoformat(),
            "segments_info": segments_information,
            "document_data": {
                "description": video_description,
                "description_embedding": description_embedding,
                "document_metadata": {
                    "video_name": video_name,
                    "file_size": file_size,
                    "video_duration": total_duration,
                    "total_segments": segment_count,
                    "segment_length": config["video_segment_length"],
                    "enable_captioning": config["enable_captioning"],
                    "enable_transcription": config["enable_transcription"],
                    "embedding_model": config["embedding_model"],
                    "description_embedding_dim": len(description_embedding),
                    "processing_completed_at": current_time.isoformat(),
                },
            },
        }

        logger.info(
            f"Successfully processed video with {segment_count} segments directly to workspace"
        )
        return workspace_segments_data

    except Exception as e:
        logger.error(f"Error processing video directly to workspace: {e}")
        raise
    finally:
        # Always restore the original working directory
        try:
            os.chdir(original_cwd)
            logger.debug(f"Restored working directory to: {original_cwd}")
        except Exception as e:
            logger.warning(f"Failed to restore working directory: {e}")

        # Clean up GPU memory after video processing
        try:
            cleanup_video_gpu_memory()
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU memory: {e}")


def _save_video_segments_to_workspace(
    video_name: str,
    video_path: str,
    working_dir: str,
    segment_dir: str,
    segment_index2name: Dict,
    segment_times_info: Dict,
    error_queue,
    video_output_format: str,
) -> None:
    """Save video segments directly to workspace storage"""
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip

        with VideoFileClip(video_path) as video:
            for index_str, segment_name in segment_index2name.items():
                try:
                    start_time, end_time = segment_times_info[index_str]["timestamp"]

                    # Create subclip
                    subvideo = video.subclip(start_time, end_time)

                    # Save directly to workspace storage
                    workspace_segment_path = os.path.join(
                        segment_dir, f"segment_{segment_name}.{video_output_format}"
                    )

                    subvideo.write_videofile(
                        workspace_segment_path,
                        codec="libx264",
                        audio_codec="aac",
                        verbose=False,
                        logger=None,
                    )

                    # Update segment info with workspace path
                    segment_times_info[index_str][
                        "video_segment_path"
                    ] = workspace_segment_path

                    logger.debug(
                        f"Saved segment directly to workspace: {workspace_segment_path}"
                    )

                except Exception as e:
                    error_queue.put(f"Failed to save segment {segment_name}: {str(e)}")
                    raise

    except Exception as e:
        error_queue.put(f"Failed to save video segments to workspace: {str(e)}")
        raise RuntimeError(f"Failed to save video segments to workspace: {str(e)}")


def _create_document_chunks_from_workspace_segments(
    segments_data: Dict[str, Any],
    document_id: str,
    workspace_id: str,
    file_id: str,
) -> List[Dict]:
    """
    Create document chunks from existing workspace segments.
    This runs for each dataset that uses the segments.
    """
    try:
        chunks = []
        segments_info = segments_data.get("segments_info", {})

        for segment_name, segment_data in segments_info.items():
            # Combine caption and transcript
            caption = segment_data.get("caption", "").strip()
            transcript = segment_data.get("transcript", "").strip()

            chunk_text = ""
            if caption and transcript:
                chunk_text = f"Caption: {caption}\nTranscript: {transcript}"
            elif caption:
                chunk_text = f"Caption: {caption}"
            elif transcript:
                chunk_text = f"Transcript: {transcript}"

            # Generate chunk embedding
            if chunk_text:
                chunk_embedding = generate_embedding_with_retry(chunk_text)
            else:
                logger.warning(
                    f"No chunk text found for segment {segment_name}, using empty embeddings"
                )
                chunk_embedding = [0.0] * settings.EMBEDDING_DIMENSIONS

            # Create chunk metadata
            chunk = {
                "chunk_text": chunk_text,
                "chunk_embedding": chunk_embedding,
                "chunk_metadata": {
                    "segment_name": segment_name,
                    "start_time": segment_data.get("start_time", 0),
                    "end_time": segment_data.get("end_time", 0),
                    "duration": segment_data.get("end_time", 0)
                    - segment_data.get("start_time", 0),
                    "caption": caption,
                    "transcript": transcript,
                    "caption_length": len(caption),
                    "transcript_length": len(transcript),
                    "chunk_text_length": len(chunk_text),
                    "embedding_dim": len(chunk_embedding),
                    "frame_count": segment_data.get("frame_count", 0),
                    "frame_times": segment_data.get("frame_times", []),
                    # Workspace segment path
                    "video_segment_path": segment_data.get("video_segment_path", ""),
                    # Document context
                    "document_id": document_id,
                    "workspace_id": workspace_id,
                    "file_id": file_id,
                },
            }

            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} document chunks from workspace segments")
        return chunks

    except Exception as e:
        logger.error(f"Error creating document chunks from workspace segments: {e}")
        raise


def _store_video_chunks(
    db,
    document: Document,
    chunks_data: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Store video chunks in database"""
    chunk_count = 0
    failed_chunks = 0

    for chunk_data in chunks_data:
        try:
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_text=chunk_data["chunk_text"],
                chunk_embedding=chunk_data["chunk_embedding"],
                chunk_metadata=chunk_data["chunk_metadata"],
                chunk_type=ChunkTypeEnum.VideoSegment,  # Use the correct chunk type for video segments
            )
            db.add(chunk)
            chunk_count += 1

        except Exception as e:
            logger.error(f"Failed to store video chunk: {str(e)}")
            failed_chunks += 1
            # Fail the entire video processing if any chunk fails
            if failed_chunks > 0:
                raise Exception(
                    f"Video chunk storage failed: {str(e)} (chunk {failed_chunks}/{len(chunks_data)})"
                )

    try:
        db.commit()
        logger.info(f"Stored {chunk_count} video chunks, {failed_chunks} failed")
        return chunk_count, failed_chunks
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit video chunks: {str(e)}")
        raise


def _update_document_with_results(
    db,
    document: Document,
    video_data: Dict[str, Any],
    chunk_count: int,
) -> None:
    """Update document with processing results"""
    try:
        # Update document with video metadata
        document.description = video_data["description"]
        document.description_embedding = video_data["description_embedding"]
        # Mark as Success - this is the authoritative status setting for video documents
        document.processing_status = DocumentProcessingStatusEnum.Success
        document.processed_at = datetime.now(UTC)

        # Update metadata
        if not document.document_metadata:
            document.document_metadata = {}

        document.document_metadata.update(video_data["document_metadata"])
        document.document_metadata["chunk_count"] = chunk_count
        document.document_metadata["processing_completed_at"] = datetime.now(
            UTC
        ).isoformat()

        attributes.flag_modified(document, "document_metadata")
        db.commit()

        logger.info(f"Updated video document {document.id} with processing results")

    except Exception as e:
        logger.error(f"Failed to update document with results: {str(e)}")
        raise


# Removed _process_segments_for_chunks - replaced with _create_document_chunks_from_workspace_segments


def _generate_video_description(segments_info: Dict, config: Dict[str, Any]) -> str:
    """Generate detailed video description from first 10 segments using Azure OpenAI GPT-4o"""
    try:
        # Get first 10 segments or all if fewer
        segment_names = sorted(segments_info.keys())[:10]

        if not segment_names:
            raise Exception("No segments found")

        # Collect content from segments
        segments_content = []
        for i, segment_name in enumerate(segment_names, 1):
            segment_data = segments_info[segment_name]
            caption = segment_data.get("caption", "").strip()
            transcript = segment_data.get("transcript", "").strip()
            start_time = segment_data.get("start_time", 0)
            end_time = segment_data.get("end_time", 0)

            segment_content = f"Segment {i} ({start_time}s - {end_time}s):"
            if caption:
                segment_content += f"\nVisual Description: {caption}"
            if transcript:
                segment_content += f"\nSpoken Content: {transcript}"

            if caption or transcript:
                segments_content.append(segment_content)

        if not segments_content:
            raise Exception("No segment content generated")

        # Generate Azure OpenAI summary using standardized retry function
        combined_content = "\n\n".join(segments_content)

        # Create detailed summarization prompt
        prompt = f"""Please provide a comprehensive and detailed summary of this video based on the following segments. The summary should capture:

1. The main topic and subject matter
2. Key concepts, ideas, or information presented
3. Visual elements and presentation style
4. Learning objectives or main takeaways
5. Target audience or context
6. Overall structure and flow of the content

Video Segments:
{combined_content}

Please provide a detailed description that would help someone understand what this video is about, what they would learn from it, and how the information is presented. Make it comprehensive enough to serve as a meaningful description for search and categorization purposes."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert video content analyst who creates detailed, comprehensive summaries of educational and informational videos. Your summaries help people understand the full scope and value of video content.",
            },
            {"role": "user", "content": prompt},
        ]

        # Use standardized retry function for chat completion
        summary = chat_completion_with_retry(
            messages=messages,
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.3,
        )

        logger.info(
            f"Generated detailed Azure OpenAI GPT-4o summary ({len(summary)} chars) for video description"
        )
        return f"Detailed Video Analysis (based on {len(segment_names)} segments):\n\n{summary}"

    except Exception as e:
        logger.error(f"Failed to generate Azure OpenAI summary: {e}")
        # Re-raise the exception instead of using fallback
        raise


def _cleanup_video_segments(document_id: str) -> None:
    """
    Clean up video segments if cleanup is enabled.
    Note: With workspace-level segments, document-level cleanup is no longer needed.
    This function is kept for compatibility but does nothing since segments are workspace-level.
    """
    try:
        if getattr(settings, "VIDEO_CLEANUP_ENABLED", False):
            logger.debug(
                f"Video segment cleanup requested for document {document_id}, but segments are workspace-level - no action needed"
            )
        else:
            logger.debug(
                "Video segment cleanup is disabled - preserving workspace segments"
            )
    except Exception as e:
        logger.warning(
            f"Error in video segment cleanup for document {document_id}: {e}"
        )


def _is_processing_error(e: Exception) -> bool:
    """
    Determine if an exception is a processing error (Failed) vs system error (Exception).

    Processing errors (Failed):
    - Video processing failures (transcription, captioning, description generation)
    - Chunk storage failures
    - Content-related errors
    - Business logic failures

    System errors (Exception):
    - Database connection failures
    - Redis connection failures
    - File system errors (permissions, disk space)
    - Infrastructure failures
    """
    error_str = str(e).lower()

    # Processing errors - should be marked as Failed
    processing_error_indicators = [
        "video chunk storage failed",
        "no segments found",
        "no segment content generated",
        "failed to generate azure openai summary",
        "video processing failed",
        "chunk storage failures",
        "transcription",
        "captioning",
        "embedding",
        "description generation",
    ]

    # System errors - should be marked as Exception
    system_error_indicators = [
        "database",
        "connection",
        "redis",
        "permission denied",
        "disk space",
        "no space left",
        "network",
        "timeout",
        "infrastructure",
    ]

    # Check for processing errors first
    for indicator in processing_error_indicators:
        if indicator in error_str:
            return True

    # Check for system errors
    for indicator in system_error_indicators:
        if indicator in error_str:
            return False

    # Default to processing error for video-specific exceptions
    return True


def _handle_video_processing_error(
    db,
    e: Exception,
    document: Optional[Document],
    file_id: str,
    user_id: Optional[UUID],
    status_type: str = "failed",
) -> Dict[str, Any]:
    """Handle exceptions during video processing with proper status updates and notifications"""
    import traceback

    error_msg = f"{str(e)}\n{traceback.format_exc()}"
    logger.error(f"Error processing video document: {error_msg}")

    try:
        if document:
            # Update document status based on error type
            if status_type == "exception":
                document.processing_status = DocumentProcessingStatusEnum.Exception
            else:
                document.processing_status = DocumentProcessingStatusEnum.Failed

            document.error_message = str(e)
            document.processed_at = datetime.now(UTC)
            db.commit()

            # Send failure notification
            if user_id:
                propagate_ingestion_status(db, "document", document.id, user_id)

    except Exception as update_err:
        logger.error(
            f"Error updating document status during error handling: {str(update_err)}"
        )

    return {
        "file_id": file_id,
        "document_id": str(document.id) if document else None,
        "success": False,
        "error": str(e),
        "status": status_type,
        "document_type": "Video",
    }


def _cleanup_stale_temp_sessions(max_age_hours: int = 2) -> None:
    """Clean up stale temporary video processing sessions"""
    try:
        if not os.path.exists(settings.VIDEO_TEMP_DIR):
            return

        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        for session_dir in os.listdir(settings.VIDEO_TEMP_DIR):
            session_path = os.path.join(settings.VIDEO_TEMP_DIR, session_dir)
            if os.path.isdir(session_path) and session_dir.startswith("session_"):
                try:
                    # Check directory creation time
                    dir_ctime = os.path.getctime(session_path)
                    if dir_ctime < cutoff_time:
                        shutil.rmtree(session_path)
                        logger.debug(f"Cleaned up stale temp session: {session_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to cleanup stale session {session_path}: {e}"
                    )

        logger.info(
            f"Completed cleanup of stale temp sessions older than {max_age_hours} hours"
        )
    except Exception as e:
        logger.warning(f"Failed to cleanup stale temp sessions: {e}")


def _delete_original_video_file(file_path: str, file_id: str) -> None:
    """
    Safely delete the original video file with proper error handling.
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Original video file not found for deletion: {file_path}")
            return

        # Get file size for logging
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        # Delete the file
        os.remove(file_path)

        logger.info(
            f"Successfully deleted original video file: {file_path} "
            f"(file_id: {file_id}, size: {file_size_mb:.1f}MB)"
        )

    except PermissionError as e:
        logger.error(f"Permission denied when deleting original video {file_path}: {e}")
    except OSError as e:
        logger.error(f"OS error when deleting original video {file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error when deleting original video {file_path}: {e}")


@celery.task(name="tasks.video_ingestion_task", bind=True, acks_late=True)
def video_ingestion_task(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: UUID,
    metadata: Optional[dict] = None,
    skip_successful_files: bool = True,
    retry_count: int = 0,
    max_retries: int = 5,  # Lower max retries for video due to resource intensity
    retry_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Video ingestion task that processes a single video file.
    Supports video segmentation, transcription, and captioning.

    Args:
        file_id: ID of the file
        file_path: Path to the video file
        ingestion_id: ID of the ingestion process
        dataset_id: ID of the dataset
        user_id: ID of the user who initiated the ingestion
        metadata: Optional metadata for the file
        skip_successful_files: Whether to skip files that are already successfully ingested
        retry_count: Current retry attempt count for token acquisition
        max_retries: Maximum number of retries before giving up
        retry_reason: Reason for retry (e.g., "timeout", "rate_limit")

    Returns:
        Dictionary with results of the ingestion process
    """
    logger.info(
        f"Starting video ingestion task for file: {file_path} "
        f"(retry #{retry_count}, reason: {retry_reason or 'initial'})"
    )

    # Early exit if video ingestion is disabled
    if not is_video_ingestion_enabled():
        logger.warning("Video ingestion is disabled via settings or feature flag")
        return {
            "file_id": file_id,
            "success": False,
            "error": "Video ingestion is disabled",
            "status": "failed",
            "document_type": "Video",
        }

    # Robust task-level deduplication
    processing_key = None
    redis_client = None

    try:
        redis_client = get_redis_client_sync()

        # Create unique processing key
        processing_key = (
            f"processing:video:{file_id}:ingestion:{ingestion_id}:dataset:{dataset_id}"
        )

        # Try to atomically acquire processing lock
        acquired, existing_task_id = _acquire_processing_lock_atomic(
            redis_client,
            processing_key,
            self.request.id,
            expiry_seconds=7200,  # 2 hours for video
        )

        if not acquired:
            # Check if original task is still alive
            if existing_task_id and not _handle_stale_lock(
                redis_client, processing_key, existing_task_id, self.request.id
            ):
                logger.info(
                    f"Video file {file_id} is being processed by active task {existing_task_id} - skipping"
                )
                redis_client.close()

                return {
                    "file_id": file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "already_processing",
                    "processing_task_id": existing_task_id,
                    "document_type": "Video",
                }

        logger.info(f"Acquired video processing lock: {processing_key}")

    except Exception as e:
        # Don't fail the task if Redis is unavailable, just log it
        logger.warning(f"Redis deduplication not available: {str(e)}")
        processing_key = None
        redis_client = None

    document_id = None

    try:
        # Step 1: Check if original file was deleted and we should use existing segments
        original_file_deleted = metadata and metadata.get(
            "original_file_deleted", False
        )

        if original_file_deleted:
            logger.info(
                f"Original video file was deleted, attempting to use existing segments for file {file_id}"
            )
            # Skip file validation since original file doesn't exist
            # We'll rely on existing segments for processing
        else:
            # Step 1: Validate video file (only if original file should exist)
            error_response = _validate_video_file(file_path, file_id)
            if error_response:
                return error_response

        # Step 2: Handle token acquisition with retry logic
        if not _handle_token_acquisition_with_retry(
            file_path,
            file_id,
            retry_count,
            max_retries,
            ingestion_id,
            dataset_id,
            user_id,
            metadata,
            skip_successful_files,
            retry_reason,
        ):
            return {
                "file_id": file_id,
                "success": False,
                "error": "Failed to acquire processing tokens",
                "status": "failed",
                "document_type": "Video",
            }

        # Step 3: Handle database operations
        with SyncSessionLocal() as db:
            # Check if we should skip this file
            skip_response = _handle_existing_document(
                db, file_id, dataset_id, ingestion_id, skip_successful_files, user_id
            )
            if skip_response:
                return skip_response

            # Create or update document record
            document = _create_or_update_document(
                db,
                file_id,
                dataset_id,
                ingestion_id,
                file_path,
                self.request.id,
                metadata,
            )
            document_id = str(document.id)

            # Check if document was failed due to timeout - skip processing BEFORE token acquisition
            logger.info(f"CHECKING if document {document.id} is failed due to timeout")
            should_skip, skip_reason = should_skip_processing_due_to_timeout(
                document.id
            )
            if should_skip:
                logger.warning(
                    f"⏰ SKIPPING video processing for {file_id}: {skip_reason}"
                )
                return {
                    "file_id": file_id,
                    "success": True,
                    "skipped": True,
                    "reason": "document_timeout_failed",
                    "message": skip_reason,
                    "document_id": document_id,
                    "document_type": "Video",
                }

        # Step 4: Ensure video temp directory exists
        try:
            os.makedirs(settings.VIDEO_TEMP_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create video temp directory: {e}")
            return {
                "file_id": file_id,
                "success": False,
                "error": "Failed to create video temp directory",
                "status": "exception",
                "document_type": "Video",
            }

        # Step 5: Process video content using workspace-level segments
        workspace_id = _get_workspace_id_from_file(file_id)
        if not workspace_id:
            raise ValueError(f"Could not determine workspace_id for file {file_id}")

        processing_result = _handle_workspace_video_processing(
            workspace_id=workspace_id,
            file_id=file_id,
            file_path=file_path,
            document_id=document_id,
            dataset_id=dataset_id,
            task_id=self.request.id,
            original_file_deleted=original_file_deleted,
        )

        # Step 6: Store chunks and update document
        with SyncSessionLocal() as db:
            document = (
                db.query(Document).filter(Document.id == UUID(document_id)).first()
            )
            if not document:
                raise ValueError(f"Document {document_id} not found after processing")

            # Store video chunks
            chunk_count, failed_chunks = _store_video_chunks(
                db, document, processing_result["chunks"]
            )

            # Update document with results
            _update_document_with_results(
                db, document, processing_result["document"], chunk_count
            )

            # Send success notification
            if user_id:
                propagate_ingestion_status(db, "document", document.id, user_id)

        logger.info(
            f"Successfully completed video processing for file {file_id}: "
            f"document_id={document_id}, chunks={chunk_count}"
        )

        # Include information about workspace processing
        segments_created = processing_result.get("segments_created", False)
        original_video_deleted = processing_result.get("original_video_deleted", False)

        message = f"Video processed successfully with {chunk_count} segments"
        if segments_created:
            message += " (new segments created)"
            if original_video_deleted:
                message += " - original video deleted"
        else:
            message += " (reused existing segments)"

        return {
            "file_id": file_id,
            "document_id": document_id,
            "ingestion_id": ingestion_id,
            "success": True,
            "status": "success",
            "chunk_count": chunk_count,
            "failed_chunks": failed_chunks,
            "document_type": "Video",
            "processing_type": "workspace_segments",
            "segments_created": segments_created,
            "original_video_deleted": original_video_deleted,
            "message": message,
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)

        # Distinguish between processing errors (Failed) and system errors (Exception)
        with SyncSessionLocal() as db:
            document = None
            if document_id:
                document = (
                    db.query(Document).filter(Document.id == UUID(document_id)).first()
                )

            # Check if this is a processing error or system error
            if _is_processing_error(e):
                return _handle_video_processing_error(
                    db=db,
                    e=e,
                    document=document,
                    file_id=file_id,
                    user_id=user_id,
                    status_type="failed",
                )
            else:
                return _handle_video_processing_error(
                    db=db,
                    e=e,
                    document=document,
                    file_id=file_id,
                    user_id=user_id,
                    status_type="exception",
                )

    finally:
        # Clean up video segments if cleanup is enabled and processing completed
        if document_id and getattr(settings, "VIDEO_CLEANUP_ENABLED", False):
            try:
                logger.info(
                    f"Cleaning up video segments for document {document_id} (cleanup enabled)"
                )
                _cleanup_video_segments(document_id)
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup video segments for document {document_id}: {e}"
                )

        # Clean up stale temp sessions (maintenance cleanup)
        try:
            _cleanup_stale_temp_sessions(max_age_hours=2)
        except Exception as e:
            logger.warning(f"Failed to cleanup stale temp sessions: {e}")

        # Clean up processing deduplication flag (only if we own it)
        if processing_key and redis_client:
            try:
                _cleanup_processing_lock(redis_client, processing_key, self.request.id)
            except Exception as e:
                logger.warning(f"Failed to clean up video processing lock: {str(e)}")

        # Close Redis connection
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {str(e)}")
