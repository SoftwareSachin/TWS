"""
Utility functions for cleaning up video segments when video files are deleted.
"""

import os
import shutil
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.be_core.config import settings
from app.be_core.logger import logger

# Simple in-memory cache for reference checks (TTL: 5 minutes)
_reference_cache = {}
_cache_ttl = 300  # 5 minutes


def is_video_file(filename: str, mimetype: Optional[str] = None) -> bool:
    """
    Check if a file is a video file based on filename and mimetype.

    Args:
        filename: The filename to check
        mimetype: Optional mimetype to check

    Returns:
        bool: True if the file is a video file, False otherwise
    """
    # Video extensions supported by the system
    video_extensions = {".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mkv"}

    # Video MIME types supported by the system
    video_mimetypes = {
        "video/mp4",
        "video/x-msvideo",  # AVI
        "video/quicktime",  # MOV
        "video/x-ms-wmv",  # WMV
        "video/x-flv",  # FLV
        "video/webm",  # WebM
        "video/x-matroska",  # MKV
    }

    # Check by mimetype first if available
    if mimetype and mimetype in video_mimetypes:
        return True

    # Check by file extension
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        return ext in video_extensions

    return False


def clear_reference_cache(workspace_id: UUID = None, file_id: UUID = None) -> None:
    """
    Clear the reference check cache.

    Args:
        workspace_id: If provided, clear cache for specific workspace
        file_id: If provided (with workspace_id), clear cache for specific file
    """
    global _reference_cache

    if workspace_id and file_id:
        # Clear specific file
        cache_key = f"{workspace_id}:{file_id}"
        _reference_cache.pop(cache_key, None)
        logger.debug(f"Cleared reference cache for file {file_id}")
    elif workspace_id:
        # Clear all files in workspace
        workspace_prefix = f"{workspace_id}:"
        keys_to_remove = [
            key for key in _reference_cache.keys() if key.startswith(workspace_prefix)
        ]
        for key in keys_to_remove:
            _reference_cache.pop(key, None)
        logger.debug(
            f"Cleared reference cache for workspace {workspace_id} ({len(keys_to_remove)} entries)"
        )
    else:
        # Clear entire cache
        cache_size = len(_reference_cache)
        _reference_cache.clear()
        logger.debug(f"Cleared entire reference cache ({cache_size} entries)")


async def check_video_segments_still_referenced(
    workspace_id: UUID, file_id: UUID, db_session: AsyncSession = None
) -> bool:
    """
    Check if video segments are still referenced by any documents in the workspace.

    Args:
        workspace_id: The workspace UUID
        file_id: The file UUID
        db_session: Optional database session to use

    Returns:
        bool: True if segments are still referenced, False if safe to delete
    """
    import time

    # Check cache first (for performance with frequent checks)
    cache_key = f"{workspace_id}:{file_id}"
    current_time = time.time()

    if cache_key in _reference_cache:
        cached_result, cached_time = _reference_cache[cache_key]
        if current_time - cached_time < _cache_ttl:
            logger.debug(
                f"Using cached reference check result for file {file_id}: {cached_result}"
            )
            return cached_result

    try:
        from sqlalchemy import select

        from app.db.session import SessionLocal
        from app.models.document_chunk_model import DocumentChunk
        from app.models.document_model import Document

        # Use provided session or create a new one
        if db_session:
            session = db_session
            close_session = False
        else:
            session = SessionLocal()
            close_session = True

        try:
            # Optimized approach: Single query with JSON search
            # This is much faster than iterating through all documents and chunks
            from sqlalchemy import text

            try:
                # Primary approach: Use PostgreSQL JSON operators for efficient search
                # This searches within the JSON metadata for file_id field
                chunks_query = (
                    select(
                        DocumentChunk.id, DocumentChunk.document_id, Document.dataset_id
                    )
                    .join(Document)
                    .where(
                        Document.file_id == file_id,
                        Document.deleted_at.is_(None),
                        # Check if chunk_metadata JSON has file_id field matching our file_id
                        # Use PostgreSQL's ->> operator for JSON text extraction
                        text("chunk_metadata->>'file_id' = :file_id_param"),
                    )
                    .params(file_id_param=str(file_id))
                    .limit(1)
                )  # We only need to know if ANY exist

                chunks_result = await session.execute(chunks_query)
                referenced_chunk = chunks_result.first()

            except Exception as db_error:
                logger.warning(
                    f"JSON search failed, falling back to document-based search: {db_error}"
                )

                # Fallback approach: Check if any documents exist for this file
                # This is less precise but much faster than the original nested loop approach
                documents_query = (
                    select(Document.id.label("document_id"), Document.dataset_id)
                    .where(Document.file_id == file_id, Document.deleted_at.is_(None))
                    .limit(1)
                )

                documents_result = await session.execute(documents_query)
                referenced_chunk = documents_result.first()

            if referenced_chunk:
                logger.info(
                    f"Video segments still referenced by document {referenced_chunk.document_id} "
                    f"in dataset {referenced_chunk.dataset_id}"
                )
                result = True
                # Cache the result for future checks
                _reference_cache[cache_key] = (result, current_time)
                return result

            logger.info(f"No active references to video segments for file {file_id}")
            result = False

            # Cache the result for future checks
            _reference_cache[cache_key] = (result, current_time)
            return result

        finally:
            if close_session and hasattr(session, "close"):
                await session.close()

    except Exception as e:
        logger.error(
            f"Error checking video segment references for file {file_id}: {e}",
            exc_info=True,
        )
        # If we can't check references, err on the side of caution and don't delete
        return True


async def delete_video_segments(
    workspace_id: UUID,
    file_id: UUID,
    force: bool = False,
    db_session: AsyncSession = None,
) -> bool:
    """
    Delete video segments directory for a specific file in a workspace.
    Only deletes if no active references exist or if forced.

    Args:
        workspace_id: The workspace UUID
        file_id: The file UUID
        force: If True, delete segments even if references exist (use with caution)
        db_session: Optional database session to use for reference checking

    Returns:
        bool: True if segments were deleted or didn't exist, False if deletion failed or skipped
    """
    try:
        # Construct the path to the video segments directory
        segment_dir = os.path.join(
            settings.VIDEO_SEGMENTS_DIR, str(workspace_id), str(file_id)
        )

        # Check if the segments directory exists
        if not os.path.exists(segment_dir):
            logger.info(f"Video segments directory does not exist: {segment_dir}")
            return True

        # Check if segments are still referenced unless forced
        if not force:
            if await check_video_segments_still_referenced(
                workspace_id, file_id, db_session
            ):
                logger.info(
                    f"Skipping video segments deletion for file {file_id} - "
                    f"segments are still referenced by active documents"
                )
                return False

        # Get directory size for logging
        total_size = 0
        file_count = 0
        try:
            for dirpath, _dirnames, filenames in os.walk(segment_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                        file_count += 1
        except (OSError, IOError) as e:
            logger.warning(
                f"Could not calculate size of segments directory {segment_dir}: {e}"
            )

        # Delete the entire segments directory
        shutil.rmtree(segment_dir)

        size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        action = "Force deleted" if force else "Deleted"
        logger.info(
            f"{action} video segments directory: {segment_dir} "
            f"(workspace_id: {workspace_id}, file_id: {file_id}, "
            f"files: {file_count}, size: {size_mb:.1f}MB)"
        )

        return True

    except PermissionError as e:
        logger.error(
            f"Permission denied when deleting video segments directory {segment_dir}: {e}"
        )
        return False
    except OSError as e:
        logger.error(
            f"OS error when deleting video segments directory {segment_dir}: {e}"
        )
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error when deleting video segments directory {segment_dir}: {e}",
            exc_info=True,
        )
        return False


async def cleanup_orphaned_video_segments(
    workspace_id: UUID, db_session: AsyncSession = None
) -> int:
    """
    Clean up orphaned video segments in a workspace following the specific logic:
    - video present && active reference → keep segment
    - video deleted && active reference → keep segment
    - video deleted && no active reference → delete segments
    - video present && no active reference → keep segments

    Args:
        workspace_id: The workspace UUID
        db_session: Optional database session to use for reference checking

    Returns:
        int: Number of orphaned segment directories cleaned up
    """
    try:
        workspace_video_dir = os.path.join(
            settings.VIDEO_SEGMENTS_DIR, str(workspace_id)
        )

        if not os.path.exists(workspace_video_dir):
            logger.info(
                f"Workspace video directory does not exist: {workspace_video_dir}"
            )
            return 0

        cleaned_count = 0

        # Check each file_id directory in the workspace
        for item in os.listdir(workspace_video_dir):
            item_path = os.path.join(workspace_video_dir, item)

            # Only process directories that look like UUIDs (file_ids)
            if os.path.isdir(item_path) and len(item) == 36:  # UUID length
                try:
                    file_id = UUID(item)

                    # Check video file status and active references using the correct logic
                    should_delete = await _should_delete_video_segments(
                        workspace_id, file_id, db_session
                    )

                    if should_delete:
                        # Safe to delete - video deleted AND no active references
                        if await delete_video_segments(
                            workspace_id, file_id, force=True, db_session=db_session
                        ):
                            cleaned_count += 1
                            logger.info(
                                f"Cleaned up orphaned segments for file {file_id} (video deleted, no references)"
                            )
                        else:
                            logger.warning(
                                f"Failed to clean up orphaned segments for file {file_id}"
                            )
                    else:
                        logger.debug(
                            f"Keeping segments for file {file_id} (video present OR has references)"
                        )

                except ValueError:
                    # Not a valid UUID, skip
                    logger.debug(f"Skipping non-UUID directory: {item}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing segments for file {item}: {e}")
                    continue

        logger.info(
            f"Cleaned up {cleaned_count} orphaned video segment directories in workspace {workspace_id}"
        )
        return cleaned_count

    except Exception as e:
        logger.error(
            f"Error during orphaned video segments cleanup for workspace {workspace_id}: {e}",
            exc_info=True,
        )
        return 0


async def _should_delete_video_segments(
    workspace_id: UUID, file_id: UUID, db_session: AsyncSession = None
) -> bool:
    """
    Determine if video segments should be deleted based on the specific logic:
    - video present && active reference → keep segment (False)
    - video deleted && active reference → keep segment (False)
    - video deleted && no active reference → delete segments (True)
    - video present && no active reference → keep segments (False)

    Returns:
        bool: True if segments should be deleted, False if they should be kept
    """
    try:
        from sqlalchemy import select

        from app.db.session import SessionLocal
        from app.models.file_model import File

        # Use provided session or create a new one
        if db_session:
            session = db_session
            close_session = False
        else:
            session = SessionLocal()
            close_session = True

        try:
            # Check if video file is present (not deleted)
            file_query = select(File).where(
                File.id == file_id, File.workspace_id == workspace_id
            )
            file_result = await session.execute(file_query)
            file_record = file_result.scalars().first()

            # Determine video file status
            video_present = file_record is not None and file_record.deleted_at is None

            # Check for active document references
            has_active_references = await check_video_segments_still_referenced(
                workspace_id, file_id, session
            )

            # Apply the logic
            if video_present and has_active_references:
                logger.debug(
                    f"File {file_id}: video present && active reference → keep"
                )
                return False
            elif not video_present and has_active_references:
                logger.debug(
                    f"File {file_id}: video deleted && active reference → keep"
                )
                return False
            elif not video_present and not has_active_references:
                logger.debug(
                    f"File {file_id}: video deleted && no active reference → delete"
                )
                return True
            elif video_present and not has_active_references:
                logger.debug(
                    f"File {file_id}: video present && no active reference → keep"
                )
                return False

            return False  # Default to keep

        finally:
            if close_session and hasattr(session, "close"):
                await session.close()

    except Exception as e:
        logger.error(
            f"Error determining if video segments should be deleted for file {file_id}: {e}",
            exc_info=True,
        )
        return False  # Default to keep on error


def cleanup_empty_workspace_video_dir(workspace_id: UUID) -> None:
    """
    Clean up empty workspace video directory if no more video segments exist.

    Args:
        workspace_id: The workspace UUID
    """
    try:
        workspace_video_dir = os.path.join(
            settings.VIDEO_SEGMENTS_DIR, str(workspace_id)
        )

        # Check if workspace video directory exists and is empty
        if os.path.exists(workspace_video_dir) and not os.listdir(workspace_video_dir):
            os.rmdir(workspace_video_dir)
            logger.info(
                f"Cleaned up empty workspace video directory: {workspace_video_dir}"
            )

    except Exception as e:
        logger.warning(
            f"Failed to cleanup empty workspace video directory for {workspace_id}: {e}"
        )
