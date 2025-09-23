import asyncio
from datetime import datetime
from functools import wraps
from typing import List, Optional, Set
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.models.dataset_file_link_model import DatasetFileLink
from app.models.dataset_model import Dataset
from app.models.document_chunk_model import DocumentChunk
from app.models.document_model import Document
from app.models.file_model import File
from app.models.file_split_model import FileSplit
from app.models.source_model import Source
from app.schemas.dataset_schema import IDatasetCreate, IDatasetRead, IDatasetUpdate

# Configuration constants for batch processing
try:
    from app.be_core.config import settings

    DEFAULT_BATCH_SIZE = getattr(
        settings, "DATABASE_BATCH_SIZE", 1000
    )  # Default batch size for file operations
    MAX_BATCH_SIZE = getattr(
        settings, "DATABASE_MAX_BATCH_SIZE", 5000
    )  # Maximum batch size to prevent memory issues
except ImportError:
    DEFAULT_BATCH_SIZE = 1000  # Default batch size for file operations
    MAX_BATCH_SIZE = 5000  # Maximum batch size to prevent memory issues


def retry_on_deadlock(max_retries: int = 3, delay: float = 0.1):
    """Decorator to retry database operations on deadlock."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "deadlock" in error_msg and attempt < max_retries - 1:
                        wait_time = delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Deadlock detected in {func.__name__}, attempt {attempt + 1}/{max_retries}, retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise
            return await func(*args, **kwargs)

        return wrapper

    return decorator


async def execute_with_transaction_retry(func, *args, **kwargs):
    """Execute a function with automatic transaction retry on deadlock."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "deadlock" in error_msg and attempt < max_retries - 1:
                wait_time = 0.1 * (2**attempt)
                logger.warning(
                    f"Deadlock in transaction, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                raise


class CRUDDatasetV2(CRUDBase[Dataset, IDatasetCreate, IDatasetUpdate]):
    async def create_dataset_v2(
        self,
        *,
        obj_in: IDatasetCreate,
        organization_id: Optional[UUID] = None,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IDatasetRead:
        db_session = db_session or super().get_db().session
        existing_query = select(Dataset).where(
            func.lower(func.trim(Dataset.name)) == func.lower(func.trim(obj_in.name)),
            Dataset.workspace_id == workspace_id,
            Dataset.deleted_at.is_(None),
        )
        existing_result = await db_session.execute(existing_query)
        existing_dataset = existing_result.scalars().first()
        if existing_dataset:
            raise HTTPException(
                status_code=400,
                detail=f"A dataset with name '{obj_in.name}' already exists in this workspace.",
            )
        # Check if file_ids is a non-empty list and source_id is also provided
        if obj_in.file_ids and obj_in.file_ids != [] and obj_in.source_id is not None:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both non-empty file_ids and source_id. Please provide only one of them.",
            )

        if obj_in.source_id:
            # Validate source if provided
            await self._validate_source_v2(obj_in.source_id, db_session)

        # Create dataset record
        dataset_record = Dataset(
            name=obj_in.name,
            description=obj_in.description,
            workspace_id=workspace_id,
            source_id=obj_in.source_id,
        )
        db_session.add(dataset_record)
        await db_session.commit()
        await db_session.refresh(dataset_record)

        # Link manually specified files if any
        if obj_in.file_ids:
            await self._link_files_v2(dataset_record.id, obj_in.file_ids, db_session)

        # Link all files from source if source_id is provided
        elif obj_in.source_id:
            source_files = await self._get_source_files_v2(obj_in.source_id, db_session)
            await self._link_files_v2(dataset_record.id, source_files, db_session)

        # calling celery task here for pusing all groove data to mount:
        source_type = None
        if obj_in.source_id:
            source_result = await db_session.execute(
                select(Source.source_type).where(Source.id == obj_in.source_id)
            )
            source_type = source_result.scalar_one_or_none()
            logger.info(
                f"Fetched source_type: {source_type} for source_id: {obj_in.source_id}"
            )
        logger.debug(f"No action taken for source_type: {source_type}")

        # Create default configurations
        await self._create_default_configs_v2(dataset_record.id)

        # Get source_type if source_id is provided
        source_type = None
        if obj_in.source_id:
            source_result = await db_session.execute(
                select(Source.source_type).where(Source.id == obj_in.source_id)
            )
            source_type = source_result.scalar_one_or_none()

        # Prepare and return response
        dataset_data = IDatasetRead(
            id=dataset_record.id,
            name=dataset_record.name,
            description=dataset_record.description,
            file_ids=obj_in.file_ids,
            source_id=obj_in.source_id if obj_in.source_id else None,
            source_type=source_type,
            knowledge_graph=dataset_record.knowledge_graph,
            graph_build_phase=dataset_record.graph_build_phase,
            graph_build_requested_at=dataset_record.graph_build_requested_at,
            graph_build_completed_at=dataset_record.graph_build_completed_at,
            last_extraction_check_at=dataset_record.last_extraction_check_at,
        )
        return dataset_data

    async def update_dataset_v2(
        self,
        *,
        obj_in: IDatasetUpdate,
        dataset_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IDatasetRead:
        """
        Update dataset without R2R operations.
        Updates basic fields and handles file/source changes using Document model.
        """
        db_session = db_session or super().get_db().session

        # Get the dataset record
        dataset_query = select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.workspace_id == workspace_id,
            Dataset.deleted_at.is_(None),
        )
        result = await db_session.execute(dataset_query)
        dataset_record = result.scalars().first()

        if not dataset_record:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Validate inputs
        await self._validate_update_inputs_v2(obj_in, db_session)

        # Check for name uniqueness if name is being updated
        if obj_in.name and obj_in.name != dataset_record.name:
            name_conflict_query = select(Dataset).where(
                func.lower(func.trim(Dataset.name))
                == func.lower(func.trim(obj_in.name)),
                Dataset.workspace_id == workspace_id,
                Dataset.deleted_at.is_(None),
                Dataset.id != dataset_id,
            )
            existing_result = await db_session.execute(name_conflict_query)
            conflict_dataset = existing_result.scalars().first()
            if conflict_dataset:
                raise HTTPException(
                    status_code=400,
                    detail=f"A dataset with name '{obj_in.name}' already exists in this workspace.",
                )

        # Update basic fields
        if obj_in.name is not None:
            dataset_record.name = obj_in.name
        if obj_in.description is not None:
            dataset_record.description = obj_in.description

        # Handle source and file changes
        await self._handle_source_and_file_changes_v2(
            dataset_record, obj_in, db_session
        )

        # Set updated timestamp
        dataset_record.updated_at = datetime.utcnow()

        # Commit changes
        await db_session.commit()
        await db_session.refresh(dataset_record)

        # Get current file IDs for response
        file_ids = await self._get_dataset_file_ids_v2(dataset_id, db_session)

        # Get source_type if source_id exists
        source_type = None
        if dataset_record.source_id:
            source_result = await db_session.execute(
                select(Source.source_type).where(Source.id == dataset_record.source_id)
            )
            source_type = source_result.scalar_one_or_none()

        logger.info(f"Dataset {dataset_record.name} updated successfully (V2)")

        # Return response
        return IDatasetRead(
            id=dataset_record.id,
            name=dataset_record.name,
            description=dataset_record.description,
            file_ids=file_ids,
            source_id=dataset_record.source_id,
            source_type=source_type,
            r2r_collection_id=dataset_record.r2r_collection_id,
            knowledge_graph=dataset_record.knowledge_graph,
            graph_build_phase=dataset_record.graph_build_phase,
            graph_build_requested_at=dataset_record.graph_build_requested_at,
            graph_build_completed_at=dataset_record.graph_build_completed_at,
            last_extraction_check_at=dataset_record.last_extraction_check_at,
        )

    @retry_on_deadlock(max_retries=3, delay=0.1)
    async def delete_dataset_v2(
        self,
        *,
        dataset_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IDatasetRead:
        """
        Delete dataset with batch processing for better performance.
        Soft deletes dataset, documents, file splits and document chunks in batches.
        """
        db_session = db_session or super().get_db().session

        # Get the dataset record
        dataset_query = select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.workspace_id == workspace_id,
            Dataset.deleted_at.is_(None),
        )
        result = await db_session.execute(dataset_query)
        dataset_record = result.scalars().first()

        if not dataset_record:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Get file IDs before deletion for response
        file_ids = await self._get_dataset_file_ids_v2(dataset_id, db_session)

        # Get source_type if source_id exists
        source_type = None
        if dataset_record.source_id:
            source_result = await db_session.execute(
                select(Source.source_type).where(Source.id == dataset_record.source_id)
            )
            source_type = source_result.scalar_one_or_none()

        # Create response data before deletion
        response_data = IDatasetRead(
            id=dataset_record.id,
            name=dataset_record.name,
            description=dataset_record.description,
            file_ids=file_ids,
            source_id=dataset_record.source_id,
            source_type=source_type,
            r2r_collection_id=dataset_record.r2r_collection_id,
            knowledge_graph=dataset_record.knowledge_graph,
            graph_build_phase=dataset_record.graph_build_phase,
            graph_build_requested_at=dataset_record.graph_build_requested_at,
            graph_build_completed_at=dataset_record.graph_build_completed_at,
            last_extraction_check_at=dataset_record.last_extraction_check_at,
        )

        logger.info(
            f"Starting batch deletion of dataset {dataset_record.name} with {len(file_ids)} files"
        )

        # Perform batch deletion operations
        await self._batch_delete_dataset_content(dataset_id, db_session)

        # Clean up orphaned video segments after dataset deletion
        await self._cleanup_orphaned_video_segments_after_dataset_deletion(
            workspace_id, db_session
        )

        # Soft delete the dataset itself
        dataset_record.deleted_at = datetime.utcnow()
        await db_session.commit()

        logger.info(f"Dataset {dataset_record.name} deleted successfully (V2)")

        return response_data

    async def _cleanup_orphaned_video_segments_after_dataset_deletion(
        self, workspace_id: UUID, db_session: AsyncSession
    ) -> None:
        """
        Clean up orphaned video segments after dataset deletion.
        """
        try:
            from app.utils.video_cleanup_utils import (
                cleanup_empty_workspace_video_dir,
                cleanup_orphaned_video_segments,
                clear_reference_cache,
            )

            logger.info(
                f"Checking for orphaned video segments in workspace {workspace_id}"
            )

            # Clear the reference cache before checking for orphaned segments
            # This ensures we don't use stale cached results after deleting dataset content
            clear_reference_cache(workspace_id=workspace_id)
            logger.debug(f"Cleared reference cache for workspace {workspace_id}")

            cleaned_count = await cleanup_orphaned_video_segments(
                workspace_id, db_session=db_session
            )

            if cleaned_count > 0:
                logger.info(
                    f"Cleaned up {cleaned_count} orphaned video segment directories after dataset deletion"
                )

                # Clean up empty workspace video directory if needed
                cleanup_empty_workspace_video_dir(workspace_id)
            else:
                logger.debug("No orphaned video segments found to clean up")

        except Exception as e:
            # Don't fail dataset deletion if video cleanup fails
            logger.error(
                f"Error cleaning up video segments after dataset deletion: {e}",
                exc_info=True,
            )

    @retry_on_deadlock(max_retries=3, delay=0.1)
    async def _batch_delete_dataset_content(
        self, dataset_id: UUID, db_session: AsyncSession
    ) -> None:
        """
        Batch delete all content related to a dataset.
        Uses bulk operations for better performance with large datasets.
        """

        # Batch size for deletion operations
        batch_size = min(DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE)
        current_time = datetime.utcnow()

        # Step 1: Get all document IDs for this dataset (single query)
        documents_query = select(Document.id).where(
            Document.dataset_id == dataset_id,
            Document.deleted_at.is_(None),
        )
        documents_result = await db_session.execute(documents_query)
        document_ids = list(documents_result.scalars().all())

        if not document_ids:
            logger.info(f"No documents found for dataset {dataset_id}")
            return

        logger.info(
            f"Found {len(document_ids)} documents to delete for dataset {dataset_id}"
        )

        # Step 2: Batch delete document chunks using bulk updates
        await self._batch_delete_document_chunks(
            document_ids, current_time, batch_size, db_session
        )

        # Step 3: Batch delete file splits using bulk updates
        await self._batch_delete_file_splits(
            dataset_id, document_ids, current_time, batch_size, db_session
        )

        # Step 4: Batch delete documents using bulk updates
        await self._batch_delete_documents(
            document_ids, current_time, batch_size, db_session
        )

        logger.info(f"Completed batch deletion of all content for dataset {dataset_id}")

    async def _batch_delete_document_chunks(
        self,
        document_ids: List[UUID],
        current_time: datetime,
        batch_size: int,
        db_session: AsyncSession,
    ) -> None:
        """Batch delete document chunks using bulk updates."""
        from sqlalchemy import update

        total_chunks_deleted = 0

        # Process document IDs in batches to avoid huge IN clauses
        for i in range(0, len(document_ids), batch_size):
            batch_doc_ids = document_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(document_ids) + batch_size - 1) // batch_size

            try:
                # Bulk update chunks for this batch of documents
                chunks_update_stmt = (
                    update(DocumentChunk)
                    .where(
                        DocumentChunk.document_id.in_(batch_doc_ids),
                        DocumentChunk.deleted_at.is_(None),
                    )
                    .values(deleted_at=current_time)
                )

                chunks_result = await db_session.execute(chunks_update_stmt)
                batch_chunks_deleted = chunks_result.rowcount
                total_chunks_deleted += batch_chunks_deleted

                # Commit this batch
                await db_session.commit()

                logger.info(
                    f"Deleted {batch_chunks_deleted} chunks in batch {batch_num}/{total_batches}"
                )

            except Exception as e:
                logger.error(f"Error deleting chunks batch {batch_num}: {str(e)}")
                await db_session.rollback()
                raise

        logger.info(f"Total document chunks deleted: {total_chunks_deleted}")

    async def _batch_delete_file_splits(
        self,
        dataset_id: UUID,
        document_ids: List[UUID],
        current_time: datetime,
        batch_size: int,
        db_session: AsyncSession,
    ) -> None:
        """Batch delete file splits using bulk updates."""
        from sqlalchemy import update

        total_splits_deleted = 0

        # Process document IDs in batches
        for i in range(0, len(document_ids), batch_size):
            batch_doc_ids = document_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(document_ids) + batch_size - 1) // batch_size

            try:
                # Bulk update splits for this batch - use both dataset_id and document_id for efficiency
                splits_update_stmt = (
                    update(FileSplit)
                    .where(
                        FileSplit.dataset_id == dataset_id,
                        FileSplit.document_id.in_(batch_doc_ids),
                        FileSplit.deleted_at.is_(None),
                    )
                    .values(deleted_at=current_time)
                )

                splits_result = await db_session.execute(splits_update_stmt)
                batch_splits_deleted = splits_result.rowcount
                total_splits_deleted += batch_splits_deleted

                # Commit this batch
                await db_session.commit()

                logger.info(
                    f"Deleted {batch_splits_deleted} splits in batch {batch_num}/{total_batches}"
                )

            except Exception as e:
                logger.error(f"Error deleting splits batch {batch_num}: {str(e)}")
                await db_session.rollback()
                raise

        logger.info(f"Total file splits deleted: {total_splits_deleted}")

    async def _batch_delete_documents(
        self,
        document_ids: List[UUID],
        current_time: datetime,
        batch_size: int,
        db_session: AsyncSession,
    ) -> None:
        """Batch delete documents using bulk updates."""
        from sqlalchemy import update

        total_docs_deleted = 0

        # Process document IDs in batches
        for i in range(0, len(document_ids), batch_size):
            batch_doc_ids = document_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(document_ids) + batch_size - 1) // batch_size

            try:
                # Bulk update documents for this batch
                docs_update_stmt = (
                    update(Document)
                    .where(
                        Document.id.in_(batch_doc_ids), Document.deleted_at.is_(None)
                    )
                    .values(deleted_at=current_time)
                )

                docs_result = await db_session.execute(docs_update_stmt)
                batch_docs_deleted = docs_result.rowcount
                total_docs_deleted += batch_docs_deleted

                # Commit this batch
                await db_session.commit()

                logger.info(
                    f"Deleted {batch_docs_deleted} documents in batch {batch_num}/{total_batches}"
                )

            except Exception as e:
                logger.error(f"Error deleting documents batch {batch_num}: {str(e)}")
                await db_session.rollback()
                raise

        logger.info(f"Total documents deleted: {total_docs_deleted}")

    async def _validate_update_inputs_v2(
        self, obj_in: IDatasetUpdate, db_session: AsyncSession
    ) -> None:
        """Validate update inputs for V2."""
        # Check if file_ids and source_id are both provided
        if (
            hasattr(obj_in, "file_ids")
            and obj_in.file_ids
            and obj_in.file_ids != []
            and obj_in.source_id is not None
        ):
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both non-empty file_ids and source_id. Please provide only one of them.",
            )

        # Validate source if provided
        if obj_in.source_id:
            await self._validate_source_v2(obj_in.source_id, db_session)

    async def _validate_source_v2(
        self, source_id: UUID, db_session: AsyncSession
    ) -> None:
        """Validate that source exists and is accessible."""
        source_query = select(Source).where(
            Source.id == source_id,
            Source.deleted_at.is_(None),
        )
        result = await db_session.execute(source_query)
        source = result.scalars().first()

        if not source:
            raise HTTPException(
                status_code=404, detail=f"Source with id {source_id} not found"
            )

    async def _get_dataset_file_ids_v2(
        self, dataset_id: UUID, db_session: AsyncSession
    ) -> List[UUID]:
        """
        Get file IDs associated with a dataset.
        """
        from app.models.dataset_file_link_model import DatasetFileLink

        links_query = select(DatasetFileLink.file_id).where(
            DatasetFileLink.dataset_id == dataset_id
        )
        links_result = await db_session.execute(links_query)
        file_ids_from_links = list(links_result.scalars())

        return set(file_ids_from_links)

    async def _get_current_file_ids_v2(
        self, dataset_id: UUID, db_session: AsyncSession
    ) -> Set[UUID]:
        """Get current file IDs for a dataset."""
        file_ids = await self._get_dataset_file_ids_v2(dataset_id, db_session)
        return set(file_ids)

    @retry_on_deadlock(max_retries=3, delay=0.1)
    async def _link_files_v2(
        self, dataset_id: UUID, file_ids: List[UUID], db_session: AsyncSession
    ) -> None:
        """Link files to the dataset (V2 version) with batch processing and deadlock prevention."""

        if not file_ids:
            return

        # Batch size for processing to prevent deadlocks and memory issues
        # Start with default, but can be reduced if deadlocks occur
        batch_size = min(DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE)

        # Convert to set for faster lookups and remove duplicates
        unique_file_ids = list(set(file_ids))

        logger.info(
            f"Linking {len(unique_file_ids)} files to dataset {dataset_id} in batches of {batch_size}"
        )

        # Process files in batches to prevent deadlocks
        for i in range(0, len(unique_file_ids), batch_size):
            batch_file_ids = unique_file_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(unique_file_ids) + batch_size - 1) // batch_size

            retry_count = 0
            max_retries = 3
            current_batch_size = len(batch_file_ids)

            while retry_count <= max_retries:
                try:
                    # Use a separate transaction for each batch to reduce lock time
                    await self._process_file_link_batch(
                        dataset_id, batch_file_ids, db_session
                    )
                    logger.info(
                        f"Successfully linked batch {batch_num}/{total_batches} ({current_batch_size} files)"
                    )
                    break  # Success, move to next batch

                except Exception as e:
                    error_msg = str(e).lower()
                    if "deadlock" in error_msg and retry_count < max_retries:
                        retry_count += 1
                        wait_time = 0.1 * (2**retry_count)  # Exponential backoff
                        logger.warning(
                            f"Deadlock in batch {batch_num}/{total_batches}, retry {retry_count}/{max_retries} after {wait_time}s"
                        )

                        # If we're getting repeated deadlocks, try reducing batch size
                        if retry_count > 1 and current_batch_size > 100:
                            # Split the batch in half
                            mid_point = current_batch_size // 2
                            first_half = batch_file_ids[:mid_point]
                            second_half = batch_file_ids[mid_point:]

                            logger.info(
                                f"Splitting batch {batch_num} into smaller chunks: {mid_point} + {len(second_half)} files"
                            )

                            await asyncio.sleep(wait_time)

                            # Process first half
                            await self._process_file_link_batch(
                                dataset_id, first_half, db_session
                            )
                            logger.info(
                                f"Successfully processed first half of batch {batch_num}"
                            )

                            # Brief pause between halves
                            await asyncio.sleep(0.05)

                            # Process second half
                            await self._process_file_link_batch(
                                dataset_id, second_half, db_session
                            )
                            logger.info(
                                f"Successfully processed second half of batch {batch_num}"
                            )
                            break
                        else:
                            await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"Error processing batch {batch_num}/{total_batches}: {str(e)}"
                        )
                        if "deadlock" in error_msg:
                            logger.error(
                                f"Failed to process batch {batch_num} after {max_retries} retries"
                            )
                        raise

    @retry_on_deadlock(max_retries=3, delay=0.1)
    async def _process_file_link_batch(
        self, dataset_id: UUID, batch_file_ids: List[UUID], db_session: AsyncSession
    ) -> None:
        """Process a batch of file links with optimized queries and reduced lock time."""
        from app.models.dataset_file_link_model import DatasetFileLink

        if not batch_file_ids:
            return

        # Batch validate files exist (single query instead of N queries)
        # Order by ID to maintain consistent lock ordering and reduce deadlocks
        existing_files_query = (
            select(File.id)
            .where(File.id.in_(batch_file_ids), File.deleted_at.is_(None))
            .order_by(File.id)
        )

        existing_files_result = await db_session.execute(existing_files_query)
        existing_file_ids = set(existing_files_result.scalars().all())

        # Check for missing files
        missing_files = set(batch_file_ids) - existing_file_ids
        if missing_files:
            logger.warning(
                f"Skipping {len(missing_files)} missing files: {list(missing_files)[:10]}{'...' if len(missing_files) > 10 else ''}"
            )
            # Only process existing files
            valid_file_ids = [fid for fid in batch_file_ids if fid in existing_file_ids]
        else:
            valid_file_ids = batch_file_ids

        if not valid_file_ids:
            return

        # Batch check for existing links (single query instead of N queries)
        # Order by dataset_id, file_id for consistent lock ordering
        existing_links_query = (
            select(DatasetFileLink.file_id)
            .where(
                DatasetFileLink.dataset_id == dataset_id,
                DatasetFileLink.file_id.in_(valid_file_ids),
            )
            .order_by(DatasetFileLink.dataset_id, DatasetFileLink.file_id)
        )

        existing_links_result = await db_session.execute(existing_links_query)
        already_linked_file_ids = set(existing_links_result.scalars().all())

        # Only create links for files that don't already have them
        files_to_link = [
            fid for fid in valid_file_ids if fid not in already_linked_file_ids
        ]

        if not files_to_link:
            logger.debug(
                f"All {len(valid_file_ids)} files in batch already linked, skipping"
            )
            return

        logger.info(
            f"Creating {len(files_to_link)} new file links out of {len(batch_file_ids)} files in batch"
        )

        # Sort file IDs to ensure consistent lock ordering across transactions
        files_to_link.sort()

        # Use bulk insert for better performance and reduced lock time
        new_links = [
            DatasetFileLink(dataset_id=dataset_id, file_id=file_id)
            for file_id in files_to_link
        ]

        # Add all links at once and commit immediately to minimize lock time
        db_session.add_all(new_links)
        await db_session.commit()

        logger.debug(f"Successfully committed {len(new_links)} file links")

    async def _create_default_configs_v2(self, dataset_id: UUID) -> None:
        """Create default configurations for the dataset (V2 version)."""
        from app.crud.chunking_config_crud import chunking_config
        from app.schemas.chunking_config_schema import (
            IUnstructuredChunkingStrategyEnum,
            IUnstructuredStrategyEnum,
            UnstructuredChunkingConfig,
        )

        chunking_details = UnstructuredChunkingConfig(
            name="Default chunking config",
            provider="unstructured_local",
            strategy=IUnstructuredStrategyEnum.auto,
            chunking_strategy=IUnstructuredChunkingStrategyEnum.by_title,
            new_after_n_char=512,
            max_characters=2500,
            combine_under_n_chars=128,
            overlap=250,
        )
        await chunking_config.create_or_update_chunking_config(
            obj_in=chunking_details, dataset_id=dataset_id
        )

    async def _handle_source_and_file_changes_v2(
        self, dataset_record: Dataset, obj_in: IDatasetUpdate, db_session: AsyncSession
    ) -> None:
        """Handle source and file changes for V2 update - REPLACEMENT behavior."""

        # Get current file IDs
        current_file_ids = await self._get_current_file_ids_v2(
            dataset_record.id, db_session
        )

        new_file_ids = current_file_ids.copy()

        # Check which fields were actually provided in the request
        fields_set = (
            getattr(obj_in, "__fields_set__", set())
            if hasattr(obj_in, "__fields_set__")
            else set()
        )

        source_id_in_request = "source_id" in fields_set
        file_ids_in_request = "file_ids" in fields_set

        logger.info(f"Fields provided in request: {fields_set}")
        logger.info(f"source_id in request: {source_id_in_request}")
        logger.info(f"file_ids in request: {file_ids_in_request}")

        # Validate mutual exclusion
        if source_id_in_request and file_ids_in_request:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both source_id and file_ids in the same request.",
            )

        # Handle source_id if provided
        if source_id_in_request:
            if obj_in.source_id is not None:
                # Setting a source
                if obj_in.source_id != dataset_record.source_id:
                    logger.info(f"Setting source to {obj_in.source_id}")
                    await self._validate_source_v2(obj_in.source_id, db_session)
                    dataset_record.source_id = obj_in.source_id

                    source_file_ids = await self._get_source_files_v2(
                        obj_in.source_id, db_session
                    )
                    new_file_ids = set(source_file_ids)
                    logger.info(f"Replaced with {len(new_file_ids)} files from source")
            else:
                # Clearing the source
                logger.info("Clearing source_id")
                dataset_record.source_id = None

        # Handle file_ids if provided
        elif file_ids_in_request:
            # Replace with explicit files
            new_file_ids = set(obj_in.file_ids or [])
            logger.info(f"Setting explicit files: {len(new_file_ids)} files")

            # Clear source when setting explicit files
            if dataset_record.source_id is not None:
                logger.info("Clearing source_id as explicit files were set")
                dataset_record.source_id = None

        # Process file changes
        if new_file_ids != current_file_ids:
            await self._process_file_changes_v2(
                dataset_record.id, current_file_ids, new_file_ids, db_session
            )
        else:
            logger.info("No file changes detected")

    async def _get_source_files_v2(
        self, source_id: UUID, db_session: AsyncSession
    ) -> Set[UUID]:
        """
        Get file IDs from a source.

        This gets all files that belong to the source, regardless of dataset association.
        The association with the dataset will be established through Document records
        during ingestion.
        """
        files_query = select(File.id).where(
            File.source_id == source_id,
            File.deleted_at.is_(None),
        )
        result = await db_session.execute(files_query)
        file_ids = set(result.scalars())

        logger.info(f"Source {source_id}: Found {len(file_ids)} files")
        return file_ids

    async def _process_file_changes_v2(
        self,
        dataset_id: UUID,
        current_file_ids: Set[UUID],
        new_file_ids: Set[UUID],
        db_session: AsyncSession,
    ) -> None:
        """
        Process file changes - REPLACEMENT behavior with batch processing.
        Removes files not in new_file_ids and adds files not in current_file_ids.
        """

        # Files to remove (in current but not in new)
        files_to_remove = current_file_ids - new_file_ids

        # Files to add (in new but not in current)
        files_to_add = new_file_ids - current_file_ids

        logger.info(f"Dataset {dataset_id} file replacement analysis:")
        logger.info(f"  Current files: {len(current_file_ids)} files")
        logger.info(f"  Target files: {len(new_file_ids)} files")
        logger.info(f"  Files to remove: {len(files_to_remove)} files")
        logger.info(f"  Files to add: {len(files_to_add)} files")

        # Log detailed file IDs for debugging (limited to avoid log spam)
        if files_to_remove:
            if len(files_to_remove) <= 10:
                logger.info(f"  Removing files: {[str(f) for f in files_to_remove]}")
            else:
                logger.info(
                    f"  Removing {len(files_to_remove)} files (too many to list)"
                )

        if files_to_add:
            if len(files_to_add) <= 10:
                logger.info(f"  Adding files: {[str(f) for f in files_to_add]}")
            else:
                logger.info(f"  Adding {len(files_to_add)} files (too many to list)")

        # Remove files that are no longer needed - use batch removal for efficiency
        if files_to_remove:
            logger.info(f"Starting batch removal of {len(files_to_remove)} files")
            if len(files_to_remove) > 100:  # Use batch removal for large sets
                await self._batch_remove_files_from_dataset(
                    dataset_id, list(files_to_remove), db_session
                )
            else:
                # Use individual removal for small sets
                for file_id in files_to_remove:
                    await self._remove_file_from_dataset_v2(
                        dataset_id, file_id, db_session
                    )

        # Add new files using batch processing
        if files_to_add:
            logger.info(f"Starting batch addition of {len(files_to_add)} files")
            await self._link_files_v2(dataset_id, list(files_to_add), db_session)

        logger.info(
            f"Dataset {dataset_id}: Processed {len(files_to_remove)} file removals, {len(files_to_add)} file additions"
        )

    async def _remove_file_from_dataset_v2(
        self, dataset_id: UUID, file_id: UUID, db_session: AsyncSession
    ) -> None:
        """
        Remove a file from dataset (V2 approach).
        Deletes Documents, DocumentChunks, and FileSplits for this file in this dataset.
        """
        from sqlalchemy import delete

        from app.models.dataset_file_link_model import DatasetFileLink
        from app.models.document_chunk_model import DocumentChunk
        from app.models.file_split_model import FileSplit

        logger.info(f"Removing file {file_id} from dataset {dataset_id}")

        # 1. Delete DocumentChunks for this file in this dataset
        chunks_delete_query = delete(DocumentChunk).where(
            DocumentChunk.document_id.in_(
                select(Document.id).where(
                    Document.file_id == file_id, Document.dataset_id == dataset_id
                )
            )
        )
        chunks_result = await db_session.execute(chunks_delete_query)
        logger.info(
            f"Deleted {chunks_result.rowcount} document chunks for file {file_id}"
        )

        # 2. Delete FileSplits for this file in this dataset
        splits_delete_query = delete(FileSplit).where(
            FileSplit.original_file_id == file_id, FileSplit.dataset_id == dataset_id
        )
        splits_result = await db_session.execute(splits_delete_query)
        logger.info(f"Deleted {splits_result.rowcount} file splits for file {file_id}")

        # 3. Soft delete Documents for this file in this dataset
        documents_query = select(Document).where(
            Document.file_id == file_id,
            Document.dataset_id == dataset_id,
            Document.deleted_at.is_(None),
        )
        documents_result = await db_session.execute(documents_query)
        documents = documents_result.scalars().all()

        deleted_docs_count = 0
        for document in documents:
            document.deleted_at = datetime.utcnow()
            deleted_docs_count += 1

        logger.info(f"Soft deleted {deleted_docs_count} documents for file {file_id}")

        # 4. Remove DatasetFileLink
        link_delete_query = delete(DatasetFileLink).where(
            DatasetFileLink.dataset_id == dataset_id, DatasetFileLink.file_id == file_id
        )
        _link_result = await db_session.execute(link_delete_query)
        logger.info(f"Removed dataset-file link for file {file_id}")

        logger.info(f"Successfully removed file {file_id} from dataset {dataset_id}")

    @retry_on_deadlock(max_retries=3, delay=0.1)
    async def _batch_remove_files_from_dataset(
        self, dataset_id: UUID, file_ids: List[UUID], db_session: AsyncSession
    ) -> None:
        """
        Batch remove multiple files from dataset for better performance.
        Uses bulk operations instead of individual file removal.
        """
        from sqlalchemy import delete, update

        from app.models.dataset_file_link_model import DatasetFileLink
        from app.models.document_chunk_model import DocumentChunk
        from app.models.file_split_model import FileSplit

        if not file_ids:
            return

        logger.info(f"Batch removing {len(file_ids)} files from dataset {dataset_id}")
        current_time = datetime.utcnow()

        # Batch size for processing
        batch_size = min(DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE)

        # Process files in batches
        for i in range(0, len(file_ids), batch_size):
            batch_file_ids = file_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(file_ids) + batch_size - 1) // batch_size

            try:
                # 1. Get document IDs for these files in this dataset
                docs_query = select(Document.id).where(
                    Document.file_id.in_(batch_file_ids),
                    Document.dataset_id == dataset_id,
                    Document.deleted_at.is_(None),
                )
                docs_result = await db_session.execute(docs_query)
                document_ids = list(docs_result.scalars().all())

                if document_ids:
                    # 2. Bulk delete DocumentChunks
                    chunks_delete_query = delete(DocumentChunk).where(
                        DocumentChunk.document_id.in_(document_ids)
                    )
                    chunks_result = await db_session.execute(chunks_delete_query)
                    logger.info(
                        f"Deleted {chunks_result.rowcount} chunks in batch {batch_num}"
                    )

                    # 3. Bulk delete FileSplits
                    splits_delete_query = delete(FileSplit).where(
                        FileSplit.original_file_id.in_(batch_file_ids),
                        FileSplit.dataset_id == dataset_id,
                    )
                    splits_result = await db_session.execute(splits_delete_query)
                    logger.info(
                        f"Deleted {splits_result.rowcount} splits in batch {batch_num}"
                    )

                    # 4. Bulk soft-delete Documents
                    docs_update_query = (
                        update(Document)
                        .where(Document.id.in_(document_ids))
                        .values(deleted_at=current_time)
                    )
                    docs_update_result = await db_session.execute(docs_update_query)
                    logger.info(
                        f"Soft deleted {docs_update_result.rowcount} documents in batch {batch_num}"
                    )

                # 5. Bulk delete DatasetFileLinks
                links_delete_query = delete(DatasetFileLink).where(
                    DatasetFileLink.dataset_id == dataset_id,
                    DatasetFileLink.file_id.in_(batch_file_ids),
                )
                links_result = await db_session.execute(links_delete_query)
                logger.info(
                    f"Deleted {links_result.rowcount} file links in batch {batch_num}"
                )

                # Commit this batch
                await db_session.commit()
                logger.info(
                    f"Successfully processed file removal batch {batch_num}/{total_batches}"
                )

            except Exception as e:
                logger.error(f"Error in file removal batch {batch_num}: {str(e)}")
                await db_session.rollback()
                raise

        logger.info(
            f"Completed batch removal of {len(file_ids)} files from dataset {dataset_id}"
        )

    def link_file_to_dataset_sync(
        self, dataset_id: UUID, file_id: UUID, db_session
    ) -> None:
        """
        Synchronous function to link a single file to a dataset.

        Args:
            dataset_id: The dataset ID to link the file to
            file_id: The file ID to link
            db_session: Synchronous database session
        """
        from app.models.dataset_file_link_model import DatasetFileLink

        # Check if file exists
        file_exists = (
            db_session.query(File.id)
            .filter(File.id == file_id, File.deleted_at.is_(None))
            .first()
        )

        if not file_exists:
            logger.warning(f"File {file_id} not found or deleted, skipping link")
            return

        # Check if link already exists
        existing_link = (
            db_session.query(DatasetFileLink)
            .filter(
                DatasetFileLink.dataset_id == dataset_id,
                DatasetFileLink.file_id == file_id,
            )
            .first()
        )

        if existing_link:
            logger.debug(f"File {file_id} already linked to dataset {dataset_id}")
            return

        # Create the link
        new_link = DatasetFileLink(dataset_id=dataset_id, file_id=file_id)
        db_session.add(new_link)
        db_session.commit()

        logger.info(f"Successfully linked file {file_id} to dataset {dataset_id}")

    def update_ingestion_status_sync(
        self, dataset_id: UUID, status: str, db_session: Session
    ) -> bool:
        """Synchronous version to update ingestion_status for a dataset"""
        try:
            # Get the dataset
            stmt = select(Dataset).where(Dataset.id == dataset_id)
            result = db_session.execute(stmt)
            dataset_record = result.scalar_one_or_none()

            if not dataset_record:
                logger.warning("Dataset %s not found for status update", dataset_id)
                return False

            # Update the ingestion_status
            dataset_record.ingestion_status = status
            db_session.commit()

            logger.info(
                "Successfully updated dataset %s ingestion_status to '%s'",
                dataset_id,
                status,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to update ingestion_status for dataset %s: %s", dataset_id, e
            )
            db_session.rollback()
            return False

    async def file_belongs_to_dataset_check(
        self, *, dataset_id: UUID, file_id: UUID, db_session: AsyncSession | None = None
    ) -> bool:
        """Check if a file belongs to a dataset."""
        from app.models.dataset_file_link_model import DatasetFileLink

        db_session = db_session or super().get_db().session

        link_query = select(DatasetFileLink).where(
            DatasetFileLink.dataset_id == dataset_id,
            DatasetFileLink.file_id == file_id,
        )
        result = await db_session.execute(link_query)
        link = result.scalars().first()

        return link is not None

    async def remove_single_file_from_dataset(
        self, *, dataset_id: UUID, file_id: UUID, db_session: AsyncSession | None = None
    ) -> None:
        """Remove a single file from dataset - public method."""
        db_session = db_session or super().get_db().session

        try:
            await self._remove_file_from_dataset_v2(dataset_id, file_id, db_session)
            await db_session.commit()
            logger.info(
                f"Successfully removed file {file_id} from dataset {dataset_id}"
            )
        except Exception as e:
            await db_session.rollback()
            logger.error(
                f"Error removing file {file_id} from dataset {dataset_id}: {str(e)}"
            )
            raise

    async def get_source_id_by_dataset_id(
        self, *, dataset_id: UUID, db_session: AsyncSession | None = None
    ) -> UUID | None:
        db_session = db_session or super().get_db().session
        query = select(Dataset.source_id).where(Dataset.id == dataset_id)
        result = await db_session.execute(query)
        source_id = result.scalar_one_or_none()
        logger.debug(
            f"[Dataset] dataset_id={dataset_id}, resolved source_id={source_id}"
        )
        return source_id

    async def update_ingestion_status(
        self, dataset_id: UUID, status: str, db_session: AsyncSession
    ) -> bool:
        """Update ingestion_status for a dataset"""
        try:
            # Get the dataset
            stmt = select(Dataset).where(Dataset.id == dataset_id)
            result = await db_session.execute(stmt)
            dataset_record = result.scalar_one_or_none()

            if not dataset_record:
                logger.warning("Dataset %s not found for status update", dataset_id)
                return False

            # Update the ingestion_status
            dataset_record.ingestion_status = status
            await db_session.commit()

            logger.info(
                "Successfully updated dataset %s ingestion_status to '%s'",
                dataset_id,
                status,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to update ingestion_status for dataset %s: %s", dataset_id, e
            )
            await db_session.rollback()
            return False

    async def get_workspace_id_of_dataset(
        self, *, dataset_id: UUID, db_session: AsyncSession | None = None
    ) -> UUID:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Dataset).where(
                Dataset.id == dataset_id, Dataset.deleted_at.is_(None)
            )
        )
        dataset_instance = result.unique().scalar_one_or_none()

        if not dataset_instance:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        return dataset_instance.workspace_id

    def get_dataset_sync(
        self,
        *,
        dataset_id: UUID,
        db_session: Session | None = None,
    ) -> IDatasetRead | None:
        db_session = db_session or super().get_db().session
        dataset_record = db_session.execute(
            select(Dataset).where(
                Dataset.id == dataset_id,
                Dataset.deleted_at.is_(None),
            )
        )
        dataset_instance = dataset_record.scalars().first()

        if not dataset_instance:
            raise HTTPException(status_code=404, detail="Dataset not found")

        file_links = db_session.execute(
            select(DatasetFileLink.file_id).where(
                DatasetFileLink.dataset_id == dataset_instance.id
            )
        )
        file_ids = list(file_links.scalars())

        # Get source_type if source_id exists
        source_type = None
        if dataset_instance.source_id:
            source_result = db_session.execute(
                select(Source.source_type).where(
                    Source.id == dataset_instance.source_id
                )
            )
            source_type = source_result.scalar_one_or_none()

        dataset_data = IDatasetRead(
            id=dataset_instance.id,
            name=dataset_instance.name,
            description=dataset_instance.description,
            file_ids=file_ids,
            r2r_collection_id=dataset_instance.r2r_collection_id,
            source_id=dataset_instance.source_id,
            source_type=source_type,
            knowledge_graph=dataset_instance.knowledge_graph,
            graph_build_phase=dataset_instance.graph_build_phase,
            graph_build_requested_at=dataset_instance.graph_build_requested_at,
            graph_build_completed_at=dataset_instance.graph_build_completed_at,
            last_extraction_check_at=dataset_instance.last_extraction_check_at,
        )

        return dataset_data


# Create instance
dataset_v2 = CRUDDatasetV2(Dataset)
