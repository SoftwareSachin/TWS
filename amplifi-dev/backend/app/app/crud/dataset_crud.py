from datetime import UTC, datetime
from typing import List, Optional, Union
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Page, Params, paginate
from sqlalchemy import and_, asc, desc, exists, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlmodel import select

from app.be_core.celery import celery
from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.crud.chunking_config_crud import chunking_config
from app.crud.ingest_crud import ingestion_crud
from app.db.session import SyncSessionLocal
from app.models import FileIngestion
from app.models.dataset_file_link_model import DatasetFileLink
from app.models.dataset_model import Dataset
from app.models.document_model import Document, DocumentProcessingStatusEnum
from app.models.file_ingestion_model import FileIngestionStatusType
from app.models.file_model import File
from app.models.graph_model import Graph, GraphStatus
from app.models.pull_status_model import PullStatusEnum, SourcePullStatus
from app.models.source_model import Source
from app.models.workspace_model import Workspace
from app.schemas.chunking_config_schema import (
    IUnstructuredChunkingStrategyEnum,
    IUnstructuredStrategyEnum,
    UnstructuredChunkingConfig,
)
from app.schemas.common_schema import IOrderEnum
from app.schemas.dataset_schema import (
    DatasetFileTypeGroup,
    IDatasetCreate,
    IDatasetRead,
    IDatasetUpdate,
)
from app.schemas.file_schema import FileStatusEnum
from app.schemas.response_schema import IngestionStatusEnum
from app.utils.optional_params import OptionalParams


class CRUDDataset(CRUDBase[Dataset, IDatasetCreate, IDatasetUpdate]):
    async def _get_graph_status_for_dataset(
        self,
        *,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Optional[bool]:
        """
        Get graph status for a dataset.
        Returns:
            Optional[bool]: True if graph successfully created, False if failed/pending, None if not started
        """
        db_session = db_session or super().get_db().session

        # Get the most recent graph for this dataset
        result = await db_session.execute(
            select(Graph)
            .where(Graph.dataset_id == dataset_id, Graph.deleted_at.is_(None))
            .order_by(Graph.created_at.desc())
            .limit(1)
        )
        graph = result.scalar_one_or_none()

        if not graph:
            return None  # No graph exists

        # Check if both entities and relationships are successful
        is_successful = (
            graph.entities_status == GraphStatus.SUCCESS
            and graph.relationships_status == GraphStatus.SUCCESS
        )

        return is_successful

    async def create_dataset(
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
            await self._validate_source(obj_in.source_id, db_session, "creation")

        # Create R2R collection
        collection_id = self.create_r2r_collection(obj_in, organization_id)

        # Create dataset record
        dataset_record = Dataset(
            name=obj_in.name,
            description=obj_in.description,
            workspace_id=workspace_id,
            source_id=obj_in.source_id,
            r2r_collection_id=collection_id,
        )
        db_session.add(dataset_record)
        await db_session.commit()
        await db_session.refresh(dataset_record)

        # Link manually specified files if any
        if obj_in.file_ids:
            await self._link_files(dataset_record.id, obj_in.file_ids, db_session)

        # Link all files from source if source_id is provided
        elif obj_in.source_id:
            source_files = await self._get_source_files(obj_in.source_id, db_session)
            await self._link_files(dataset_record.id, source_files, db_session)

        # Create default configurations
        await self._create_default_configs(dataset_record.id)

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
            r2r_collection_id=collection_id,
            knowledge_graph=dataset_record.knowledge_graph,
            graph_build_phase=dataset_record.graph_build_phase,
            graph_build_requested_at=dataset_record.graph_build_requested_at,
            graph_build_completed_at=dataset_record.graph_build_completed_at,
            last_extraction_check_at=dataset_record.last_extraction_check_at,
        )
        return dataset_data

    async def get_dataset(
        self,
        *,
        dataset_id: UUID,
        workspace_id: UUID | None = None,
        db_session: AsyncSession | None = None,
    ) -> IDatasetRead | None:
        db_session = db_session or super().get_db().session

        # Build query conditionally based on workspace_id
        conditions = [
            Dataset.id == dataset_id,
            Dataset.deleted_at.is_(None),
        ]
        if workspace_id is not None:
            conditions.append(Dataset.workspace_id == workspace_id)

        dataset_record = await db_session.execute(select(Dataset).where(*conditions))
        dataset_instance = dataset_record.scalars().first()

        if not dataset_instance:
            raise HTTPException(status_code=404, detail="Dataset not found")
        file_links = await db_session.execute(
            select(DatasetFileLink.file_id).where(
                DatasetFileLink.dataset_id == dataset_instance.id
            )
        )
        file_ids = list(file_links.scalars())

        # Get source_type if source_id exists
        source_type = None
        if dataset_instance.source_id:
            source_result = await db_session.execute(
                select(Source.source_type).where(
                    Source.id == dataset_instance.source_id
                )
            )
            source_type = source_result.scalar_one_or_none()

        # Get graph status
        graph_status = await self._get_graph_status_for_dataset(
            dataset_id=dataset_instance.id, db_session=db_session
        )

        dataset_data = IDatasetRead(
            id=dataset_instance.id,
            name=dataset_instance.name,
            description=dataset_instance.description,
            file_ids=file_ids,
            source_id=dataset_instance.source_id,
            source_type=source_type,
            r2r_collection_id=dataset_instance.r2r_collection_id,
            knowledge_graph=dataset_instance.knowledge_graph,
            graph_build_phase=dataset_instance.graph_build_phase,
            graph_build_requested_at=dataset_instance.graph_build_requested_at,
            graph_build_completed_at=dataset_instance.graph_build_completed_at,
            last_extraction_check_at=dataset_instance.last_extraction_check_at,
            graph_status=graph_status,
        )

        return dataset_data

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

    async def update_dataset(
        self,
        *,
        obj_in: IDatasetUpdate,
        dataset_id: UUID,
        workspace_id: UUID,
        organization_id: Optional[UUID] = None,
        db_session: AsyncSession | None = None,
    ):
        db_session = db_session or super().get_db().session

        # Validate inputs
        await self._validate_update_inputs(obj_in)

        # Get dataset and check ingestion status
        dataset_record = await self._get_dataset_for_update(
            dataset_id, workspace_id, db_session
        )

        # Track explicitly updated fields
        explicit_updates = self._get_explicit_updates(obj_in)
        logger.info(
            f"Updating dataset {dataset_id} with explicit fields: {explicit_updates}"
        )

        # Check for name uniqueness if name is being updated
        if "name" in explicit_updates and obj_in.name != dataset_record.name:
            name_conflict_query = select(Dataset).where(
                func.lower(func.trim(Dataset.name))
                == func.lower(func.trim(obj_in.name)),
                Dataset.workspace_id == workspace_id,
                Dataset.deleted_at.is_(None),
                Dataset.id != dataset_id,  # Exclude the current dataset from the check
            )
            existing_result = await db_session.execute(name_conflict_query)
            conflict_dataset = existing_result.scalars().first()
            if conflict_dataset:
                raise HTTPException(
                    status_code=400,
                    detail=f"A dataset with name '{obj_in.name}' already exists in this workspace.",
                )

        # Update basic fields (name, description)
        self._update_basic_fields(dataset_record, obj_in, explicit_updates)

        # Get current file IDs for comparison
        current_file_ids = await self._get_current_file_ids(dataset_id, db_session)

        # Handle source and file changes
        new_file_ids, source_changed = await self._handle_source_and_file_changes(
            dataset_record, obj_in, explicit_updates, current_file_ids, db_session
        )

        # Process file changes if needed
        await self._process_changed_files(
            dataset_record,
            organization_id,
            dataset_id,
            current_file_ids,
            new_file_ids,
            source_changed,
            db_session,
        )

        # Commit changes
        await db_session.commit()

        # Get source_type if source_id exists
        source_type = None
        if dataset_record.source_id:
            source_result = await db_session.execute(
                select(Source.source_type).where(Source.id == dataset_record.source_id)
            )
            source_type = source_result.scalar_one_or_none()

        # Return response with updated data
        logger.info(
            f"Dataset {dataset_record.name} updated with {len(new_file_ids)} files."
        )
        return self._create_dataset_response(dataset_record, new_file_ids, source_type)

    def _get_explicit_updates(self, obj_in: IDatasetUpdate) -> dict:
        """Identify which fields are being explicitly updated."""
        return {
            "name": obj_in.name is not None,
            "description": obj_in.description is not None,
            "source_id": obj_in.source_id is not None,
            "file_ids": hasattr(obj_in, "file_ids") and obj_in.file_ids is not None,
        }

    def _update_basic_fields(self, dataset_record, obj_in, explicit_updates):
        """Update basic fields if they are provided."""
        if explicit_updates["name"]:
            dataset_record.name = obj_in.name
        if explicit_updates["description"]:
            dataset_record.description = obj_in.description

    async def _handle_source_and_file_changes(
        self,
        dataset_record,
        obj_in,
        explicit_updates,
        current_file_ids,
        db_session,
    ) -> tuple[set, bool]:
        """Handle source and file changes, returning new file IDs and whether source changed."""
        new_file_ids = current_file_ids.copy()  # Default to keeping current files
        source_changed = False

        # Handle source_id changes
        if explicit_updates["source_id"]:
            source_changed, new_file_ids = await self._handle_source_change(
                dataset_record, obj_in, explicit_updates, current_file_ids, db_session
            )

        # Handle file_ids updates
        if explicit_updates["file_ids"] and not source_changed:
            new_file_ids = await self._handle_file_ids_update(
                dataset_record, obj_in, explicit_updates, current_file_ids, db_session
            )

            # Check if source needs to be cleared
            if (
                dataset_record.source_id is not None
                and not explicit_updates["source_id"]
            ):
                # Don't clear source if empty file_ids with existing source
                if not (obj_in.file_ids == [] and dataset_record.source_id is not None):
                    logger.info(
                        f"Clearing source_id {dataset_record.source_id} as explicit file_ids were provided"
                    )
                    dataset_record.source_id = None
                    source_changed = True

        return new_file_ids, source_changed

    async def _handle_source_change(
        self,
        dataset_record,
        obj_in,
        explicit_updates,
        current_file_ids,
        db_session,
    ) -> tuple[bool, set]:
        """Handle changes to the source_id field."""
        source_changed = False
        new_file_ids = current_file_ids.copy()

        # Check if source_id is changing
        if obj_in.source_id != dataset_record.source_id:
            logger.info(
                f"Updating dataset source from {dataset_record.source_id} to {obj_in.source_id}"
            )
            await self._validate_source(obj_in.source_id, db_session, "update")
            dataset_record.source_id = obj_in.source_id
            source_changed = True

            # Get files from new source
            if obj_in.source_id is not None:
                source_file_ids = await self._get_source_files(
                    obj_in.source_id, db_session
                )
                new_file_ids = set(source_file_ids)
                logger.info(
                    f"Using {len(new_file_ids)} files from new source {obj_in.source_id}"
                )
            else:
                # Source was removed, stick with current files or empty set depending on file_ids
                if not explicit_updates["file_ids"]:
                    logger.info(
                        f"Source removed but no file_ids provided, keeping current {len(current_file_ids)} files"
                    )
                else:
                    new_file_ids = set()
                    logger.info(
                        "Source removed and file_ids is empty, clearing all files"
                    )

        return source_changed, new_file_ids

    async def _handle_file_ids_update(
        self,
        dataset_record,
        obj_in,
        explicit_updates,
        current_file_ids,
        db_session,
    ) -> set:
        """Handle updates to the file_ids field."""
        # Special case: Empty file_ids array provided but dataset has a source
        if obj_in.file_ids == [] and dataset_record.source_id is not None:
            # Check if source isn't changing or is being explicitly set to the same value
            if (
                not explicit_updates["source_id"]
                or obj_in.source_id == dataset_record.source_id
            ):
                logger.warning(
                    f"Empty file_ids array with existing source_id {dataset_record.source_id}, keeping current files"
                )
                return current_file_ids
            else:
                # Using explicit empty file_ids with a source change
                logger.info(
                    "Using explicitly provided empty file_ids with source change"
                )
                return set(obj_in.file_ids)
        else:
            # Explicit file IDs provided, use them
            logger.info(
                f"Using explicitly provided file_ids: {len(obj_in.file_ids)} files"
            )
            return set(obj_in.file_ids)

    async def _process_changed_files(
        self,
        dataset_record,
        organization_id,
        dataset_id,
        current_file_ids,
        new_file_ids,
        source_changed,
        db_session,
    ) -> None:
        """Process file changes and handle cleanup if needed."""
        # Check if files actually changed
        files_changed = current_file_ids != new_file_ids

        if files_changed:
            logger.info(
                f"Files changed: removing {len(current_file_ids - new_file_ids)} files, "
                f"adding {len(new_file_ids - current_file_ids)} files"
            )

            # Process file changes
            r2r_ids_to_delete, file_ids_removed = await self._process_file_changes(
                dataset_id, current_file_ids, new_file_ids, db_session
            )

            # Handle knowledge graph reset if needed
            if dataset_record.knowledge_graph:
                reason = "files changed"
                await self._reset_knowledge_graph(
                    dataset_record, organization_id, db_session, reason
                )

            # Schedule R2R deletions if needed
            if r2r_ids_to_delete:
                logger.info(
                    f"Scheduling deletion of {len(r2r_ids_to_delete)} files from R2R"
                )
                self._schedule_r2r_deletions(
                    r2r_ids_to_delete, file_ids_removed, dataset_id, dataset_record
                )
            else:
                logger.info("No R2R deletions needed")

        elif source_changed and dataset_record.knowledge_graph:
            # Source changed but files didn't, still need to reset knowledge graph
            reason = "source changed"
            await self._reset_knowledge_graph(
                dataset_record, organization_id, db_session, reason
            )
        else:
            logger.info(
                f"No file changes detected, keeping current {len(current_file_ids)} files"
            )

    async def delete_dataset(
        self,
        *,
        dataset_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IDatasetRead | None:
        db_session = db_session or super().get_db().session

        # Get dataset record and validate it exists
        dataset_record = await self._get_dataset_by_id(
            dataset_id, workspace_id, db_session
        )

        # Get all files associated with the dataset
        file_link_records = await self._get_dataset_file_links(dataset_id, db_session)
        file_ids = [link.file_id for link in file_link_records]

        # Track R2R IDs to delete and file IDs processed
        r2r_ids_to_delete = []
        file_ids_removed = []

        # Process each file for R2R deletions and cleanup
        for file_link in file_link_records:
            await self._process_file_for_deletion(
                dataset_id, file_link, r2r_ids_to_delete, file_ids_removed, db_session
            )

        # Mark dataset as deleted and remove file links
        await self._delete_dataset_record(dataset_record, dataset_id, db_session)

        # Schedule R2R deletions if needed
        if r2r_ids_to_delete:
            self._schedule_r2r_deletion_task(
                r2r_ids_to_delete, file_ids_removed, dataset_id, dataset_record
            )

        # Get source_type if source_id exists
        source_type = None
        if dataset_record.source_id:
            source_result = await db_session.execute(
                select(Source.source_type).where(Source.id == dataset_record.source_id)
            )
            source_type = source_result.scalar_one_or_none()

        # Create and return response data
        return self._build_dataset_response(dataset_record, file_ids, source_type)

    async def _get_dataset_by_id(
        self, dataset_id: UUID, workspace_id: UUID, db_session: AsyncSession
    ) -> Dataset:
        """Get dataset record and validate it exists."""
        result = await db_session.execute(
            select(Dataset).where(
                Dataset.id == dataset_id,
                Dataset.workspace_id == workspace_id,
                Dataset.deleted_at.is_(None),
            )
        )
        dataset_record = result.scalars().first()

        if not dataset_record:
            raise HTTPException(status_code=404, detail="Dataset does not exist")

        return dataset_record

    async def _get_dataset_file_links(
        self, dataset_id: UUID, db_session: AsyncSession
    ) -> list:
        """Get all file links for a dataset."""
        file_links = await db_session.execute(
            select(DatasetFileLink).where(DatasetFileLink.dataset_id == dataset_id)
        )
        return file_links.scalars().all()

    async def _process_file_for_deletion(
        self,
        dataset_id: UUID,
        file_link: DatasetFileLink,
        r2r_ids_to_delete: list[str],
        file_ids_removed: list[str],
        db_session: AsyncSession,
    ) -> None:
        """Process a file for deletion within a dataset."""
        r2r_id = file_link.r2r_id
        file_id = file_link.file_id

        # Check if the file was successfully ingested
        ingestion_record = await self._get_successful_ingestion(
            dataset_id, file_id, db_session
        )
        was_successfully_ingested = ingestion_record is not None

        if not was_successfully_ingested:
            return

        # Handle file splits if the file was successfully ingested
        split_r2r_ids, splits = await self._get_file_splits(
            dataset_id, file_id, db_session
        )
        has_splits = len(splits) > 0

        # Process splits if they exist
        if has_splits:
            logger.info(
                f"File {file_id} has {len(splits)} splits that will be removed during dataset deletion"
            )
            await self._mark_splits_as_deleted(
                splits, split_r2r_ids, r2r_ids_to_delete, db_session
            )

        # Handle original file R2R ID
        if r2r_id and not has_splits:
            logger.info(
                f"Adding original file {file_id} with R2R ID {r2r_id} to deletion list"
            )
            r2r_ids_to_delete.append(str(r2r_id))
        elif has_splits:
            logger.info(
                f"File {file_id} has splits - skipping deletion of original R2R ID"
            )

        # Add file to removed list and mark ingestion as deleted
        file_ids_removed.append(str(file_id))
        if ingestion_record:
            ingestion_record.deleted_at = datetime.utcnow()
            db_session.add(ingestion_record)

    async def _get_successful_ingestion(
        self, dataset_id: UUID, file_id: UUID, db_session: AsyncSession
    ) -> FileIngestion | None:
        """Get successful ingestion record for a file in a dataset."""
        ingestion_stmt = select(FileIngestion).where(
            FileIngestion.dataset_id == dataset_id,
            FileIngestion.file_id == file_id,
            FileIngestion.status == FileIngestionStatusType.Success,
            FileIngestion.deleted_at.is_(None),
        )
        ingestion_result = await db_session.execute(ingestion_stmt)
        return ingestion_result.scalar_one_or_none()

    async def _get_file_splits(
        self, dataset_id: UUID, file_id: UUID, db_session: AsyncSession
    ) -> tuple[list[str], list]:
        """
        Get file splits and their R2R IDs, ordered by split_index.

        Args:
            dataset_id: Dataset ID
            file_id: Original file ID
            db_session: Database session

        Returns:
            Tuple of (list of R2R IDs for splits, list of split objects)
        """
        from uuid import NAMESPACE_URL, uuid5

        from app.models.file_split_model import FileSplit

        split_r2r_ids = []

        # Get all splits for this file, ordered by split_index for proper sequence
        splits_stmt = (
            select(FileSplit)
            .where(
                FileSplit.original_file_id == file_id,
                FileSplit.dataset_id == dataset_id,
            )
            .order_by(FileSplit.split_index)
        )  # Order by split_index to preserve file sequence

        splits_result = await db_session.execute(splits_stmt)
        splits = splits_result.scalars().all()

        # Generate R2R IDs for each split
        for split in splits:
            split_namespace = f"{file_id}_{split.id}_{dataset_id}"
            split_doc_id = str(uuid5(NAMESPACE_URL, split_namespace))
            split_r2r_ids.append(split_doc_id)
            logger.debug(
                f"Adding split {split.id} with R2R ID {split_doc_id} to deletion list"
            )

        return split_r2r_ids, splits

    async def _mark_splits_as_deleted(
        self,
        splits: list,
        split_r2r_ids: list[str],
        r2r_ids_to_delete: list[str],
        db_session: AsyncSession,
    ) -> None:
        """Mark splits as deleted and add their R2R IDs to deletion list."""
        # Add split R2R IDs to deletion list
        r2r_ids_to_delete.extend(split_r2r_ids)

        # Mark each split as deleted
        for split in splits:
            split.deleted_at = datetime.utcnow()
            db_session.add(split)

    async def _delete_dataset_record(
        self, dataset_record: Dataset, dataset_id: UUID, db_session: AsyncSession
    ) -> None:
        """Mark dataset as deleted and remove dataset-file links."""
        # Mark dataset as deleted
        dataset_record.deleted_at = datetime.now()
        await db_session.commit()

        # Remove all dataset-file links
        await db_session.execute(
            DatasetFileLink.__table__.delete().where(
                DatasetFileLink.dataset_id == dataset_id
            )
        )
        await db_session.commit()

    def _schedule_r2r_deletion_task(
        self,
        r2r_ids_to_delete: list[str],
        file_ids_removed: list[str],
        dataset_id: UUID,
        dataset_record: Dataset,
    ) -> None:
        """Schedule Celery task to delete documents from R2R."""
        logger.info(
            f"Scheduling deletion of {len(r2r_ids_to_delete)} documents from R2R for dataset {dataset_id}"
        )

        organization_id = (
            dataset_record.workspace.organization_id
            if hasattr(dataset_record, "workspace")
            else None
        )

        celery.signature(
            "tasks.delete_files_task",
            kwargs={
                "document_ids_to_delete": r2r_ids_to_delete,
                "file_ids": file_ids_removed,
                "dataset_id": str(dataset_id),
                "organization_id": organization_id,
            },
        ).apply_async()

    def _build_dataset_response(
        self,
        dataset_record: Dataset,
        file_ids: list[UUID],
        source_type: str | None = None,
    ) -> IDatasetRead:
        """Build response for deleted dataset."""
        return IDatasetRead(
            id=dataset_record.id,
            name=dataset_record.name,
            description=dataset_record.description,
            file_ids=file_ids,
            r2r_collection_id=dataset_record.r2r_collection_id,
            source_id=dataset_record.source_id,
            source_type=source_type,
            knowledge_graph=dataset_record.knowledge_graph,
            graph_build_phase=dataset_record.graph_build_phase,
            graph_build_requested_at=dataset_record.graph_build_requested_at,
            graph_build_completed_at=dataset_record.graph_build_completed_at,
            last_extraction_check_at=dataset_record.last_extraction_check_at,
        )

    async def file_belongs_to_dataset_check(
        self, *, dataset_id: UUID, file_id: UUID, db_session: AsyncSession | None = None
    ):
        db_session = db_session or super().get_db().session
        query = select(DatasetFileLink).where(
            DatasetFileLink.dataset_id == dataset_id, DatasetFileLink.file_id == file_id
        )
        result = await db_session.execute(query)
        link = result.scalar_one_or_none()

        if not link:
            raise HTTPException(
                status_code=404,
                detail=f"File ID {file_id} does not belong to Dataset ID {dataset_id}",
            )

    async def get_ingestion_id(
        self, *, dataset_id: UUID, file_id: UUID, db_session: AsyncSession | None = None
    ) -> UUID:
        db_session = db_session or super().get_db().session
        query = select(FileIngestion.ingestion_id).where(
            FileIngestion.file_id == file_id, FileIngestion.dataset_id == dataset_id
        )
        result = await db_session.execute(query)
        ingestion_id = result.scalar_one_or_none()
        return ingestion_id

    async def is_dataset_chat_ready(
        self,
        *,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> bool:
        db_session = db_session or super().get_db().session
        statement = await db_session.execute(
            select(FileIngestion).where(
                FileIngestion.dataset_id == dataset_id,
                FileIngestion.status == IngestionStatusEnum.Success,
                FileIngestion.deleted_at.is_(None),
            )
        )
        records = statement.scalars().all()
        return bool(records)

    async def is_dataset_chat_ready_v2(
        self,
        *,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> bool:
        db_session = db_session or super().get_db().session

        stmt = select(
            exists().where(
                Document.dataset_id == dataset_id,
                Document.deleted_at.is_(None),
                Document.processing_status != DocumentProcessingStatusEnum.Success,
            )
        )

        result = await db_session.execute(stmt)
        has_non_success = result.scalar()

        return not has_non_success

    async def get_datasets(
        self,
        *,
        workspace_id: UUID,
        ingested: Optional[bool] = None,
        type: Optional[str] = None,
        params: OptionalParams = OptionalParams(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Union[Page[Dataset], list[IDatasetRead]]:
        db_session = db_session or self.get_db().session
        columns = Dataset.__table__.columns

        # Validate dataset_type if provided
        if type and type not in {"sql", "unstructured"}:
            raise ValueError("Invalid dataset type. Must be 'sql' or 'unstructured'.")

        if order_by not in columns:
            order_by = "created_at"

        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = (
            select(Dataset)
            .where(
                and_(
                    Dataset.workspace_id == workspace_id,
                    Dataset.deleted_at.is_(None),  # Exclude soft-deleted records
                )
            )
            .order_by(order_clause)
        )

        # Only join and filter for SQL datasets
        if type == "sql":
            query = query.join(Source, Dataset.source_id == Source.id).where(
                or_(Source.source_type == "pg_db", Source.source_type == "mysql_db")
            )

        results = await db_session.execute(query)
        datasets = results.scalars().all()

        dataset_list: list[IDatasetRead] = []

        for dataset in datasets:
            # Get source_type if source_id exists
            source_type = None
            if dataset.source_id:
                source_result = await db_session.execute(
                    select(Source.source_type).where(Source.id == dataset.source_id)
                )
                source_type = source_result.scalar_one_or_none()

            # If dataset_type is unstructured, manually skip SQL datasets
            if type == "unstructured":
                if source_type and source_type in ["pg_db", "mysql_db"]:
                    logger.debug(
                        f"Filtered out SQL dataset from source type: {source_type}"
                    )
                    continue

            # Filter ingested datasets
            if ingested:
                is_ready = await self.is_dataset_chat_ready_v2(dataset_id=dataset.id)
                if not is_ready:
                    continue

            # Collect file IDs
            file_links = await db_session.execute(
                select(DatasetFileLink.file_id).where(
                    DatasetFileLink.dataset_id == dataset.id
                )
            )
            file_ids = list(file_links.scalars())

            # Get graph status
            graph_status = await self._get_graph_status_for_dataset(
                dataset_id=dataset.id, db_session=db_session
            )

            dataset_list.append(
                IDatasetRead(
                    id=dataset.id,
                    name=dataset.name,
                    description=dataset.description,
                    file_ids=file_ids,
                    source_id=dataset.source_id,
                    source_type=source_type,
                    r2r_collection_id=dataset.r2r_collection_id,
                    knowledge_graph=dataset.knowledge_graph,
                    graph_build_phase=dataset.graph_build_phase,
                    graph_build_requested_at=dataset.graph_build_requested_at,
                    graph_build_completed_at=dataset.graph_build_completed_at,
                    last_extraction_check_at=dataset.last_extraction_check_at,
                    graph_status=graph_status,
                )
            )

        if params.page is None or params.size is None:
            return dataset_list  # No pagination

        raw_params = Params(page=params.page, size=params.size)
        return paginate(dataset_list, raw_params)

    async def get_datasets_by_organization(
        self,
        *,
        organization_id: UUID,
        params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[IDatasetRead]:
        db_session = db_session or super().get_db().session
        columns = Dataset.__table__.columns

        if order_by not in columns:
            order_by = "created_at"

        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = (
            select(Dataset)
            .join(Workspace, Dataset.workspace_id == Workspace.id)
            .where(
                and_(
                    Workspace.organization_id == organization_id,
                    Dataset.deleted_at.is_(None),  # Exclude soft-deleted records
                )
            )
            .order_by(order_clause)
        )

        results = await db_session.execute(query)
        datasets = results.scalars().all()

        dataset_list = []
        for dataset in datasets:
            # Get source_type if source_id exists
            source_type = None
            if dataset.source_id:
                source_result = await db_session.execute(
                    select(Source.source_type).where(Source.id == dataset.source_id)
                )
                source_type = source_result.scalar_one_or_none()

            file_links = await db_session.execute(
                select(DatasetFileLink.file_id).where(
                    DatasetFileLink.dataset_id == dataset.id
                )
            )
            file_ids = list(file_links.scalars())

            # Get graph status
            graph_status = await self._get_graph_status_for_dataset(
                dataset_id=dataset.id, db_session=db_session
            )

            dataset_data = IDatasetRead(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                file_ids=file_ids,
                source_id=dataset.source_id,
                source_type=source_type,
                r2r_collection_id=dataset.r2r_collection_id,
                knowledge_graph=dataset.knowledge_graph,
                graph_build_phase=dataset.graph_build_phase,
                graph_build_requested_at=dataset.graph_build_requested_at,
                graph_build_completed_at=dataset.graph_build_completed_at,
                last_extraction_check_at=dataset.last_extraction_check_at,
                graph_status=graph_status,
            )
            dataset_list.append(dataset_data)

        return paginate(dataset_list, params)

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

    def get_dataset_files(self, *, dataset_id, db):
        """Get all file IDs associated with a dataset"""

        file_ids_result = db.execute(
            select(DatasetFileLink.file_id).where(
                DatasetFileLink.dataset_id == dataset_id
            )
        )
        return [row.file_id for row in file_ids_result]

    async def is_image_file(self, *, mimetype: str, filename: str) -> bool:
        """
        Asynchronously checks if a given file is an image based on its mimetype or extension.
        """
        # Check mimetype first (e.g., "image/jpeg", "image/png")
        if mimetype and mimetype.startswith("image/"):
            # PRD specifies support for jpg, jpeg, and png
            if any(fmt in mimetype for fmt in ["jpeg", "jpg", "png"]):
                return True

        # Fallback to extension check
        if filename:
            ext = filename.lower().split(".")[-1] if "." in filename else ""
            if ext in ["jpg", "jpeg", "png"]:
                return True

        return False

    async def is_image_or_audio_file(self, *, mimetype: str, filename: str) -> bool:

        image_mimetypes = ["jpeg", "jpg", "png"]
        audio_mimetypes = ["wav", "mp3", "aac"]

        # Check mimetype
        if mimetype:
            if mimetype.startswith("image/") and any(
                fmt in mimetype for fmt in image_mimetypes
            ):
                return True
            if mimetype.startswith("audio/") and any(
                fmt in mimetype for fmt in audio_mimetypes
            ):
                return True

        # Fallback to extension check
        if filename:
            ext = filename.lower().split(".")[-1] if "." in filename else ""
            if ext in image_mimetypes or ext in audio_mimetypes:
                return True

        return False

    async def get_datasets_by_file_type(
        self, *, dataset_ids: List[UUID], db_session: AsyncSession | None = None
    ) -> DatasetFileTypeGroup:
        db_session = db_session or super().get_db().session
        text_datasets = []
        image_datasets = []
        mixed_datasets = []

        stmt = select(Dataset).where(Dataset.id.in_(dataset_ids))
        result = await db_session.execute(stmt)
        datasets = result.scalars().all()

        for dataset in datasets:
            # Assuming relationship "files" is lazy-loaded; you can change this to explicit join if eager
            await db_session.refresh(dataset, attribute_names=["files"])

            image_files = []
            text_files = []

            for file in dataset.files:
                if await self.is_image_or_audio_file(
                    mimetype=file.mimetype, filename=file.filename
                ):
                    image_files.append(file)
                else:
                    text_files.append(file)

            if image_files and text_files:
                mixed_datasets.append(dataset.id)
            elif image_files:
                image_datasets.append(dataset.id)
            elif text_files:
                text_datasets.append(dataset.id)

        return DatasetFileTypeGroup(
            text_datasets=text_datasets,
            image_datasets=image_datasets,
            mixed_datasets=mixed_datasets,
        )

    def update_extraction_check_time(self, *, dataset_id):
        """Update the last extraction check time"""
        with SyncSessionLocal() as db:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                dataset.last_extraction_check_at = datetime.now(UTC)
                db.commit()

    async def _validate_source(
        self,
        source_id: UUID | None,
        db_session: AsyncSession,
        operation: str = "creation",
    ) -> None:
        """Validate that the source has successfully pulled files."""
        if not source_id:
            return

        pull_status_record = await db_session.execute(
            select(SourcePullStatus).where(SourcePullStatus.source_id == source_id)
        )
        pull_status_record = pull_status_record.scalars().first()

        if (
            not pull_status_record
            or pull_status_record.pull_status != PullStatusEnum.SUCCESS
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset {operation} failed: Files not pulled completely for source {source_id}. Current status: {pull_status_record.pull_status if pull_status_record else 'Not found'}",
            )

        return pull_status_record

    async def _link_files(
        self, dataset_id: UUID, file_ids: List[UUID], db_session: AsyncSession
    ) -> None:
        """Link files to the dataset."""
        if not file_ids:
            return

        for file_id in file_ids:
            file_record = await db_session.get(File, file_id)
            if not file_record:
                raise HTTPException(
                    status_code=400, detail=f"File with ID {file_id} not found"
                )
            link = DatasetFileLink(dataset_id=dataset_id, file_id=file_id)
            db_session.add(link)
        await db_session.commit()

    async def _get_source_files(
        self, source_id: UUID, db_session: AsyncSession
    ) -> List[UUID]:
        """Get all file IDs associated with a source."""
        records = await db_session.execute(
            select(File.id).where(
                File.source_id == source_id,
                File.status == FileStatusEnum.Uploaded,
            )
        )
        source_file_ids = list(records.scalars().all())

        if not source_file_ids:
            raise HTTPException(
                status_code=404,
                detail=f"No files present in source_id {source_id}",
            )

        return source_file_ids

    async def _process_file_changes(
        self,
        dataset_id: UUID,
        current_file_ids: set[UUID],
        new_file_ids: set[UUID],
        db_session: AsyncSession,
    ) -> tuple[list[str], list[str]]:
        """Process file additions and deletions."""
        # Determine which files to delete
        file_ids_to_delete = current_file_ids - new_file_ids
        r2r_ids_to_delete = []
        file_ids_removed = []

        # Process deletions
        for file_id in file_ids_to_delete:
            r2r_id = await self._get_file_r2r_id(dataset_id, file_id, db_session)
            was_successfully_ingested = await self._check_ingestion_success(
                dataset_id, file_id, db_session
            )

            # Handle file splits and get split R2R IDs
            split_r2r_ids, has_splits = await self._process_file_splits(
                dataset_id, file_id, was_successfully_ingested, db_session
            )

            # Collect R2R IDs to delete based on ingestion status and splits
            self._collect_r2r_ids_for_deletion(
                r2r_id,
                was_successfully_ingested,
                has_splits,
                split_r2r_ids,
                r2r_ids_to_delete,
                file_ids_removed,
                file_id,
            )

            # Delete dataset-file link and mark ingestion as deleted
            await self._remove_file_links(dataset_id, file_id, db_session)

            # Mark splits as deleted if they exist
            if has_splits:
                await self._mark_splits_deleted(dataset_id, file_id, db_session)

        # Process additions
        await self._add_new_files(
            dataset_id, current_file_ids, new_file_ids, db_session
        )

        await db_session.commit()
        return r2r_ids_to_delete, file_ids_removed

    async def _get_file_r2r_id(
        self, dataset_id: UUID, file_id: UUID, db_session: AsyncSession
    ) -> str | None:
        """Get R2R ID for a file in a dataset."""
        r2r_stmt = select(DatasetFileLink.r2r_id).where(
            DatasetFileLink.dataset_id == dataset_id,
            DatasetFileLink.file_id == file_id,
        )
        r2r_result = await db_session.execute(r2r_stmt)
        return r2r_result.scalar_one_or_none()

    async def _check_ingestion_success(
        self, dataset_id: UUID, file_id: UUID, db_session: AsyncSession
    ) -> bool:
        """Check if a file was successfully ingested in a dataset."""
        ingestion_stmt = select(FileIngestion).where(
            FileIngestion.dataset_id == dataset_id,
            FileIngestion.file_id == file_id,
            FileIngestion.status == FileIngestionStatusType.Success,
            FileIngestion.deleted_at.is_(None),
        )
        ingestion_result = await db_session.execute(ingestion_stmt)
        ingestion_record = ingestion_result.scalar_one_or_none()
        return ingestion_record is not None

    async def _process_file_splits(
        self,
        dataset_id: UUID,
        file_id: UUID,
        was_successfully_ingested: bool,
        db_session: AsyncSession,
    ) -> tuple[list[str], bool]:
        """Process file splits and return split R2R IDs and whether the file has splits."""
        split_r2r_ids = []
        has_splits = False

        if was_successfully_ingested:
            from app.models.file_split_model import FileSplit

            splits_stmt = select(FileSplit).where(
                FileSplit.original_file_id == file_id,
                FileSplit.dataset_id == dataset_id,
            )
            splits_result = await db_session.execute(splits_stmt)
            splits = splits_result.scalars().all()

            has_splits = len(splits) > 0
            if has_splits:
                logger.info(
                    f"File {file_id} has {len(splits)} splits that will also be removed"
                )

                # Generate R2R IDs for each split
                from uuid import NAMESPACE_URL, uuid5

                for split in splits:
                    split_namespace = f"{file_id}_{split.id}_{dataset_id}"
                    split_doc_id = str(uuid5(NAMESPACE_URL, split_namespace))
                    split_r2r_ids.append(split_doc_id)
                    logger.debug(
                        f"Adding split {split.id} with R2R ID {split_doc_id} to deletion list"
                    )

        return split_r2r_ids, has_splits

    def _collect_r2r_ids_for_deletion(
        self,
        r2r_id: str | None,
        was_successfully_ingested: bool,
        has_splits: bool,
        split_r2r_ids: list[str],
        r2r_ids_to_delete: list[str],
        file_ids_removed: list[str],
        file_id: UUID,
    ):
        """Collect R2R IDs to delete based on ingestion status and splits."""
        if not was_successfully_ingested:
            logger.info(
                f"File {file_id} was not successfully ingested, skipping R2R deletion"
            )
            return

        # Add file ID to removed list
        file_ids_removed.append(str(file_id))

        # Handle original file R2R ID
        if r2r_id and not has_splits:
            logger.info(
                f"Adding original file {file_id} with R2R ID {r2r_id} to deletion list"
            )
            r2r_ids_to_delete.append(str(r2r_id))
        elif has_splits:
            logger.info(
                f"File {file_id} has splits - skipping deletion of original R2R ID"
            )

        # Add split R2R IDs if the file has splits
        if has_splits and split_r2r_ids:
            r2r_ids_to_delete.extend(split_r2r_ids)

    async def _remove_file_links(
        self, dataset_id: UUID, file_id: UUID, db_session: AsyncSession
    ) -> None:
        """Remove dataset-file link and mark ingestion as deleted."""
        # Delete the dataset-file link
        link_to_remove = await db_session.execute(
            select(DatasetFileLink).where(
                DatasetFileLink.dataset_id == dataset_id,
                DatasetFileLink.file_id == file_id,
            )
        )
        link_instance = link_to_remove.scalars().first()
        if link_instance:
            await db_session.delete(link_instance)

        # Mark ingestion record as deleted
        ingestion_record_to_delete = await db_session.execute(
            select(FileIngestion).where(
                FileIngestion.dataset_id == dataset_id,
                FileIngestion.file_id == file_id,
            )
        )
        ingestion_instance = ingestion_record_to_delete.scalars().first()

        if ingestion_instance:
            ingestion_instance.deleted_at = datetime.utcnow()
            db_session.add(ingestion_instance)

    async def _mark_splits_deleted(
        self, dataset_id: UUID, file_id: UUID, db_session: AsyncSession
    ) -> None:
        """Mark file splits as deleted."""
        from app.models.file_split_model import FileSplit

        # Fetch all splits for this file in this dataset
        update_splits_stmt = select(FileSplit).where(
            FileSplit.original_file_id == file_id,
            FileSplit.dataset_id == dataset_id,
        )
        update_splits_result = await db_session.execute(update_splits_stmt)
        splits_to_delete = update_splits_result.scalars().all()

        # Mark each split as deleted
        for split in splits_to_delete:
            split.deleted_at = datetime.utcnow()
            db_session.add(split)

        logger.info(
            f"Marked {len(splits_to_delete)} splits for file {file_id} as deleted"
        )

    async def _add_new_files(
        self,
        dataset_id: UUID,
        current_file_ids: set[UUID],
        new_file_ids: set[UUID],
        db_session: AsyncSession,
    ) -> None:
        """Add new files to the dataset."""
        files_to_add = new_file_ids - current_file_ids
        for file_id in files_to_add:
            new_link = DatasetFileLink(dataset_id=dataset_id, file_id=file_id)
            db_session.add(new_link)

    async def _reset_knowledge_graph(
        self,
        dataset_record: Dataset,
        organization_id: UUID | None,
        db_session: AsyncSession,
        reason: str,
    ) -> None:
        """Reset the knowledge graph for a dataset."""
        if not dataset_record.knowledge_graph:
            return

        logger.info(
            f"Dataset {dataset_record.id} {reason}, resetting knowledge graph status"
        )
        dataset_record.knowledge_graph = False
        dataset_record.graph_build_phase = None
        dataset_record.graph_build_requested_at = None
        dataset_record.graph_build_completed_at = None
        dataset_record.last_extraction_check_at = None
        await db_session.commit()

        # Reset R2R graph if it exists
        if dataset_record.r2r_collection_id and organization_id:
            try:
                from app.api.deps import get_r2r_client_sync

                client = get_r2r_client_sync(organization_id=organization_id)
                client.graphs.reset(collection_id=dataset_record.r2r_collection_id)
                logger.info(
                    f"Reset knowledge graph for collection {dataset_record.r2r_collection_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to reset knowledge graph in R2R: {e}")

    async def _create_default_configs(self, dataset_id: UUID) -> None:
        """Create default configurations for the dataset."""
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

    async def _validate_update_inputs(self, obj_in: IDatasetUpdate):
        """Validate that the update inputs are valid."""
        if obj_in.file_ids and obj_in.file_ids != [] and obj_in.source_id is not None:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both non-empty file_ids and source_id. Please provide only one of them.",
            )

    async def _get_dataset_for_update(
        self, dataset_id: UUID, workspace_id: UUID, db_session: AsyncSession
    ):
        """Get dataset record and check if ingestion is in progress."""
        # Get dataset record
        dataset_records = await db_session.execute(
            select(Dataset).where(
                Dataset.id == dataset_id,
                Dataset.workspace_id == workspace_id,
                Dataset.deleted_at.is_(None),
            )
        )
        dataset_record = dataset_records.scalars().first()
        if not dataset_record:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Check if ingestion is in progress
        ingestion_started = await ingestion_crud.dataset_ingestion_started_check(
            dataset_id=dataset_id
        )
        if ingestion_started:
            dataset_ingestion_complete = (
                await ingestion_crud.dataset_ingestion_complete_check(
                    dataset_id=dataset_id
                )
            )
            if not dataset_ingestion_complete:
                raise HTTPException(
                    status_code=409,
                    detail="Please wait till the ingestion for dataset is finished.",
                )

        return dataset_record

    async def _get_current_file_ids(self, dataset_id: UUID, db_session: AsyncSession):
        """Get the current file IDs for the dataset."""
        current_links = await db_session.execute(
            select(DatasetFileLink).where(DatasetFileLink.dataset_id == dataset_id)
        )
        return {link.file_id for link in current_links.scalars()}

    def _schedule_r2r_deletions(
        self, r2r_ids_to_delete, file_ids_removed, dataset_id, dataset_record
    ):
        """Schedule R2R deletions."""
        logger.info(f"Scheduling deletion of {len(r2r_ids_to_delete)} files from R2R")
        celery.signature(
            "tasks.delete_files_task",
            kwargs={
                "document_ids_to_delete": r2r_ids_to_delete,
                "file_ids": file_ids_removed,
                "dataset_id": str(dataset_id),
                "organization_id": dataset_record.workspace.organization_id,
            },
        ).apply_async()

    def _create_dataset_response(
        self, dataset_record, new_file_ids, source_type: str | None = None
    ):
        """Create response with updated data."""
        return IDatasetRead(
            id=dataset_record.id,
            name=dataset_record.name,
            description=dataset_record.description,
            file_ids=list(new_file_ids),
            source_id=dataset_record.source_id,
            source_type=source_type,
            r2r_collection_id=dataset_record.r2r_collection_id,
            knowledge_graph=dataset_record.knowledge_graph,
            graph_build_phase=dataset_record.graph_build_phase,
            graph_build_requested_at=dataset_record.graph_build_requested_at,
            graph_build_completed_at=dataset_record.graph_build_completed_at,
            last_extraction_check_at=dataset_record.last_extraction_check_at,
        )

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


dataset = CRUDDataset(Dataset)
