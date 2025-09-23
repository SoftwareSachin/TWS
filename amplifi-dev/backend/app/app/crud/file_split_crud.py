"""
CRUD operations for file splits.
"""

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, update
from sqlalchemy.orm import Session

from app.be_core.logger import logger
from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType
from app.models.file_model import File
from app.models.file_split_model import FileSplit, SplitFileStatusType


class FileSplitCRUD:
    """CRUD operations for file splits."""

    def get_by_id(self, db: Session, split_id: UUID) -> Optional[FileSplit]:
        """
        Get a file split by ID.

        Args:
            db: Database session
            split_id: ID of the split to retrieve

        Returns:
            FileSplit object if found, None otherwise
        """
        return db.query(FileSplit).filter(FileSplit.id == split_id).first()

    def get_splits_for_file(
        self, db: Session, file_id: UUID, file_ingestion_id: Optional[UUID] = None
    ) -> List[FileSplit]:
        """
        Get all splits for a file.

        Args:
            db: Database session
            file_id: ID of the file
            file_ingestion_id: Optional ID of the file ingestion to filter by

        Returns:
            List of FileSplit objects
        """
        query = db.query(FileSplit).filter(FileSplit.original_file_id == file_id)

        if file_ingestion_id:
            query = query.filter(FileSplit.file_ingestion_id == file_ingestion_id)

        return query.order_by(FileSplit.split_index).all()

    def get_splits_for_ingestion(
        self, db: Session, file_ingestion_id: UUID
    ) -> List[FileSplit]:
        """
        Get all splits for a file ingestion.

        Args:
            db: Database session
            file_ingestion_id: ID of the file ingestion

        Returns:
            List of FileSplit objects
        """
        return (
            db.query(FileSplit)
            .filter(FileSplit.file_ingestion_id == file_ingestion_id)
            .order_by(FileSplit.split_index)
            .all()
        )

    def update_split_status(
        self,
        db: Session,
        split_id: UUID,
        status: SplitFileStatusType,
        task_id: Optional[str] = None,
    ) -> Optional[FileSplit]:
        """
        Update the status of a file split.

        Args:
            db: Database session
            split_id: ID of the split to update
            status: New status
            task_id: Optional task ID to update

        Returns:
            Updated FileSplit object if found, None otherwise
        """
        split = self.get_by_id(db, split_id)
        if not split:
            return None

        # Update status and task_id if provided
        split.status = status
        if task_id:
            split.task_id = task_id

        # Set finished_at timestamp if reaching a terminal status
        if status in [
            SplitFileStatusType.Success,
            SplitFileStatusType.Failed,
            SplitFileStatusType.Exception,
        ]:
            from datetime import datetime

            split.finished_at = datetime.now()

        db.add(split)
        db.commit()
        db.refresh(split)

        return split

    def create_split(
        self,
        db: Session,
        original_file_id: UUID,
        file_ingestion_id: UUID,
        dataset_id: UUID,  # Added dataset_id parameter
        split_file_path: str,
        split_index: int,
        total_splits: int,
        size: int,
        token_count: int,
        config_hash: Optional[str] = None,
    ) -> FileSplit:
        """
        Create a new file split record.

        Args:
            db: Database session
            original_file_id: ID of the original file
            file_ingestion_id: ID of the file ingestion
            dataset_id: ID of the dataset this split belongs to
            split_file_path: Path to the split file
            split_index: Index of this split
            total_splits: Total number of splits
            size: Size of the split in bytes
            token_count: Estimated token count
            config_hash: Hash of the configuration used to create this split

        Returns:
            Created FileSplit object
        """
        # Create the split with dataset_id
        split = FileSplit(
            original_file_id=original_file_id,
            dataset_id=dataset_id,  # Include dataset_id
            file_ingestion_id=file_ingestion_id,
            split_file_path=split_file_path,
            split_index=split_index,
            total_splits=total_splits,
            size=size,
            token_count=token_count,
            status=SplitFileStatusType.Not_Started,
            config_hash=config_hash,
        )
        db.add(split)
        db.flush()  # Flush to get the ID without committing transaction
        return split

    def update_parent_ingestion_status(
        self, db: Session, file_ingestion_id: UUID
    ) -> Optional[FileIngestionStatusType]:
        """
        Update the status of the parent file ingestion based on its splits.
        Only considers an ingestion successful if ALL splits are successfully ingested.

        Args:
            db: Database session
            file_ingestion_id: ID of the file ingestion

        Returns:
            The new status of the parent file ingestion, or None if the ingestion wasn't found
        """
        # Get the file ingestion
        file_ingestion = (
            db.query(FileIngestion)
            .filter(FileIngestion.id == file_ingestion_id)
            .first()
        )

        if not file_ingestion:
            logger.error(f"No file ingestion record found with ID {file_ingestion_id}")
            return None

        # Get all splits for this ingestion
        splits = self.get_splits_for_ingestion(db, file_ingestion_id)
        total_splits = len(splits)

        if total_splits == 0:
            logger.warning(f"No splits found for file ingestion {file_ingestion_id}")
            return None

        # Count splits by status
        successful_splits = sum(
            1 for s in splits if s.status == SplitFileStatusType.Success
        )
        failed_splits = sum(
            1
            for s in splits
            if s.status in [SplitFileStatusType.Failed, SplitFileStatusType.Exception]
        )
        processing_splits = total_splits - successful_splits - failed_splits

        # Update the counts
        file_ingestion.successful_splits_count = successful_splits
        file_ingestion.total_splits_count = total_splits

        # Calculate the new status
        new_status = None

        if processing_splits > 0:
            # Still processing
            new_status = FileIngestionStatusType.Processing
        elif successful_splits == total_splits:
            # All splits successful
            new_status = FileIngestionStatusType.Success
            from datetime import datetime

            file_ingestion.finished_at = datetime.now()

            # Update the original file too
            file = db.query(File).filter(File.id == file_ingestion.file_id).first()
            if file:
                file.requires_splitting = True  # Mark that this file was split
        else:
            # Some or all splits failed - consider the entire ingestion as failed
            new_status = FileIngestionStatusType.Failed
            from datetime import datetime

            file_ingestion.finished_at = datetime.now()

            # Log a clear message about the failure reason
            logger.warning(
                f"Marking ingestion {file_ingestion_id} as FAILED: "
                f"Only {successful_splits}/{total_splits} splits were successfully ingested, "
                f"but ALL splits must succeed for the ingestion to be considered successful"
            )

        # Update the status
        file_ingestion.status = new_status
        db.commit()

        logger.info(
            f"Updated file ingestion {file_ingestion_id} status to {new_status.value} "
            f"({successful_splits}/{total_splits} splits successful)"
        )

        return new_status

    def clean_existing_splits(
        self,
        db: Session,
        file_id: UUID,
        delete_files: bool = True,
        dataset_id: Optional[UUID] = None,
    ) -> Tuple[int, int]:
        """
        Remove existing splits for a file from the database, filesystem, and R2R.
        If dataset_id is provided, only clean splits for that specific dataset.

        Args:
            db: Database session
            file_id: ID of the original file
            delete_files: Whether to delete the physical split files
            dataset_id: Optional ID of the dataset to restrict cleanup to

        Returns:
            Tuple containing (number of DB records deleted, number of files deleted)
        """
        # Find relevant splits
        splits = self._find_splits_to_clean(db, file_id, dataset_id)

        if not splits:
            dataset_info = (
                f" in dataset {dataset_id}" if dataset_id else " in any dataset"
            )
            logger.debug(f"No existing splits found for file {file_id}{dataset_info}")
            return 0, 0

        # Delete documents from R2R
        r2r_delete_count = self._delete_splits_from_r2r(splits, file_id)

        # Delete physical files if requested
        file_delete_count = 0
        if delete_files:
            file_delete_count = self._delete_split_files(splits)

        # Delete database records
        db_delete_count = self._delete_split_records(db, file_id, dataset_id)

        # Log cleanup summary
        dataset_info = f" in dataset {dataset_id}" if dataset_id else ""
        logger.info(
            f"Cleaned up {db_delete_count} DB records, {file_delete_count} files, "
            f"and {r2r_delete_count} R2R documents for existing splits of file {file_id}{dataset_info}"
        )

        return db_delete_count, file_delete_count

    def _find_splits_to_clean(
        self, db: Session, file_id: UUID, dataset_id: Optional[UUID] = None
    ) -> List[FileSplit]:
        """Find all splits that need to be cleaned based on file_id and optional dataset_id."""
        if dataset_id:
            splits = (
                db.query(FileSplit)
                .filter(
                    FileSplit.original_file_id == file_id,
                    FileSplit.dataset_id == dataset_id,
                )
                .all()
            )
            logger.debug(f"Finding splits for file {file_id} in dataset {dataset_id}")
        else:
            splits = self.get_splits_for_file(db, file_id)
            logger.debug(f"Finding splits for file {file_id} across all datasets")

        return splits

    def _delete_splits_from_r2r(self, splits: List[FileSplit], file_id: UUID) -> int:
        """Delete split documents from R2R and return count of deleted documents."""
        from uuid import NAMESPACE_URL, uuid5

        from app.api.deps import get_r2r_client_sync

        r2r_delete_count = 0

        try:
            # Get R2R client
            r2r_client = get_r2r_client_sync()

            # Delete each split document
            for split in splits:
                # Create deterministic UUID for this split using UUID v5
                split_namespace = f"{file_id}_{split.id}_{split.dataset_id}"
                doc_id = str(uuid5(NAMESPACE_URL, split_namespace))

                try:
                    # Try to delete the document from R2R
                    logger.debug(
                        f"Attempting to delete R2R document {doc_id} for split {split.id} in dataset {split.dataset_id}"
                    )
                    r2r_client.documents.delete(id=doc_id)
                    r2r_delete_count += 1
                    logger.info(
                        f"Successfully deleted R2R document {doc_id} for split {split.id}"
                    )
                except Exception as e:
                    # Document might not exist in R2R, which is fine
                    logger.debug(f"Could not delete R2R document {doc_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting splits from R2R: {str(e)}")
            # Continue with cleanup even if R2R deletion fails

        return r2r_delete_count

    def _delete_split_files(self, splits: List[FileSplit]) -> int:
        """Delete physical split files and return count of deleted files."""
        import os

        file_delete_count = 0

        for split in splits:
            split_path = split.split_file_path
            try:
                if os.path.exists(split_path):
                    os.remove(split_path)
                    logger.debug(f"Removed existing split file: {split_path}")
                    file_delete_count += 1
                else:
                    logger.warning(f"Existing split file not found: {split_path}")
            except Exception as e:
                logger.error(
                    f"Error removing existing split file {split_path}: {str(e)}"
                )
                # Continue with cleanup even if one file fails

        return file_delete_count

    def _delete_split_records(
        self, db: Session, file_id: UUID, dataset_id: Optional[UUID] = None
    ) -> int:
        """Delete split records from database and return count of deleted records."""
        query = db.query(FileSplit).filter(FileSplit.original_file_id == file_id)

        if dataset_id:
            query = query.filter(FileSplit.dataset_id == dataset_id)

        db_delete_count = query.delete(synchronize_session=False)
        db.commit()

        return db_delete_count

    def get_splits_by_status(
        self, db: Session, statuses: List[SplitFileStatusType], limit: int = 100
    ) -> List[FileSplit]:
        """
        Get splits with specific statuses.

        Args:
            db: Database session
            statuses: List of statuses to filter by
            limit: Maximum number of splits to return

        Returns:
            List of FileSplit objects
        """
        return (
            db.query(FileSplit)
            .filter(FileSplit.status.in_(statuses))
            .order_by(FileSplit.created_at)
            .limit(limit)
            .all()
        )

    def bulk_update_status_for_ingestion(
        self,
        db: Session,
        file_ingestion_id: UUID,
        from_status: SplitFileStatusType,
        to_status: SplitFileStatusType,
    ) -> int:
        """
        Bulk update the status of all splits for a file ingestion that have a specific status.

        Args:
            db: Database session
            file_ingestion_id: ID of the file ingestion
            from_status: Current status to match
            to_status: New status to set

        Returns:
            Number of updated records
        """
        stmt = (
            update(FileSplit)
            .where(
                and_(
                    FileSplit.file_ingestion_id == file_ingestion_id,
                    FileSplit.status == from_status,
                )
            )
            .values(status=to_status)
        )

        result = db.execute(stmt)
        db.commit()

        return result.rowcount


# Create a singleton instance for global use
file_split_crud = FileSplitCRUD()
