"""
CRUD operations for file splits in version 2, using Document model instead of FileIngestion.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.be_core.logger import logger
from app.models.file_split_model import FileSplit, SplitFileStatusType


class FileSplitCRUDV2:
    """CRUD operations for file splits with Document model integration."""

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
        self,
        db: Session,
        file_id: UUID,
        document_id: Optional[UUID] = None,
        dataset_id: Optional[UUID] = None,
    ) -> List[FileSplit]:
        """
        Get all splits for a file.

        Args:
            db: Database session
            file_id: ID of the file
            document_id: Optional ID of the document to filter by
            dataset_id: Optional ID of the dataset to filter by

        Returns:
            List of FileSplit objects
        """
        query = db.query(FileSplit).filter(FileSplit.original_file_id == file_id)

        if document_id:
            query = query.filter(FileSplit.document_id == document_id)

        if dataset_id:
            query = query.filter(FileSplit.dataset_id == dataset_id)

        return query.order_by(FileSplit.split_index).all()

    def get_splits_for_document(
        self, db: Session, document_id: UUID
    ) -> List[FileSplit]:
        """
        Get all splits for a document.

        Args:
            db: Database session
            document_id: ID of the document

        Returns:
            List of FileSplit objects
        """
        return (
            db.query(FileSplit)
            .filter(FileSplit.document_id == document_id)
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
        document_id: UUID,
        dataset_id: UUID,
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
            document_id: ID of the document
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
        # Create the split with document_id
        split = FileSplit(
            original_file_id=original_file_id,
            document_id=document_id,
            dataset_id=dataset_id,
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


# Create a singleton instance for global use
file_split_crud = FileSplitCRUDV2()
