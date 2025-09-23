import asyncio
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import and_, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlmodel import select

from app import crud
from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.db.session import SyncSessionLocal
from app.models import ChunkingConfig
from app.models.dataset_file_link_model import DatasetFileLink
from app.models.dataset_model import Dataset
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.models.file_model import File, FileStatusEnum
from app.models.source_model import Source
from app.schemas.chunking_config_schema import IChunkingMethodEnum
from app.schemas.response_schema import (
    IGetResponseBase,
    IIngestFilesOperationRead,
    create_response,
)
from app.utils.datetime_utils import ensure_naive_datetime

# Configuration constants for batch processing
try:
    from app.be_core.config import settings

    DOCUMENT_BATCH_SIZE = getattr(
        settings, "DOCUMENT_BATCH_SIZE", 1000
    )  # Default batch size for document operations
    DOCUMENT_MAX_BATCH_SIZE = getattr(
        settings, "DOCUMENT_MAX_BATCH_SIZE", 5000
    )  # Maximum batch size
except ImportError:
    DOCUMENT_BATCH_SIZE = 1000  # Default batch size for document operations
    DOCUMENT_MAX_BATCH_SIZE = 5000  # Maximum batch size


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


def _get_document_type_from_file_path(file_path: str) -> DocumentTypeEnum:
    """
    Determine document type based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        DocumentTypeEnum corresponding to the file type
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    type_mapping = {
        ".pdf": DocumentTypeEnum.PDF,
        ".docx": DocumentTypeEnum.DOCX,
        ".xlsx": DocumentTypeEnum.XLSX,
        ".pptx": DocumentTypeEnum.PPTX,
        ".md": DocumentTypeEnum.Markdown,
        ".html": DocumentTypeEnum.HTML,
        ".htm": DocumentTypeEnum.HTML,
        ".csv": DocumentTypeEnum.CSV,
        # Image types
        ".jpg": DocumentTypeEnum.Image,
        ".jpeg": DocumentTypeEnum.Image,
        ".png": DocumentTypeEnum.Image,
        # Audio types
        ".wav": DocumentTypeEnum.Audio,
        ".mp3": DocumentTypeEnum.Audio,
        ".aac": DocumentTypeEnum.Audio,
        ".m4a": DocumentTypeEnum.Audio,
        # Video types
        ".mp4": DocumentTypeEnum.Video,
        ".avi": DocumentTypeEnum.Video,
        ".mov": DocumentTypeEnum.Video,
        ".wmv": DocumentTypeEnum.Video,
        ".flv": DocumentTypeEnum.Video,
        ".webm": DocumentTypeEnum.Video,
        ".mkv": DocumentTypeEnum.Video,
    }

    return type_mapping.get(extension, DocumentTypeEnum.PDF)  # Default to PDF


def _get_mime_type_from_extension(file_path: str) -> str:
    """
    Get MIME type based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    mime_mapping = {
        # Document types
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".csv": "text/csv",
        # Image types
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        # Audio types
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
    }

    return mime_mapping.get(extension, "application/octet-stream")


class CRUDIngestionV2(CRUDBase[Document, None, None]):
    async def get_chunking_config_for_dataset(
        self, *, dataset_id: UUID, db_session: AsyncSession | None = None
    ) -> Tuple[Optional[dict], Optional[UUID]]:
        db_session = db_session or super().get_db().session
        statement = select(ChunkingConfig).where(
            ChunkingConfig.dataset_id == dataset_id
        )
        record = await db_session.execute(statement)
        result = record.scalar_one_or_none()
        chunking_config_id = result.id
        chunking_configuration = await crud.chunking_config.get_chunking_config(
            chunking_config_id=chunking_config_id, dataset_id=dataset_id
        )
        chunking_config_in_dict = chunking_configuration.dict(exclude={"id", "name"})
        if "method" in chunking_config_in_dict:
            chunking_config_in_dict["chunking_strategy"] = chunking_config_in_dict.pop(
                "method"
            )
        if isinstance(
            chunking_config_in_dict.get("chunking_strategy"), IChunkingMethodEnum
        ):
            chunking_config_in_dict["chunking_strategy"] = chunking_config_in_dict[
                "chunking_strategy"
            ].value
        return chunking_config_in_dict, chunking_config_id

    def get_chunking_config_for_dataset_sync(
        self, *, dataset_id: UUID, db_session
    ) -> Tuple[Optional[dict], Optional[UUID]]:
        """
        Synchronous version of get_chunking_config_for_dataset.
        Follows the exact same logic as the async version.
        """
        statement = select(ChunkingConfig).where(
            ChunkingConfig.dataset_id == dataset_id
        )
        record = db_session.execute(statement)
        result = record.scalar_one_or_none()
        chunking_config_id = result.id
        chunking_configuration = crud.chunking_config.get_chunking_config_sync(
            chunking_config_id=chunking_config_id,
            dataset_id=dataset_id,
            db_session=db_session,
        )
        chunking_config_in_dict = chunking_configuration.dict(exclude={"id", "name"})
        if "method" in chunking_config_in_dict:
            chunking_config_in_dict["chunking_strategy"] = chunking_config_in_dict.pop(
                "method"
            )
        if isinstance(
            chunking_config_in_dict.get("chunking_strategy"), IChunkingMethodEnum
        ):
            chunking_config_in_dict["chunking_strategy"] = chunking_config_in_dict[
                "chunking_strategy"
            ].value
        return chunking_config_in_dict, chunking_config_id

    async def get_files_for_dataset(
        self, *, dataset_id: UUID, db_session: AsyncSession | None = None
    ) -> Sequence[File]:
        db_session = db_session or super().get_db().session

        # Get file IDs from dataset-file links
        statement = select(DatasetFileLink.file_id).where(
            DatasetFileLink.dataset_id == dataset_id
        )
        logger.info(f"Looking up files for dataset {dataset_id}")
        result = await db_session.execute(statement)
        file_ids = result.scalars().all()

        if not file_ids:
            logger.warning(
                f"No file links found for dataset {dataset_id}, checking if dataset has a source"
            )

            # Check if dataset has a source_id
            source_statement = select(Dataset.source_id).where(Dataset.id == dataset_id)
            source_result = await db_session.execute(source_statement)
            source_id = source_result.scalar_one_or_none()

            if source_id:
                # getting source_type
                source_type = None
                source_result = await db_session.execute(
                    select(Source.source_type).where(Source.id == source_id)
                )
                source_type = source_result.scalar_one_or_none()
                logger.info(
                    f"Fetched source_type: {source_type} for source_id: {source_id}"
                )

                logger.info(
                    f"Dataset {dataset_id} has source {source_id}, trying to get files from source"
                )

                # Try to get files from source
                try:
                    source_files_statement = select(File).where(
                        File.source_id == source_id,
                        File.status == FileStatusEnum.Uploaded,
                    )
                    source_files_result = await db_session.execute(
                        source_files_statement
                    )
                    source_files = source_files_result.scalars().all()

                    if source_files:
                        logger.info(
                            f"Found {len(source_files)} files from source {source_id}, linking them to dataset {dataset_id}"
                        )
                        # Link files to dataset
                        for file in source_files:
                            link = DatasetFileLink(
                                dataset_id=dataset_id, file_id=file.id
                            )
                            db_session.add(link)
                        await db_session.commit()
                        return source_files
                    else:
                        logger.warning(f"No files found for source {source_id}")
                except Exception as e:
                    logger.error(
                        f"Error getting files from source {source_id}: {str(e)}"
                    )

            # If we get here, we couldn't find or recover any files
            raise HTTPException(
                status_code=404, detail=f"No files found for dataset {dataset_id}."
            )

        # Get file objects for all file IDs
        files = []
        for file_id in file_ids:
            file_statement = select(File).where(File.id == file_id)
            file_result = await db_session.execute(file_statement)
            file_record = file_result.scalar_one_or_none()

            if file_record:
                files.append(file_record)
            else:
                logger.warning(f"File {file_id} from dataset {dataset_id} not found")

        if not files:
            logger.error(
                f"None of the file IDs linked to dataset {dataset_id} exist in the files table"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Files linked to dataset {dataset_id} not found in database.",
            )

        logger.info(f"Found {len(files)} files for dataset {dataset_id}")
        return files

    def get_files_for_dataset_sync(
        self, *, dataset_id: UUID, db_session: Session | None = None
    ) -> Sequence[File]:
        db_session = db_session or super().get_db().session

        # Get file IDs from dataset-file links
        statement = select(DatasetFileLink.file_id).where(
            DatasetFileLink.dataset_id == dataset_id
        )
        logger.info(f"Looking up files for dataset {dataset_id} (sync)")
        result = db_session.execute(statement)
        file_ids = result.scalars().all()

        if not file_ids:
            logger.warning(
                f"No file links found for dataset {dataset_id}, checking if dataset has a source (sync)"
            )

            # Check if dataset has a source_id
            source_statement = select(Dataset.source_id).where(Dataset.id == dataset_id)
            source_result = db_session.execute(source_statement)
            source_id = source_result.scalar_one_or_none()

            if source_id:
                logger.info(
                    f"Dataset {dataset_id} has source {source_id}, trying to get files from source (sync)"
                )
                # Try to get files from source
                try:
                    source_files_statement = select(File).where(
                        File.source_id == source_id,
                        File.status == FileStatusEnum.Uploaded,
                    )
                    source_files_result = db_session.execute(source_files_statement)
                    source_files = source_files_result.scalars().all()

                    if source_files:
                        logger.info(
                            f"Found {len(source_files)} files from source {source_id}, linking them to dataset {dataset_id} (sync)"
                        )
                        # Link files to dataset
                        for file in source_files:
                            link = DatasetFileLink(
                                dataset_id=dataset_id, file_id=file.id
                            )
                            db_session.add(link)
                        db_session.commit()
                        return source_files
                    else:
                        logger.warning(f"No files found for source {source_id} (sync)")
                except Exception as e:
                    logger.error(
                        f"Error getting files from source {source_id}: {str(e)} (sync)"
                    )

            # If we get here, we couldn't find or recover any files
            raise HTTPException(
                status_code=404, detail=f"No files found for dataset {dataset_id}."
            )

        # Get file objects for all file IDs
        files = []
        for file_id in file_ids:
            file_statement = select(File).where(File.id == file_id)
            file_result = db_session.execute(file_statement)
            file_record = file_result.scalar_one_or_none()

            if file_record:
                files.append(file_record)
            else:
                logger.warning(
                    f"File {file_id} from dataset {dataset_id} not found (sync)"
                )

        if not files:
            logger.error(
                f"None of the file IDs linked to dataset {dataset_id} exist in the files table (sync)"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Files linked to dataset {dataset_id} not found in database.",
            )

        logger.info(f"Found {len(files)} files for dataset {dataset_id} (sync)")
        return files

    async def get_successful_files(
        self,
        *,
        file_ids: List[UUID],
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> List[UUID]:
        db_session = db_session or super().get_db().session
        successful_file_ids = []
        statement = select(Document.file_id).where(
            Document.file_id.in_(file_ids),
            Document.processing_status == DocumentProcessingStatusEnum.Success,
            Document.dataset_id == dataset_id,
            Document.deleted_at.is_(None),
        )
        result = await db_session.execute(statement)
        if result:
            successful_file_ids = [row.file_id for row in result]
        return successful_file_ids

    async def get_exception_files(
        self,
        *,
        file_ids: List[UUID],
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> List[UUID]:
        db_session = db_session or super().get_db().session
        exception_file_ids = []
        statement = select(Document.file_id).where(
            Document.file_id.in_(file_ids),
            Document.processing_status == DocumentProcessingStatusEnum.Exception,
            Document.dataset_id == dataset_id,
            Document.deleted_at.is_(None),
        )
        result = await db_session.execute(statement)
        if result:
            exception_file_ids = [row.file_id for row in result]
        return exception_file_ids

    async def create_ingestion_initiation_response(
        self,
        files: List[File],
        ingestion_id: str,
        chunking_config_id: UUID,
        created_time: datetime,
    ) -> List[IIngestFilesOperationRead]:
        """Create API response for successful ingestion initiation."""
        from app.schemas.response_schema import IIngestFilesOperationRead

        response = []
        for file in files:
            response_item = IIngestFilesOperationRead(
                file_id=str(file.id),
                filename=file.filename,
                ingestion_id=ingestion_id,
                status="processing",
                chunking_config_id=(
                    str(chunking_config_id) if chunking_config_id else None
                ),
                created_at=created_time.isoformat(),
            )
            response.append(response_item)

        return response

    # def _determine_document_type(self, file: File) -> Optional[DocumentTypeEnum]:
    #     """Determine document type based on file mimetype."""
    #     if not file or not file.mimetype:
    #         return None

    #     mimetype = file.mimetype.lower()

    #     if mimetype.startswith("image/"):
    #         return DocumentTypeEnum.Image
    #     elif mimetype.startswith("audio/"):
    #         return DocumentTypeEnum.Audio
    #     elif mimetype.startswith("video/"):
    #         return DocumentTypeEnum.Video
    #     elif mimetype.startswith("application/pdf"):
    #         return DocumentTypeEnum.PDF
    #     else:
    #         return None

    def _create_file_task_mapping(
        self, file_ids: List[UUID], task_ids: List[UUID]
    ) -> Dict[UUID, UUID]:
        """Create a mapping from file_id to task_id for efficient lookup."""
        if len(file_ids) != len(task_ids):
            raise ValueError("file_ids and task_ids must have the same length")
        return dict(zip(file_ids, task_ids))

    def _update_existing_document(
        self,
        record: Document,
        ingestion_id: str,
        task_id: UUID,
        status: DocumentProcessingStatusEnum,
        created_time: datetime,
        file: File,
        file_path: str,
        document_type: Optional[DocumentTypeEnum],
        mime_type: str,
    ) -> None:
        """Update an existing document record with new values."""
        record.ingestion_id = ingestion_id
        record.task_id = task_id
        record.processing_status = status
        record.name = file.filename
        record.updated_at = created_time
        record.processed_at = None
        record.mime_type = mime_type
        record.file_path = file_path
        record.document_type = document_type

    def _create_new_document(
        self,
        file_id: UUID,
        dataset_id: UUID,
        ingestion_id: str,
        task_id: UUID,
        status: DocumentProcessingStatusEnum,
        created_time: datetime,
        file: File,
        file_path: str,
        document_type: Optional[DocumentTypeEnum],
        mime_type: str,
    ) -> Document:
        """Create a new document record."""
        return Document(
            created_at=created_time,
            ingestion_id=ingestion_id,
            task_id=task_id,
            file_id=file_id,
            dataset_id=dataset_id,
            processing_status=status,
            name=file.filename,
            file_path=file_path,
            mime_type=mime_type,
            document_type=document_type,
        )

    @retry_on_deadlock(max_retries=3, delay=0.1)
    async def create_or_update_document_records(
        self,
        *,
        ingestion_id: str,
        dataset_id: UUID,
        created_time: datetime,
        status: DocumentProcessingStatusEnum,
        file_ids: List[UUID],
        task_ids: List[UUID],
        db_session: AsyncSession | None = None,
    ):
        """Create or update document records with batch processing for better performance."""
        db_session = db_session or super().get_db().session

        created_time = ensure_naive_datetime(created_time)

        # Create a mapping of file_id to task_id for efficient lookup
        file_task_mapping = self._create_file_task_mapping(file_ids, task_ids)

        # Batch size for processing to prevent deadlocks and memory issues
        batch_size = min(DOCUMENT_BATCH_SIZE, DOCUMENT_MAX_BATCH_SIZE)

        # Remove duplicates and convert to list
        unique_file_ids = list(set(file_ids))

        logger.info(
            f"Processing {len(unique_file_ids)} document records in batches of {batch_size}"
        )

        # Process files in batches to prevent deadlocks and improve performance
        for i in range(0, len(unique_file_ids), batch_size):
            batch_file_ids = unique_file_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(unique_file_ids) + batch_size - 1) // batch_size

            retry_count = 0
            max_retries = 3

            while retry_count <= max_retries:
                try:
                    # Process this batch
                    await self._process_document_batch(
                        batch_file_ids=batch_file_ids,
                        ingestion_id=ingestion_id,
                        dataset_id=dataset_id,
                        created_time=created_time,
                        status=status,
                        file_task_mapping=file_task_mapping,
                        db_session=db_session,
                    )
                    logger.info(
                        f"Successfully processed document batch {batch_num}/{total_batches} ({len(batch_file_ids)} files)"
                    )
                    break  # Success, move to next batch

                except Exception as e:
                    error_msg = str(e).lower()
                    if "deadlock" in error_msg and retry_count < max_retries:
                        retry_count += 1
                        wait_time = 0.1 * (2**retry_count)  # Exponential backoff
                        logger.warning(
                            f"Deadlock in document batch {batch_num}/{total_batches}, retry {retry_count}/{max_retries} after {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"Error processing document batch {batch_num}/{total_batches}: {str(e)}"
                        )
                        raise

    @retry_on_deadlock(max_retries=3, delay=0.1)
    async def _process_document_batch(
        self,
        batch_file_ids: List[UUID],
        ingestion_id: str,
        dataset_id: UUID,
        created_time: datetime,
        status: DocumentProcessingStatusEnum,
        file_task_mapping: Dict[UUID, UUID],
        db_session: AsyncSession,
    ) -> None:
        """Process a batch of document records with optimized queries."""

        if not batch_file_ids:
            return

        # Batch query for existing documents (single query instead of N queries)
        existing_docs_query = (
            select(Document)
            .where(
                Document.file_id.in_(batch_file_ids),
                Document.dataset_id == dataset_id,
                Document.deleted_at.is_(None),
            )
            .order_by(Document.file_id)
        )  # Consistent ordering for lock prevention

        existing_docs_result = await db_session.execute(existing_docs_query)
        existing_docs = existing_docs_result.scalars().all()
        existing_docs_map = {doc.file_id: doc for doc in existing_docs}

        # Batch query for file information (single query instead of N queries)
        files_query = (
            select(File).where(File.id.in_(batch_file_ids)).order_by(File.id)
        )  # Consistent ordering for lock prevention

        files_result = await db_session.execute(files_query)
        files = files_result.scalars().all()
        files_map = {file.id: file for file in files}

        # Prepare documents to update and create
        documents_to_update = []
        documents_to_create = []

        for file_id in batch_file_ids:
            file = files_map.get(file_id)
            if not file:
                logger.warning(f"File {file_id} not found, skipping document creation")
                continue

            file_path = file.file_path
            document_type = _get_document_type_from_file_path(file_path)
            mime_type = _get_mime_type_from_extension(file_path)
            task_id = file_task_mapping.get(file_id)

            existing_doc = existing_docs_map.get(file_id)

            if existing_doc:
                # Update existing document
                self._update_existing_document(
                    existing_doc,
                    ingestion_id=ingestion_id,
                    task_id=task_id,
                    status=status,
                    created_time=created_time,
                    file=file,
                    file_path=file_path,
                    document_type=document_type,
                    mime_type=mime_type,
                )
                documents_to_update.append(existing_doc)
            else:
                # Create new document
                new_document = self._create_new_document(
                    file_id=file_id,
                    dataset_id=dataset_id,
                    ingestion_id=ingestion_id,
                    task_id=task_id,
                    status=status,
                    created_time=created_time,
                    file=file,
                    file_path=file_path,
                    document_type=document_type,
                    mime_type=mime_type,
                )
                documents_to_create.append(new_document)

        # Bulk operations
        if documents_to_create:
            db_session.add_all(documents_to_create)
            logger.debug(f"Added {len(documents_to_create)} new documents to session")

        if documents_to_update:
            logger.debug(f"Updated {len(documents_to_update)} existing documents")

        # Commit the batch
        await db_session.commit()
        logger.debug(
            f"Successfully committed batch with {len(documents_to_create)} new and {len(documents_to_update)} updated documents"
        )

    def create_or_update_document_records_sync(
        self,
        *,
        ingestion_id: str,
        dataset_id: UUID,
        created_time: datetime,
        status: DocumentProcessingStatusEnum,
        file_ids: List[UUID],
        task_ids: List[UUID],
        db_session,
    ):
        """
        Synchronous version of create_or_update_document_records.
        Creates or updates document records for files in a dataset.
        """
        created_time = ensure_naive_datetime(created_time)

        # Create a mapping of file_id to task_id for efficient lookup
        file_task_mapping = self._create_file_task_mapping(file_ids, task_ids)

        # Remove duplicates and convert to list
        unique_file_ids = list(set(file_ids))

        logger.info(f"Processing {len(unique_file_ids)} document records synchronously")

        # Process all files in a single batch for sync version
        self._process_document_batch_sync(
            batch_file_ids=unique_file_ids,
            ingestion_id=ingestion_id,
            dataset_id=dataset_id,
            created_time=created_time,
            status=status,
            file_task_mapping=file_task_mapping,
            db_session=db_session,
        )

    def _process_document_batch_sync(
        self,
        batch_file_ids: List[UUID],
        ingestion_id: str,
        dataset_id: UUID,
        created_time: datetime,
        status: DocumentProcessingStatusEnum,
        file_task_mapping: Dict[UUID, UUID],
        db_session,
    ) -> None:
        """Synchronous version of _process_document_batch."""

        if not batch_file_ids:
            return

        # Batch query for existing documents
        existing_docs = (
            db_session.query(Document)
            .filter(
                Document.file_id.in_(batch_file_ids),
                Document.dataset_id == dataset_id,
                Document.deleted_at.is_(None),
            )
            .order_by(Document.file_id)
            .all()
        )
        existing_docs_map = {doc.file_id: doc for doc in existing_docs}

        # Batch query for file information
        files = (
            db_session.query(File)
            .filter(File.id.in_(batch_file_ids))
            .order_by(File.id)
            .all()
        )
        files_map = {file.id: file for file in files}

        # Prepare documents to update and create
        documents_to_update = []
        documents_to_create = []

        for file_id in batch_file_ids:
            file = files_map.get(file_id)
            if not file:
                logger.warning(f"File {file_id} not found, skipping document creation")
                continue

            file_path = file.file_path
            document_type = _get_document_type_from_file_path(file_path)
            mime_type = _get_mime_type_from_extension(file_path)
            task_id = file_task_mapping.get(file_id)

            existing_doc = existing_docs_map.get(file_id)

            if existing_doc:
                # Update existing document
                self._update_existing_document(
                    existing_doc,
                    ingestion_id=ingestion_id,
                    task_id=task_id,
                    status=status,
                    created_time=created_time,
                    file=file,
                    file_path=file_path,
                    document_type=document_type,
                    mime_type=mime_type,
                )
                documents_to_update.append(existing_doc)
            else:
                # Create new document
                new_document = self._create_new_document(
                    file_id=file_id,
                    dataset_id=dataset_id,
                    ingestion_id=ingestion_id,
                    task_id=task_id,
                    status=status,
                    created_time=created_time,
                    file=file,
                    file_path=file_path,
                    document_type=document_type,
                    mime_type=mime_type,
                )
                documents_to_create.append(new_document)

        # Bulk operations
        if documents_to_create:
            db_session.add_all(documents_to_create)
            logger.debug(f"Added {len(documents_to_create)} new documents to session")

        if documents_to_update:
            logger.debug(f"Updated {len(documents_to_update)} existing documents")

        # Commit the batch
        db_session.commit()
        logger.debug(
            f"Successfully committed batch with {len(documents_to_create)} new and {len(documents_to_update)} updated documents"
        )

    async def get_filename(
        self, file_id: UUID, db_session: AsyncSession | None = None
    ) -> str:
        db_session = db_session or super().get_db().session

        statement = select(File).where(File.id == file_id)
        result = await db_session.execute(statement)
        file = result.scalar_one_or_none()
        return file.filename if file else "Unknown Filename"

    def get_filename_sync(
        self, file_id: UUID, db_session: Session | None = None
    ) -> str:
        db_session = db_session or super().get_db().session

        statement = select(File).where(File.id == file_id)
        result = db_session.execute(statement)
        file = result.scalar_one_or_none()
        return file.filename if file else "Unknown Filename"

    async def fetch_incomplete_records(
        self,
        *,
        dataset_id: UUID,
        ingestion_id: Optional[UUID],
        db_session: AsyncSession | None = None,
    ) -> Sequence[Document]:
        db_session = db_session or super().get_db().session
        if ingestion_id:
            statement = select(Document).where(
                Document.ingestion_id == ingestion_id,
                Document.processed_at.is_(None),
            )
        else:
            statement = select(Document).where(
                Document.dataset_id == dataset_id,
                Document.processed_at.is_(None),
                Document.deleted_at.is_(None),
            )

        result = await db_session.execute(statement)
        return_result = result.scalars().all()
        return return_result

    async def update_records(
        self,
        *,
        records: Sequence[Document],
        db_session: AsyncSession | None = None,
    ):
        db_session = db_session or super().get_db().session

        for record in records:
            db_session.add(record)
        await db_session.commit()

    async def get_ingestion_status(
        self,
        dataset_id: UUID,
        ingestion_id: Optional[UUID] = None,
        source_type: Optional[str] = None,
        db_session: AsyncSession | None = None,
    ) -> IGetResponseBase[List[IIngestFilesOperationRead]]:
        db_session = db_session or super().get_db().session

        if source_type == "pg_db" or source_type == "mysql_db":
            # When source_type is not None, fetch ingestion_status from Dataset table
            dataset_statement = select(Dataset).where(Dataset.id == dataset_id)
            dataset_result = await db_session.execute(dataset_statement)
            dataset_record = dataset_result.scalars().first()

            if not dataset_record:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset with ID {dataset_id} not found.",
                )

            ingestion_status = dataset_record.ingestion_status
            dataset_created_at = dataset_record.created_at

            result_records = [
                IIngestFilesOperationRead(
                    file_id=str(
                        dataset_id
                    ),  # Using dataset_id as file_id for source-based datasets
                    filename=f"Dataset_{dataset_id}",  # Generic filename for dataset
                    ingestion_id=None,  # Required field, using dummy UUID for dataset-level status
                    chunking_config_id=None,  # No chunking config for dataset-level status
                    status=ingestion_status
                    or "unknown",  # Use dataset ingestion_status
                    created_at=(
                        dataset_created_at.isoformat() if dataset_created_at else None
                    ),
                    finished_at=None,  # Dataset-level doesn't have finished_at
                )
            ]
        else:
            if ingestion_id:
                statement = select(Document).where(
                    Document.ingestion_id == ingestion_id, Document.deleted_at.is_(None)
                )
            else:
                statement = (
                    select(Document)
                    .where(
                        Document.dataset_id == dataset_id, Document.deleted_at.is_(None)
                    )
                    .order_by(Document.created_at.desc())
                )

            result = await db_session.execute(statement)
            records = result.scalars().all()

            if not records:
                raise HTTPException(
                    status_code=404,
                    detail="No records found for the given dataset_id or ingestion_id.",
                )

            _, chunking_config_id = await self.get_chunking_config_for_dataset(
                dataset_id=dataset_id
            )
            result_records = []
            for document in records:
                try:
                    filename = await self.get_filename(document.file_id)
                    ingestion_record = IIngestFilesOperationRead(
                        file_id=document.file_id,
                        filename=filename,
                        ingestion_id=document.ingestion_id,
                        chunking_config_id=str(chunking_config_id),
                        status=document.processing_status.value,
                        created_at=document.created_at.isoformat(),
                        finished_at=(
                            document.processed_at.isoformat()
                            if document.processed_at
                            else None
                        ),
                    )
                    result_records.append(ingestion_record)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing document with file ID {document.file_id}: {str(e)}",
                    )

        return create_response(
            data=result_records, message="Ingestion status retrieved successfully"
        )

    async def are_all_files_ingested(
        self,
        *,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> bool:
        db_session = db_session or super().get_db().session

        await self.get_ingestion_status(dataset_id=dataset_id)

        # Retrieve the files associated with the dataset_id
        files = await self.get_files_for_dataset(
            dataset_id=dataset_id, db_session=db_session
        )

        # Check the latest ingestion status for each file
        for file in files:
            document_record = await db_session.execute(
                select(Document)
                .where(Document.file_id == file.id)
                .order_by(Document.created_at.desc())
                .limit(1)
            )
            document_record = document_record.scalar_one_or_none()
            if (
                not document_record
                or document_record.processing_status
                != DocumentProcessingStatusEnum.Success
            ):
                return False

        return True

    async def dataset_ingestion_complete_check(
        self,
        *,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> bool:
        db_session = db_session or super().get_db().session
        statement = await db_session.execute(
            select(Document).where(
                Document.dataset_id == dataset_id,
                Document.processing_status == DocumentProcessingStatusEnum.Processing,
                Document.deleted_at.is_(None),
            )
        )
        records = statement.scalars().all()
        if records:
            return False
        return True

    async def dataset_ingestion_started_check(
        self, *, dataset_id: UUID, db_session: AsyncSession = None
    ) -> bool:
        """
        Check if ingestion has been started for a dataset

        Returns:
            True if at least one file has a document record
            False if no document records found
        """
        db_session = db_session or super().get_db().session

        ingestion_check = await db_session.execute(
            select(func.count(Document.id)).where(
                and_(
                    Document.dataset_id == dataset_id,
                    Document.deleted_at.is_(None),
                )
            )
        )

        ingestion_count = ingestion_check.scalar_one()
        return ingestion_count > 0

    def update_document_task_id_sync(
        self,
        *,
        ingestion_id: str,
        file_id: UUID,
        task_id: str,
        dataset_id: UUID,
    ) -> None:
        with SyncSessionLocal() as db:
            stmt = (
                update(Document)
                .where(
                    and_(
                        Document.ingestion_id == ingestion_id,
                        Document.file_id == file_id,
                        Document.dataset_id == dataset_id,
                        Document.deleted_at.is_(None),
                    )
                )
                .values(task_id=task_id)
            )
            db.execute(stmt)
            db.commit()

            logger.info(
                f"Updated task_id {task_id} for document with file {file_id} in ingestion "
                f"{ingestion_id} for dataset {dataset_id}"
            )

    def get_document_task_id_sync(
        self,
        *,
        ingestion_id: str,
        file_id: UUID,
        dataset_id: UUID,
    ) -> Document:
        with SyncSessionLocal() as db:
            stmt = select(Document).where(
                and_(
                    Document.ingestion_id == ingestion_id,
                    Document.file_id == file_id,
                    Document.dataset_id == dataset_id,
                    Document.deleted_at.is_(None),
                )
            )
            result = db.execute(stmt)
            return result.scalar_one_or_none()


ingestion_crud = CRUDIngestionV2(Document)
