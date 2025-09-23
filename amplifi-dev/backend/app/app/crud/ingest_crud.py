from datetime import datetime
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
from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType
from app.models.file_model import File, FileStatusEnum
from app.schemas.chunking_config_schema import IChunkingMethodEnum
from app.schemas.response_schema import (
    IGetResponseBase,
    IIngestFilesOperationRead,
    IngestionStatusEnum,
    create_response,
)


class CRUDIngestion(CRUDBase[FileIngestion, None, None]):

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
        statement = select(FileIngestion.file_id).where(
            FileIngestion.file_id.in_(file_ids),
            FileIngestion.status == FileIngestionStatusType.Success,
            FileIngestion.dataset_id == dataset_id,
            FileIngestion.deleted_at.is_(None),
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
        statement = select(FileIngestion.file_id).where(
            FileIngestion.file_id.in_(file_ids),
            FileIngestion.status == FileIngestionStatusType.Exception,
            FileIngestion.dataset_id == dataset_id,
            FileIngestion.deleted_at.is_(None),
        )
        result = await db_session.execute(statement)
        if result:
            exception_file_ids = [row.file_id for row in result]
        return exception_file_ids

    async def get_r2r_ids(
        self,
        *,
        dataset_id: UUID,
        file_ids: List[UUID],
        db_session: AsyncSession | None = None,
    ) -> Sequence[UUID]:
        db_session = db_session or super().get_db().session
        statement = select(DatasetFileLink.file_id, DatasetFileLink.r2r_id).where(
            DatasetFileLink.dataset_id == dataset_id,
            DatasetFileLink.file_id.in_(file_ids),
        )
        result = await db_session.execute(statement)
        file_r2r_map = dict(result.fetchall())
        ordered_r2r_ids = [file_r2r_map.get(file_id) for file_id in file_ids]
        return ordered_r2r_ids

    def get_r2r_id_sync(
        self,
        *,
        file_id: UUID,
        dataset_id: UUID,
        db_session: Session | None = None,
    ) -> Optional[UUID]:
        db_session = db_session or super().get_db().session
        statement = select(DatasetFileLink.r2r_id).where(
            DatasetFileLink.file_id == file_id, DatasetFileLink.dataset_id == dataset_id
        )
        result = db_session.execute(statement)
        record = result.scalar_one_or_none()
        return record

    async def create_ingestion_initiation_response(
        self,
        *,
        files: Sequence[File],
        # task_ids: List[UUID],
        ingestion_id: str,
        chunking_config_id: UUID,
        created_time: datetime,
    ) -> List[IIngestFilesOperationRead]:
        ingestion_records = []
        for file in files:
            ingestion_record = IIngestFilesOperationRead(
                file_id=file.id,
                filename=file.filename,
                ingestion_id=ingestion_id,
                # task_id=task_ids[files.index(file)],
                chunking_config_id=str(chunking_config_id),
                status="Not_Started",
                created_at=created_time.isoformat(),
            )
            ingestion_records.append(ingestion_record)
        return ingestion_records

    async def create_or_update_file_ingestion_records(
        self,
        *,
        ingestion_id: str,
        dataset_id: UUID,
        created_time: datetime,
        status,
        name: str,
        file_ids: List[UUID],
        task_ids: List[UUID],
        db_session: AsyncSession | None = None,
    ):
        db_session = db_session or super().get_db().session

        for file_id in file_ids:
            statement = select(FileIngestion).where(
                FileIngestion.file_id == file_id, FileIngestion.dataset_id == dataset_id
            )
            result = await db_session.execute(statement)
            record = result.scalar_one_or_none()

            if record:
                record.ingestion_id = ingestion_id
                record.task_id = task_ids[file_ids.index(file_id)]
                record.status = status
                record.name = name
                record.updated_at = created_time
                record.finished_at = None
            else:
                ingestion_record = FileIngestion(
                    created_at=created_time,
                    ingestion_id=ingestion_id,
                    task_id=task_ids[file_ids.index(file_id)],
                    file_id=file_id,
                    dataset_id=dataset_id,
                    status=status,
                    name=name,
                )
                db_session.add(ingestion_record)
        await db_session.commit()

    async def get_file_ids_from_r2r_ids(
        self, *, r2r_ids: List[UUID], db_session: AsyncSession | None = None
    ) -> Dict[str, str]:

        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(DatasetFileLink.r2r_id, DatasetFileLink.file_id).where(
                DatasetFileLink.r2r_id.in_(r2r_ids)
            )
        )
        result_in_dict = {
            str(r2r_id): str(file_id) for r2r_id, file_id in result.fetchall()
        }

        return result_in_dict

    def get_file_ids_from_r2r_ids_sync(self, *, r2r_ids: List[UUID]) -> Dict[str, str]:

        with SyncSessionLocal() as db_session:
            db_session = db_session or super().get_db().session
            result = db_session.execute(
                select(DatasetFileLink.r2r_id, DatasetFileLink.file_id).where(
                    DatasetFileLink.r2r_id.in_(r2r_ids)
                )
            )
            result_in_dict = {
                str(r2r_id): str(file_id) for r2r_id, file_id in result.fetchall()
            }

            return result_in_dict

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
    ) -> Sequence[FileIngestion]:
        db_session = db_session or super().get_db().session
        if ingestion_id:
            statement = select(FileIngestion).where(
                FileIngestion.ingestion_id == ingestion_id,
                FileIngestion.finished_at.is_(None),
            )
        else:
            statement = select(FileIngestion).where(
                FileIngestion.dataset_id == dataset_id,
                FileIngestion.finished_at.is_(None),
                FileIngestion.deleted_at.is_(None),
            )

        result = await db_session.execute(statement)
        return_result = result.scalars().all()
        return return_result

    async def update_records(
        self,
        *,
        records: Sequence[FileIngestion],
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
        db_session: AsyncSession | None = None,
    ) -> IGetResponseBase[List[IIngestFilesOperationRead]]:
        db_session = db_session or super().get_db().session

        if ingestion_id:
            statement = select(FileIngestion).where(
                FileIngestion.ingestion_id == ingestion_id
            )
        else:
            statement = (
                select(FileIngestion)
                .where(FileIngestion.dataset_id == dataset_id)
                .order_by(FileIngestion.created_at.desc())
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
        for file in records:
            try:
                filename = await self.get_filename(file.file_id)
                ingestion_record = IIngestFilesOperationRead(
                    file_id=file.file_id,
                    filename=filename,
                    ingestion_id=file.ingestion_id,
                    task_id=file.task_id,
                    chunking_config_id=str(chunking_config_id),
                    status=file.status,
                    created_at=file.created_at.isoformat(),
                    finished_at=(
                        file.finished_at.isoformat() if file.finished_at else None
                    ),
                )
                result_records.append(ingestion_record)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file with ID {file.file_id}: {str(e)}",
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
            ingestion_record = await db_session.execute(
                select(FileIngestion)
                .where(FileIngestion.file_id == file.id)
                .order_by(FileIngestion.created_at.desc())
                .limit(1)
            )
            ingestion_record = ingestion_record.scalar_one_or_none()
            if (
                not ingestion_record
                or ingestion_record.status != FileIngestionStatusType.Success
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
            select(FileIngestion).where(
                FileIngestion.dataset_id == dataset_id,
                FileIngestion.status == IngestionStatusEnum.Processing,
                FileIngestion.deleted_at.is_(None),
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
            True if at least one file has an ingestion record
            False if no ingestion records found
        """
        db_session = db_session or super().get_db().session

        ingestion_check = await db_session.execute(
            select(func.count(FileIngestion.id)).where(
                and_(
                    FileIngestion.dataset_id == dataset_id,
                    FileIngestion.deleted_at.is_(None),
                )
            )
        )

        ingestion_count = ingestion_check.scalar_one()
        return ingestion_count > 0

    def update_file_ingestion_task_id_sync(
        self,
        *,
        ingestion_id: str,
        file_id: UUID,
        task_id: str,
        dataset_id: UUID,
    ) -> None:
        with SyncSessionLocal() as db:
            stmt = (
                update(FileIngestion)
                .where(
                    and_(
                        FileIngestion.ingestion_id == ingestion_id,
                        FileIngestion.file_id == file_id,
                        FileIngestion.dataset_id == dataset_id,
                        FileIngestion.deleted_at.is_(None),
                    )
                )
                .values(task_id=task_id)
            )
            db.execute(stmt)
            db.commit()

            logger.info(
                f"Updated task_id {task_id} for file {file_id} in ingestion "
                f"{ingestion_id} for dataset {dataset_id}"
            )

    def get_file_ingestion_task_id_sync(
        self,
        *,
        ingestion_id: str,
        file_id: UUID,
        dataset_id: UUID,
    ) -> FileIngestion:
        with SyncSessionLocal() as db:
            stmt = select(FileIngestion).where(
                and_(
                    FileIngestion.ingestion_id == ingestion_id,
                    FileIngestion.file_id == file_id,
                    FileIngestion.dataset_id == dataset_id,
                    FileIngestion.deleted_at.is_(None),
                )
            )
            result = db.execute(stmt)
            return result.scalar_one_or_none()


ingestion_crud = CRUDIngestion(FileIngestion)
