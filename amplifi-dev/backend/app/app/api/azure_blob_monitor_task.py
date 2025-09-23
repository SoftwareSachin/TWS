from datetime import UTC, datetime
from typing import Any, Dict, Optional

from azure.storage.blob.aio import BlobServiceClient
from sqlalchemy.orm import Session

from app.api.v2.endpoints.ingest_file import process_files_background
from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.dataset_crud_v2 import CRUDDatasetV2
from app.crud.ingest_crud_v2 import ingestion_crud
from app.crud.source_connector_crud import CRUDSource
from app.db.session import SyncSessionLocal
from app.models import Source
from app.models.dataset_model import Dataset
from app.models.document_model import DocumentProcessingStatusEnum
from app.models.file_model import File
from app.schemas.file_schema import FileStatusEnum
from app.utils.uuid6 import uuid7

crud = CRUDSource(Source)
ds_crud = CRUDDatasetV2(Dataset)


def _get_dataset_and_chunking_config(db_session, source_id):
    dataset = (
        db_session.query(Dataset)
        .filter(Dataset.source_id == source_id, Dataset.deleted_at.is_(None))
        .first()
    )
    if not dataset:
        raise ValueError(f"No dataset found for source {source_id}")

    # Use the synchronous CRUD function to get a clean dictionary
    from app.crud.ingest_crud_v2 import ingestion_crud

    chunking_config_dict, chunking_config_id = (
        ingestion_crud.get_chunking_config_for_dataset_sync(
            dataset_id=dataset.id, db_session=db_session
        )
    )

    if not chunking_config_dict:
        raise ValueError(f"No chunking config found for dataset {dataset.id}")

    return dataset, chunking_config_dict


def _download_azure_blob_to_local(
    file_path: str, metadata: Optional[dict] = None
) -> str:
    """
    Download Azure blob to local storage for processing.
    Returns the local file path for processing.
    """
    import asyncio
    import os

    from azure.storage.blob.aio import BlobServiceClient

    # Extract Azure Storage details from file_path or metadata
    if metadata and "azure_storage" in metadata:
        azure_info = metadata["azure_storage"]
        sas_url = azure_info["sas_url"]
        container_name = azure_info["container_name"]
        blob_name = azure_info["blob_name"]
    else:
        # Parse from file_path format: azure://container/blob_name
        parts = file_path.replace("azure://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid Azure Storage URL format: {file_path}")
        container_name, blob_name = parts
        # You'd need to get SAS URL from the source record
        raise ValueError("SAS URL not provided in metadata")

    # Use DEFAULT_UPLOAD_FOLDER like azure_source_file_pull_task.py
    from app.be_core.config import settings

    DEFAULT_UPLOAD_FOLDER = settings.DEFAULT_UPLOAD_FOLDER

    # Create unique filename with source_id prefix if available
    # if metadata and "source_id" in metadata:
    #     unique_filename = f"{metadata['source_id']}_{blob_name}"
    # else:
    #     unique_filename = blob_name

    unique_filename = blob_name

    file_path = os.path.join(DEFAULT_UPLOAD_FOLDER, unique_filename)

    # Ensure the folder path exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Remove existing file if it exists to ensure clean overwrite
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Removed existing file before overwriting: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove existing file {file_path}: {str(e)}")

    async def _download():
        async with BlobServiceClient(account_url=sas_url) as blob_service_client:
            container_client = blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)

            with open(file_path, "wb") as f:
                stream = await blob_client.download_blob()
                data = await stream.readall()
                f.write(data)

    try:
        asyncio.run(_download())
        logger.info(f"Downloaded Azure blob {blob_name} to {file_path}")
        return file_path
    except Exception as e:
        # Clean up file on error
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except OSError:  # Fixed E722: specify exception type
            pass
        raise e


@celery.task(name="tasks.process_new_azure_blob_task", bind=True, max_retries=3)
def process_new_azure_blob_task(
    self,
    source_id: str,
    blob_name: str,
    blob_url: str,
    event_time: str,
    blob_metadata: Optional[Dict[str, Any]] = None,
    dataset_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a newly created Azure blob and trigger ingestion.
    Uses polling-based detection mechanism with deduplication.
    """
    # Create a unique key for this blob processing task
    # Use source_id + blob_name to prevent the same blob from being processed multiple times
    # Add a short window (30 seconds) to allow for retries but prevent duplicates
    dedup_key = f"azure_blob_processing:{source_id}:{blob_name}"

    # Try to acquire a lock to prevent duplicate processing
    try:
        from app.api.deps import get_redis_client_sync

        redis_client = get_redis_client_sync()

        # Try to set the key with expiration (30 seconds) - short enough for retries, long enough to prevent duplicates
        lock_acquired = redis_client.set(dedup_key, "processing", ex=30, nx=True)

        if not lock_acquired:
            logger.info(
                f"Task already being processed for blob {blob_name}, skipping duplicate"
            )
            return {
                "status": "skipped",
                "reason": "duplicate_task",
                "blob_name": blob_name,
            }

    except Exception as e:
        logger.warning(
            f"Failed to acquire deduplication lock: {str(e)}, proceeding anyway"
        )
        redis_client = None

    logger.info(f"Processing new blob: {blob_name} from source {source_id}")

    db_session: Session = SyncSessionLocal()

    try:
        # Get source details
        source = db_session.query(Source).filter(Source.id == source_id).first()
        if not source or source.source_type != "azure_storage":
            raise ValueError(f"Invalid source {source_id} or not Azure Storage")

        # Get Azure Storage credentials from vault or model
        azure_details = crud.get_azure_storage_details_sync(source_id, db_session)
        if not azure_details:
            raise ValueError(f"Azure Storage details not found for source {source_id}")
        sas_url = azure_details["sas_url"]
        container_name = azure_details["container_name"]

        # Validate file type using config settings
        file_extension = blob_name.split(".")[-1].lower()
        valid_blob_type = crud.sync_check_valid_file_type(
            file_extension=file_extension,
        )
        if not valid_blob_type:
            logger.info(
                f"Skipping blob {blob_name}: invalid file type (extension: {file_extension})"
            )
            return {
                "status": "skipped",
                "reason": "invalid_file_type",
                "extension": file_extension,
            }

        # Check if file already exists
        existing_file = (
            db_session.query(File)
            .filter(File.filename == blob_name, File.source_id == source_id)
            .first()
        )

        if existing_file:
            logger.info(f"File {blob_name} already exists, skipping")
            return {"status": "skipped", "reason": "already_exists"}

        # Get blob metadata from Azure
        blob_metadata = _get_blob_metadata(sas_url, container_name, blob_name)

        # Validate MIME type if available
        if blob_metadata.get("content_type"):
            if not _is_valid_mime_type(blob_metadata["content_type"]):
                logger.info(
                    f"Skipping blob {blob_name}: invalid MIME type ({blob_metadata['content_type']})"
                )
                return {
                    "status": "skipped",
                    "reason": "invalid_mime_type",
                    "mime_type": blob_metadata["content_type"],
                }

        # Create file record with Azure Storage URL as file_path
        azure_file_path = f"azure://{container_name}/{blob_name}"

        metadata = {
            "source": "azure_storage_auto_detection",
            "blob_name": blob_name,
            "event_time": event_time,
            "source_id": source_id,
            "azure_storage": {
                "sas_url": sas_url,
                "container_name": container_name,
                "blob_name": blob_name,
            },
        }

        download_file_path = _download_azure_blob_to_local(azure_file_path, metadata)

        file_record = File(
            filename=blob_name,
            mimetype=blob_metadata.get(
                "content_type", _get_mimetype_from_extension(file_extension)
            ),
            size=blob_metadata.get("size", 0),
            file_path=download_file_path,
            status=FileStatusEnum.Uploaded,
            source_id=source_id,
            workspace_id=source.workspace_id,
        )

        db_session.add(file_record)
        db_session.commit()
        db_session.refresh(file_record)

        logger.info(f"File record created for {blob_name}")

        # Find the dataset for this source
        dataset, chunking_config = _get_dataset_and_chunking_config(
            db_session, source_id
        )

        ds_crud.link_file_to_dataset_sync(dataset_id, file_record.id, db_session)

        ingestion_id = str(uuid7())

        # Create document records in database
        ingestion_crud.create_or_update_document_records_sync(
            ingestion_id=ingestion_id,
            dataset_id=dataset_id,
            created_time=datetime.now(UTC),
            status=DocumentProcessingStatusEnum.Processing,
            file_ids=[file_record.id],
            task_ids=[None],
            db_session=db_session,
        )

        # Trigger ingestion of new fiile
        process_files_background(
            files=[file_record],
            dataset_id=dataset_id,
            organization_id=source.workspace.organization_id,
            ingestion_id=ingestion_id,
            chunking_config_instance=chunking_config,
            metadata={},
            skip_successful_files=True,
            user_id=None,
        )

        return {
            "status": "success",
            "file_id": str(file_record.id),
            "blob_name": blob_name,
        }

    except Exception as e:
        logger.error(f"Error processing new blob {blob_name}: {str(e)}")
        if self.request.retries >= self.max_retries:
            logger.error(f"Max retries reached for blob {blob_name}")
        raise self.retry(exc=e, countdown=60)
    finally:
        # Clean up the deduplication lock
        if "redis_client" in locals() and redis_client:
            try:
                redis_client.delete(dedup_key)
            except Exception as e:
                logger.warning(f"Failed to clean up deduplication lock: {str(e)}")

        db_session.close()


def _is_valid_mime_type(mime_type: str) -> bool:
    """Check if MIME type is allowed based on config settings."""
    return mime_type.lower() in settings.ALLOWED_MIME_TYPES


def _get_blob_metadata(
    sas_url: str, container_name: str, blob_name: str
) -> Dict[str, Any]:
    """Get blob metadata from Azure Storage."""
    import asyncio

    async def _async_get_metadata():
        async with BlobServiceClient(account_url=sas_url) as blob_service_client:
            container_client = blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)

            properties = await blob_client.get_blob_properties()
            return {
                "size": properties.size,
                "content_type": properties.content_settings.content_type,
                "last_modified": properties.last_modified,
            }

    return asyncio.run(_async_get_metadata())


def _get_mimetype_from_extension(extension: str) -> str:
    """Get MIME type from file extension based on config settings."""
    # Exact MIME type mapping matching the config settings
    mime_type_mapping = {
        # Document formats
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        # Text formats
        "html": "text/html",
        "csv": "text/csv",
        "md": "text/markdown",
        # Image formats
        "png": "image/png",
        "jpg": "image/jpg",  # Note: config has both 'image/jpg' and 'image/jpeg'
        "jpeg": "image/jpeg",
        # Audio formats
        "wav": "audio/wav",
        "mp3": "audio/mp3",  # Config has both 'audio/mpeg' and 'audio/mp3'
        "aac": "audio/aac",
    }

    # Return the mapped MIME type or a default
    return mime_type_mapping.get(extension.lower(), "application/octet-stream")


@celery.task(name="tasks.process_updated_azure_blob_task", bind=True, max_retries=3)
def process_updated_azure_blob_task(
    self,
    source_id: str,
    blob_name: str,
    blob_url: str,
    event_time: str,
    blob_metadata: Dict[str, Any] = None,
    dataset_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an updated Azure blob.
    This can either re-ingest the file or just update metadata.
    """
    # Create a unique key for this blob processing task
    # Use source_id + blob_name to prevent the same blob from being processed multiple times
    # Add a short window (30 seconds) to allow for retries but prevent duplicates
    dedup_key = f"azure_blob_update_processing:{source_id}:{blob_name}"

    # Try to acquire a lock to prevent duplicate processing
    try:
        from app.api.deps import get_redis_client_sync

        redis_client = get_redis_client_sync()

        # Try to set the key with expiration (30 seconds) - short enough for retries, long enough to prevent duplicates
        lock_acquired = redis_client.set(dedup_key, "processing", ex=30, nx=True)

        if not lock_acquired:
            logger.info(
                f"Update task already being processed for blob {blob_name}, skipping duplicate"
            )
            return {
                "status": "skipped",
                "reason": "duplicate_update_task",
                "blob_name": blob_name,
            }

    except Exception as e:
        logger.warning(
            f"Failed to acquire deduplication lock: {str(e)}, proceeding anyway"
        )
        redis_client = None

    logger.info(f"Processing updated blob: {blob_name} from source {source_id}")
    db_session: Session = SyncSessionLocal()
    try:
        existing_file = (
            db_session.query(File)
            .filter(File.filename == blob_name, File.source_id == source_id)
            .first()
        )

        # Get Azure Storage credentials from vault or model
        azure_details = crud.get_azure_storage_details_sync(source_id, db_session)
        if not azure_details:
            raise ValueError(f"Azure Storage details not found for source {source_id}")
        container_name = azure_details["container_name"]
        sas_url = azure_details["sas_url"]

        if not existing_file:
            logger.warning(
                f"Updated blob {blob_name} not found in database, treating as new"
            )
            return {
                "status": "success",
                "file_id": str(existing_file.id),
                "blob_name": blob_name,
                "action": "re_ingestion_not_triggered",
            }

        # Download blob to local storage
        metadata = {
            "source": "azure_storage_auto_detection",
            "blob_name": blob_name,
            "event_time": event_time,
            "source_id": source_id,
            "update_type": "blob_updated",
            "azure_storage": {
                "sas_url": sas_url,
                "container_name": container_name,
                "blob_name": blob_name,
            },
        }

        azure_file_path = f"azure://{container_name}/{blob_name}"

        download_file_path = _download_azure_blob_to_local(azure_file_path, metadata)
        existing_file.file_path = download_file_path

        if blob_metadata:
            existing_file.size = blob_metadata.get("size", existing_file.size)
            existing_file.mimetype = blob_metadata.get(
                "content_type", existing_file.mimetype
            )
        existing_file.status = FileStatusEnum.Uploaded
        db_session.commit()
        dataset, chunking_config = _get_dataset_and_chunking_config(
            db_session, source_id
        )

        # Create document records in database

        ingestion_crud.create_or_update_document_records_sync(
            ingestion_id=str(uuid7()),
            dataset_id=dataset_id,
            created_time=datetime.now(UTC),
            status=DocumentProcessingStatusEnum.Processing,
            file_ids=[existing_file.id],
            task_ids=[None],
            db_session=db_session,
        )

        # Trigger re-ingestion of existing file
        process_files_background(
            files=[existing_file],
            dataset_id=dataset_id,
            organization_id=existing_file.workspace.organization_id,
            ingestion_id=str(uuid7()),
            chunking_config_instance=chunking_config,
            metadata={},
            skip_successful_files=False,
            user_id=None,
        )

        return {
            "status": "success",
            "file_id": str(existing_file.id),
            "blob_name": blob_name,
            "action": "re_ingestion_triggered",
        }
    except Exception as e:
        logger.error(f"Error processing updated blob {blob_name}: {str(e)}")
        if self.request.retries >= self.max_retries:
            logger.error(f"Max retries reached for updated blob {blob_name}")
        raise self.retry(exc=e, countdown=60)
    finally:
        # Clean up the deduplication lock
        if "redis_client" in locals() and redis_client:
            try:
                redis_client.delete(dedup_key)
            except Exception as e:
                logger.warning(f"Failed to clean up deduplication lock: {str(e)}")

        db_session.close()
