import asyncio
from datetime import datetime
from uuid import UUID

from azure.storage.blob import BlobServiceClient
from sqlalchemy.orm import Session

from app.be_core.celery import celery
from app.be_core.logger import logger
from app.crud.source_connector_crud import CRUDSource
from app.db.session import SyncSessionLocal
from app.models import Source
from app.models.azure_storage_model import AzureStorage
from app.models.chunking_config_model import ChunkingConfig
from app.models.dataset_model import Dataset
from app.models.file_model import File
from app.utils.datetime_utils import ensure_naive_datetime

crud = CRUDSource(Source)


@celery.task(name="tasks.monitor_azure_sources_task")
def monitor_azure_sources_task():
    """
    Enhanced periodic task to monitor Azure Storage sources for new files.
    This is the primary mechanism for auto-detection in multi-tenant scenarios.
    """
    logger.info("Starting enhanced Azure Storage monitoring task")
    db_session: Session = SyncSessionLocal()

    def monitor_all():
        try:
            # Get all Azure Storage sources with auto-detection enabled
            azure_sources = (
                db_session.query(AzureStorage)
                .filter(AzureStorage.auto_detection_enabled)
                .all()
            )
            logger.info(
                f"Found {len(azure_sources)} Azure Storage sources with auto-detection enabled"
            )
            # Only monitor sources that are due for monitoring
            sources_to_monitor = [
                src for src in azure_sources if _should_monitor_source(src)
            ]
            logger.info(
                f"Monitoring {len(sources_to_monitor)} sources that are due for monitoring."
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tasks = [
                loop.run_in_executor(
                    None, _monitor_single_source_enhanced_sync, azure_source, db_session
                )
                for azure_source in sources_to_monitor
            ]
            loop.run_until_complete(asyncio.gather(*tasks))
        except Exception as e:
            logger.error(f"Error in Azure monitoring task: {str(e)}")
        finally:
            db_session.close()

    return monitor_all()


def _should_monitor_source(azure_source: AzureStorage) -> bool:
    """Check if source should be monitored based on frequency."""
    last_monitored = ensure_naive_datetime(azure_source.last_monitored)
    if last_monitored is None:
        return True

    time_since_last = ensure_naive_datetime(datetime.utcnow()) - last_monitored
    frequency_minutes = azure_source.monitoring_frequency_minutes or 30
    return time_since_last.total_seconds() > frequency_minutes * 60


def _monitor_single_source_enhanced_sync(
    azure_source: AzureStorage, db_session: Session
):
    """Enhanced monitoring of a single Azure Storage source for new files, using vault secret retrieval (sync version)."""
    try:
        logger.info(f"Monitoring Azure Storage source {azure_source.source_id}")

        # Fetch the actual SAS URL from the vault (sync)
        details = crud.get_azure_storage_details_sync(
            UUID(str(azure_source.source_id)), db_session
        )
        sas_url = details["sas_url"]
        container_name = details["container_name"]

        # Find the dataset for this source
        dataset = (
            db_session.query(Dataset)
            .filter(
                Dataset.source_id == azure_source.source_id,
                Dataset.deleted_at.is_(None),
            )
            .first()
        )
        if not dataset:
            raise ValueError(f"No dataset found for source {azure_source.source_id}")

        # Find the chunking config for this dataset
        chunking_config = (
            db_session.query(ChunkingConfig)
            .filter(
                ChunkingConfig.dataset_id == dataset.id,
                ChunkingConfig.deleted_at.is_(None),
            )
            .first()
        )
        if not chunking_config:
            raise ValueError(f"No chunking config found for dataset {dataset.id}")

        # Get list of existing files in database with their last modified times
        existing_files = (
            db_session.query(File)
            .filter(File.source_id == azure_source.source_id)
            .all()
        )
        existing_file_info = {f.filename: f.updated_at for f in existing_files}

        logger.debug(
            f"Found {len(existing_file_info)} existing files in database for source {azure_source.source_id}"
        )

        # Get list of blobs in Azure container with their properties
        blob_service_client = BlobServiceClient(account_url=sas_url)
        container_client = blob_service_client.get_container_client(container_name)

        new_blobs = []
        updated_blobs = []
        blob_count = 0

        for blob in container_client.list_blobs():
            blob_count += 1
            blob_name = blob.name

            # Check if blob is new or updated
            if blob_name not in existing_file_info:
                new_blobs.append(
                    {
                        "name": blob_name,
                        "size": blob.size,
                        "last_modified": blob.last_modified,
                        "content_type": blob.content_settings.content_type,
                    }
                )
            else:
                # Check if blob has been updated since last check
                db_last_modified = existing_file_info[blob_name]
                blob_last_modified = ensure_naive_datetime(blob.last_modified)
                db_last_modified_naive = ensure_naive_datetime(db_last_modified)
                if db_last_modified and blob_last_modified > db_last_modified_naive:
                    updated_blobs.append(
                        {
                            "name": blob_name,
                            "size": blob.size,
                            "last_modified": blob_last_modified,
                            "content_type": blob.content_settings.content_type,
                        }
                    )

        logger.info(
            f"Found {blob_count} total blobs, {len(new_blobs)} new blobs, {len(updated_blobs)} updated blobs in source {azure_source.source_id}"
        )

        # Process new blobs with deduplication
        for blob_info in new_blobs:
            blob_name = blob_info["name"]
            logger.info(f"Processing new blob: {blob_name}")

            # Check if this blob is already being processed
            try:
                from app.api.deps import get_redis_client_sync

                redis_client = get_redis_client_sync()
                dedup_key = (
                    f"azure_blob_processing:{azure_source.source_id}:{blob_name}"
                )

                # Check if already being processed
                if redis_client.exists(dedup_key):
                    logger.info(f"Blob {blob_name} already being processed, skipping")
                    continue

            except Exception as e:
                logger.warning(f"Failed to check deduplication: {str(e)}")

            celery.send_task(
                "tasks.process_new_azure_blob_task",
                kwargs={
                    "source_id": str(azure_source.source_id),
                    "blob_name": blob_info["name"],
                    "blob_url": f"{container_client.url}/{blob_info['name']}",
                    "event_time": datetime.utcnow().isoformat(),
                    "blob_metadata": {
                        "size": blob_info["size"],
                        "last_modified": (
                            blob_info["last_modified"].isoformat()
                            if blob_info["last_modified"]
                            else None
                        ),
                        "content_type": blob_info["content_type"],
                    },
                    "dataset_id": str(dataset.id),
                    "chunking_config": (
                        chunking_config.to_dict()
                        if hasattr(chunking_config, "to_dict")
                        else None
                    ),
                    "organization_id": str(dataset.workspace_id),
                },
                queue="file_pull_queue",
            )

        # Process updated blobs based on source-level setting
        if updated_blobs:
            if azure_source.re_ingest_updated_blobs:
                logger.info(
                    f"Re-ingestion enabled for source {azure_source.source_id}: Processing {len(updated_blobs)} updated blobs"
                )
                for blob_info in updated_blobs:
                    blob_name = blob_info["name"]
                    logger.info(f"Re-ingesting updated blob: {blob_name}")

                    # Check if this blob is already being processed
                    try:
                        from app.api.deps import get_redis_client_sync

                        redis_client = get_redis_client_sync()
                        dedup_key = f"azure_blob_update_processing:{azure_source.source_id}:{blob_name}"

                        # Check if already being processed
                        if redis_client.exists(dedup_key):
                            logger.info(
                                f"Updated blob {blob_name} already being processed, skipping"
                            )
                            continue

                    except Exception as e:
                        logger.warning(f"Failed to check deduplication: {str(e)}")

                    celery.send_task(
                        "tasks.process_updated_azure_blob_task",
                        kwargs={
                            "source_id": str(azure_source.source_id),
                            "blob_name": blob_info["name"],
                            "blob_url": f"{container_client.url}/{blob_info['name']}",
                            "event_time": datetime.utcnow().isoformat(),
                            "blob_metadata": {
                                "size": blob_info["size"],
                                "last_modified": (
                                    blob_info["last_modified"].isoformat()
                                    if blob_info["last_modified"]
                                    else None
                                ),
                                "content_type": blob_info["content_type"],
                            },
                            "dataset_id": str(dataset.id),
                            "chunking_config": (
                                chunking_config.to_dict()
                                if hasattr(chunking_config, "to_dict")
                                else None
                            ),
                            "organization_id": str(dataset.workspace_id),
                        },
                        queue="file_pull_queue",
                    )
            else:
                logger.info(
                    f"Re-ingestion disabled for source {azure_source.source_id}: Logging {len(updated_blobs)} updated blobs without re-ingestion"
                )
                for blob_info in updated_blobs:
                    logger.info(
                        f"Updated blob detected (re-ingestion disabled for source {azure_source.source_id}): {blob_info['name']} "
                        f"(size: {blob_info['size']}, last_modified: {blob_info['last_modified']}, "
                        f"content_type: {blob_info['content_type']})"
                    )

        # Update last monitored time
        azure_source.last_monitored = datetime.utcnow()
        db_session.commit()

        if new_blobs:
            logger.info(
                f"Successfully queued {len(new_blobs)} new blobs for processing in source {azure_source.source_id}"
            )

        if updated_blobs:
            if azure_source.re_ingest_updated_blobs:
                logger.info(
                    f"Successfully queued {len(updated_blobs)} updated blobs for re-ingestion in source {azure_source.source_id}"
                )
            else:
                logger.info(
                    f"Logged {len(updated_blobs)} updated blobs without re-ingestion in source {azure_source.source_id}"
                )

    except Exception as e:
        logger.error(f"Error monitoring source {azure_source.source_id}: {str(e)}")
        # Don't update last_monitored on error to retry sooner


# Note: For AWS S3, use get_s3_storage_details in the future for secret retrieval.
