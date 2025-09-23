import os
from typing import Any, Dict, List
from uuid import UUID

from azure.storage.blob import BlobServiceClient
from sqlalchemy.orm import Session

from app.api import deps
from app.api.deps import publish_websocket_message
from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.source_connector_crud import CRUDSource
from app.db.session import SyncSessionLocal
from app.models import Source
from app.models.file_model import File
from app.models.pull_status_model import PullStatusEnum, SourcePullStatus
from app.schemas.file_schema import FileStatusEnum, IFileUploadRead
from app.schemas.long_task_schema import ITaskType

crud = CRUDSource(Source)


@celery.task(name="tasks.pull_files_from_azure_source_task", bind=True, max_retries=3)
def pull_files_from_azure_source_task(
    self,
    workspace_id: UUID,
    user_id: UUID,
    source_id: UUID,
    container_name: str,
    sas_url: str,
) -> List[Dict[str, Any]]:
    logger.info("Starting pull_files_from_azure_source_task")
    logger.info(f"Source ID: {source_id}, Container Name: {container_name}")

    db_session: Session = SyncSessionLocal()
    DEFAULT_UPLOAD_FOLDER = settings.DEFAULT_UPLOAD_FOLDER
    pulled_files: List[File] = []

    def process_blobs(container_client, prefix: str = ""):
        nonlocal pulled_files
        try:
            for blob in container_client.list_blobs(name_starts_with=prefix):
                if blob.name.endswith("/"):
                    logger.info(f"Found directory: {blob.name}, processing recursively")
                    process_blobs(container_client, blob.name)
                    continue

                original_filename = blob.name
                file_extension = original_filename.split(".")[-1].lower()
                valid_blob_type = crud.sync_check_valid_file_type(
                    file_extension=file_extension,
                )
                if not valid_blob_type:
                    logger.info(
                        f"Skipping blob: {blob.name}. Invalid type (extension: '{file_extension}')"
                    )
                    continue

                # Prefix source_id with underscore to the blob name
                unique_filename = f"{blob.name}"
                file_path = os.path.join(DEFAULT_UPLOAD_FOLDER, unique_filename)

                # Ensure the folder path exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Create DB record
                file = File(
                    filename=unique_filename,
                    mimetype=blob.content_settings.content_type,
                    size=blob.size,
                    file_path=file_path,
                    status=FileStatusEnum.Uploading,
                    source_id=source_id,
                    workspace_id=workspace_id,
                )
                db_session.add(file)
                db_session.commit()
                db_session.refresh(file)
                logger.info(f"File record created for {blob.name}")

                try:
                    # Download and save the file locally
                    with open(file_path, "wb") as f:
                        stream = container_client.download_blob(blob)
                        data = stream.readall()
                        f.write(data)
                    file.status = FileStatusEnum.Uploaded
                    pulled_files.append(
                        {
                            "filename": file.filename,
                            "size": file.size,
                            "status": file.status,
                            "id": str(file.id),
                        }
                    )
                    logger.info(f"File {blob.name} uploaded and saved successfully")
                except Exception as e:
                    file.status = FileStatusEnum.Failed
                    logger.error(f"Error uploading file {blob.name}: {e}")
                    if self.request.retries >= self.max_retries:
                        file.status = FileStatusEnum.Stopped
                        logger.error(
                            f"Max retries reached for file {blob.name}. Status set to Stopped."
                        )
                    self.retry(exc=e, countdown=0)
                finally:
                    db_session.commit()
        except Exception as e:
            logger.error(f"Error processing blobs: {e}")
            raise

    try:
        if not container_name or not sas_url:
            raise ValueError("Container name and SAS URL are required")

        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.STARTED})
        db_session.commit()

        blob_service_client = BlobServiceClient(account_url=sas_url)
        container_client = blob_service_client.get_container_client(container_name)
        logger.info("Connected to Azure Blob Storage")

        process_blobs(container_client)

        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.SUCCESS})
        db_session.commit()

    except Exception as e:
        logger.error(f"Error in pull_files_from_azure_source_task: {e}")
        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.FAILED})
        db_session.commit()

        if self.request.retries >= self.max_retries:
            logger.error("Max retries reached for task. Status set to Stopped.")
        raise self.retry(exc=e, countdown=0)
    finally:
        db_session.close()

    return pulled_files


def publish_pull_file_status(user_id, file):
    publish_websocket_message(
        f"{user_id}:{ITaskType.pull_files}",
        IFileUploadRead(
            filename=deps.get_filename(file_name=file.filename),
            mimetype=file.mimetype,
            size=file.size,
            status=file.status,
            id=file.id,
        ).model_dump_json(),
    )
