import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
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
    """Get dataset and chunking config for a source - same as Azure implementation"""
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


def _download_groove_ticket_to_local(
    ticket_number: int, api_key: str, source_id: str, metadata: Optional[dict] = None
) -> str:
    """
    Download Groove ticket to local storage for processing.
    Returns the local file path for processing.
    """
    GROOVE_SOURCE_DIR = Path(settings.GROOVE_SOURCE_DIR)

    # Ensure directory exists
    GROOVE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename and path
    filename = f"ticket_{source_id}_{ticket_number}.md"
    file_path = GROOVE_SOURCE_DIR / filename

    # Remove existing file if it exists to ensure clean overwrite
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"Removed existing file before overwriting: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove existing file {file_path}: {str(e)}")

    # Fetch ticket content from Groove API
    try:
        content = _fetch_groove_ticket_content(ticket_number, api_key)

        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Downloaded Groove ticket {ticket_number} to {file_path}")
        return str(file_path)
    except Exception as e:
        # Clean up file on error
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass
        raise e


def _fetch_groove_ticket_content(ticket_number: int, api_key: str) -> str:
    """Fetch ticket content from Groove API and format as markdown"""
    GROOVE_API_URL = settings.GROOVE_API_URL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def clean_text(text: str) -> str:
        """Clean and format text content"""
        if not text:
            return ""
        cleaned = re.sub(r"\n\s*\n+", "\n\n", text).replace("\t", " ")
        lines = [line.strip() for line in cleaned.split("\n")]
        return "\n".join(lines).strip()

    # Fetch ticket details
    ticket_query = """
    query q($n: Int!) {
      conversation(number: $n) {
        id
        number
        subject
        state
      }
    }
    """

    try:
        resp = requests.post(
            GROOVE_API_URL,
            headers=headers,
            json={"query": ticket_query, "variables": {"n": ticket_number}},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            raise Exception(f"API error for ticket #{ticket_number}: {data['errors']}")

        ticket = data.get("data", {}).get("conversation")
        if not ticket:
            raise Exception(f"Ticket #{ticket_number} not found")

    except Exception as e:
        logger.error(f"Failed to fetch ticket #{ticket_number}: {e}")
        raise

    # Fetch conversation history
    history_query = """
    query q($id: ID!) {
      eventGroups(filter: {conversationId: $id}, first: 500) {
        edges {
          node {
            actor {
              __typename
              ...on Agent {name}
              ...on Contact {name}
            }
            events {
              edges {
                node {
                  createdAt
                  change {
                    __typename
                    ...on Note {bodyPlainText}
                    ...on EmailMessage {bodyPlainText}
                    ...on Reply {bodyPlainText}
                    ...on StateChanged {from to}
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    try:
        resp = requests.post(
            GROOVE_API_URL,
            headers=headers,
            json={"query": history_query, "variables": {"id": ticket["id"]}},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            logger.warning(
                f"API error on conversation history {ticket['id']}: {data['errors']}"
            )
            history = []
        else:
            history = data.get("data", {}).get("eventGroups", {}).get("edges", [])

    except Exception as e:
        logger.warning(f"Failed to fetch conversation history {ticket['id']}: {e}")
        history = []

    # Generate markdown content
    content = f"# Ticket #{ticket_number}\n\n"
    content += f"**Subject:** {ticket.get('subject', 'No subject')}\n"
    content += f"**State:** {ticket.get('state', 'Unknown')}\n\n"
    content += "---\n\n## Conversation History\n\n"

    if history:
        for group in history:
            node = group.get("node", {})
            actor = node.get("actor", {})
            for event_edge in node.get("events", {}).get("edges", []):
                event = event_edge.get("node", {})
                change = event.get("change", {})
                content += f"**From:** {actor.get('name', 'System')} ({actor.get('__typename')})  \n"
                content += f"**At:** {event.get('createdAt')}  \n\n"

                if "bodyPlainText" in change and change["bodyPlainText"]:
                    cleaned_body = clean_text(change["bodyPlainText"])
                    content += "```text\n" + cleaned_body + "\n```\n\n"
                elif change.get("__typename") == "StateChanged":
                    content += f"> *State changed from **{change.get('from')}** to **{change.get('to')}***\n\n"
                else:
                    content += f"> *Action: {change.get('__typename')}*\n\n"
                content += "---\n\n"
    else:
        content += "No conversation history found.\n"

    return content


@celery.task(name="tasks.process_new_groove_ticket_task", bind=True, max_retries=3)
def process_new_groove_ticket_task(
    self,
    source_id: str,
    ticket_number: int,
    event_time: str,
    dataset_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a newly detected Groove ticket and trigger ingestion.
    Uses polling-based detection mechanism with deduplication.
    Follows the same pattern as Azure blob processing.
    """
    # Create a unique key for this ticket processing task
    dedup_key = f"groove_ticket_processing:{source_id}:{ticket_number}"

    # Try to acquire a lock to prevent duplicate processing
    try:
        from app.api.deps import get_redis_client_sync

        redis_client = get_redis_client_sync()

        # Try to set the key with expiration (30 seconds) - short enough for retries, long enough to prevent duplicates
        lock_acquired = redis_client.set(dedup_key, "processing", ex=30, nx=True)

        if not lock_acquired:
            logger.info(
                f"Task already being processed for ticket {ticket_number}, skipping duplicate"
            )
            return {
                "status": "skipped",
                "reason": "duplicate_task",
                "ticket_number": ticket_number,
            }

    except Exception as e:
        logger.warning(
            f"Failed to acquire deduplication lock: {str(e)}, proceeding anyway"
        )
        redis_client = None

    logger.info(
        f"Processing new Groove ticket: {ticket_number} from source {source_id}"
    )

    db_session: Session = SyncSessionLocal()

    try:
        # Get source details
        source = db_session.query(Source).filter(Source.id == source_id).first()
        if not source or source.source_type != "groove_source":
            raise ValueError(f"Invalid source {source_id} or not Groove source")

        # Get Groove API credentials from vault
        groove_details = crud.get_groove_details_sync(source_id, db_session)
        if not groove_details:
            raise ValueError(f"Groove details not found for source {source_id}")
        api_key = groove_details["api_key"]

        # Check if file already exists
        filename = f"ticket_{source_id}_{ticket_number}.md"
        existing_file = (
            db_session.query(File)
            .filter(File.filename == filename, File.source_id == source_id)
            .first()
        )

        if existing_file:
            logger.info(f"Ticket {ticket_number} already exists, skipping")
            return {"status": "skipped", "reason": "already_exists"}

        # Download ticket to local storage
        metadata = {
            "source": "groove_source_auto_detection",
            "ticket_number": ticket_number,
            "event_time": event_time,
            "source_id": source_id,
        }

        download_file_path = _download_groove_ticket_to_local(
            ticket_number, api_key, source_id, metadata
        )

        # Get file size
        file_size = os.path.getsize(download_file_path)

        # Create file record
        file_record = File(
            filename=filename,
            mimetype="text/markdown",
            size=file_size,
            file_path=download_file_path,
            status=FileStatusEnum.Uploaded,
            source_id=source_id,
            workspace_id=source.workspace_id,
        )

        db_session.add(file_record)
        db_session.commit()
        db_session.refresh(file_record)

        logger.info(f"File record created for {filename}")

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

        # Trigger ingestion of new file
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
            "ticket_number": ticket_number,
        }

    except Exception as e:
        logger.error(f"Error processing new ticket {ticket_number}: {str(e)}")
        if self.request.retries >= self.max_retries:
            logger.error(f"Max retries reached for ticket {ticket_number}")
        raise self.retry(exc=e, countdown=60)
    finally:
        # Clean up the deduplication lock
        if "redis_client" in locals() and redis_client:
            try:
                redis_client.delete(dedup_key)
            except Exception as e:
                logger.warning(f"Failed to clean up deduplication lock: {str(e)}")

        db_session.close()


@celery.task(name="tasks.process_updated_groove_ticket_task", bind=True, max_retries=3)
def process_updated_groove_ticket_task(
    self,
    source_id: str,
    ticket_number: int,
    event_time: str,
    dataset_id: Optional[str] = None,
    chunking_config: Optional[dict] = None,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an updated Groove ticket.
    This can either re-ingest the ticket or just update metadata.
    """
    # Create a unique key for this ticket processing task
    dedup_key = f"groove_ticket_update_processing:{source_id}:{ticket_number}"

    # Try to acquire a lock to prevent duplicate processing
    try:
        from app.api.deps import get_redis_client_sync

        redis_client = get_redis_client_sync()

        # Try to set the key with expiration (30 seconds) - short enough for retries, long enough to prevent duplicates
        lock_acquired = redis_client.set(dedup_key, "processing", ex=30, nx=True)

        if not lock_acquired:
            logger.info(
                f"Update task already being processed for ticket {ticket_number}, skipping duplicate"
            )
            return {
                "status": "skipped",
                "reason": "duplicate_update_task",
                "ticket_number": ticket_number,
            }

    except Exception as e:
        logger.warning(
            f"Failed to acquire deduplication lock: {str(e)}, proceeding anyway"
        )
        redis_client = None

    logger.info(
        f"Processing updated Groove ticket: {ticket_number} from source {source_id}"
    )
    db_session: Session = SyncSessionLocal()

    try:
        filename = f"ticket_{source_id}_{ticket_number}.md"
        existing_file = (
            db_session.query(File)
            .filter(File.filename == filename, File.source_id == source_id)
            .first()
        )

        # Get Groove API credentials from vault
        groove_details = crud.get_groove_details_sync(source_id, db_session)
        if not groove_details:
            raise ValueError(f"Groove details not found for source {source_id}")
        api_key = groove_details["api_key"]

        if not existing_file:
            logger.warning(
                f"Updated ticket {ticket_number} not found in database, treating as new"
            )
            return {
                "status": "success",
                "file_id": str(existing_file.id) if existing_file else "none",
                "ticket_number": ticket_number,
                "action": "re_ingestion_not_triggered",
            }

        # Download ticket to local storage
        metadata = {
            "source": "groove_source_auto_detection",
            "ticket_number": ticket_number,
            "event_time": event_time,
            "source_id": source_id,
            "update_type": "ticket_updated",
        }

        download_file_path = _download_groove_ticket_to_local(
            ticket_number, api_key, source_id, metadata
        )
        existing_file.file_path = download_file_path

        # Update file metadata
        file_size = os.path.getsize(download_file_path)
        existing_file.size = file_size
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
            "ticket_number": ticket_number,
            "action": "re_ingestion_triggered",
        }
    except Exception as e:
        logger.error(f"Error processing updated ticket {ticket_number}: {str(e)}")
        if self.request.retries >= self.max_retries:
            logger.error(f"Max retries reached for updated ticket {ticket_number}")
        raise self.retry(exc=e, countdown=60)
    finally:
        # Clean up the deduplication lock
        if "redis_client" in locals() and redis_client:
            try:
                redis_client.delete(dedup_key)
            except Exception as e:
                logger.warning(f"Failed to clean up deduplication lock: {str(e)}")

        db_session.close()
