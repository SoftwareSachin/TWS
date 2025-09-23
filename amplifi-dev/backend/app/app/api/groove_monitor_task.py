import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional  # Add other types as needed
from uuid import UUID

import requests
from sqlalchemy.orm import Session

from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.source_connector_crud import CRUDSource
from app.db.session import SyncSessionLocal
from app.models import Source
from app.models.dataset_model import Dataset
from app.models.file_model import File
from app.models.groove_source_model import GrooveSource
from app.utils.datetime_utils import ensure_naive_datetime
from app.utils.feature_flags import FeatureFlags, is_feature_enabled_sync

crud = CRUDSource(Source)


@celery.task(name="tasks.monitor_groove_sources_task")
def monitor_groove_sources_task():
    """
    Enhanced periodic task to monitor Groove sources for new tickets.
    This is the primary mechanism for auto-detection in multi-tenant scenarios.
    """
    logger.info("Starting enhanced Groove monitoring task")

    # Check if groove_connector_feature is enabled via Flagsmith
    if not is_feature_enabled_sync(feature_name=FeatureFlags.GROOVE_CONNECTOR_FEATURE):
        logger.debug(
            "groove_connector_feature is disabled via Flagsmith, skipping Groove monitoring"
        )
        return

    db_session: Session = SyncSessionLocal()

    def monitor_all():
        try:
            # Get all Groove sources with auto-detection enabled
            groove_sources = (
                db_session.query(GrooveSource)
                .filter(GrooveSource.auto_detection_enabled)
                .filter(GrooveSource.deleted_at.is_(None))
                .all()
            )
            logger.info(
                f"Found {len(groove_sources)} Groove sources with auto-detection enabled"
            )

            # Only monitor sources that are due for monitoring
            sources_to_monitor = [
                src for src in groove_sources if _should_monitor_groove_source(src)
            ]
            logger.info(
                f"Monitoring {len(sources_to_monitor)} sources that are due for monitoring."
            )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tasks = [
                loop.run_in_executor(
                    None, _monitor_single_groove_source_sync, groove_source, db_session
                )
                for groove_source in sources_to_monitor
            ]
            loop.run_until_complete(asyncio.gather(*tasks))
        except Exception as e:
            logger.error(f"Error in Groove monitoring task: {str(e)}")
        finally:
            db_session.close()

    return monitor_all()


def _should_monitor_groove_source(groove_source: GrooveSource) -> bool:
    """Check if source should be monitored based on frequency."""
    last_monitored = ensure_naive_datetime(groove_source.last_monitored)
    if last_monitored is None:
        return True

    time_since_last = ensure_naive_datetime(datetime.utcnow()) - last_monitored
    frequency_minutes = groove_source.monitoring_frequency_minutes or 30
    return time_since_last.total_seconds() > frequency_minutes * 60


def _monitor_single_groove_source_sync(
    groove_source: GrooveSource, db_session: Session
):
    """Enhanced monitoring of a single Groove source for new tickets."""
    try:
        logger.info(f"Monitoring Groove source {groove_source.source_id}")

        # Fetch the actual API key from the vault (sync)
        details = crud.get_groove_details_sync(
            UUID(str(groove_source.source_id)), db_session
        )
        api_key = details["api_key"]

        # Find the dataset for this source
        dataset = (
            db_session.query(Dataset)
            .filter(
                Dataset.source_id == groove_source.source_id,
                Dataset.deleted_at.is_(None),
            )
            .first()
        )
        if not dataset:
            raise ValueError(f"No dataset found for source {groove_source.source_id}")

        # Get chunking config for this dataset
        from app.crud.ingest_crud_v2 import ingestion_crud

        chunking_config_dict, chunking_config_id = (
            ingestion_crud.get_chunking_config_for_dataset_sync(
                dataset_id=dataset.id, db_session=db_session
            )
        )
        if not chunking_config_dict:
            raise ValueError(f"No chunking config found for dataset {dataset.id}")

        # Get latest ticket number from Groove API
        latest_ticket_number = _get_latest_ticket_number_from_groove(api_key)
        if not latest_ticket_number:
            logger.warning(
                f"Could not fetch latest ticket number for source {groove_source.source_id}"
            )
            return

        # Determine range of tickets to check
        last_processed = groove_source.last_ticket_number or (latest_ticket_number - 10)
        start_ticket = last_processed + 1
        end_ticket = latest_ticket_number

        # Get list of existing files in database
        existing_files = (
            db_session.query(File)
            .filter(File.source_id == groove_source.source_id)
            .all()
        )
        existing_ticket_numbers = {
            int(
                f.filename.replace(f"ticket_{groove_source.source_id}_", "").replace(
                    ".md", ""
                )
            )
            for f in existing_files
            if f.filename.startswith(f"ticket_{groove_source.source_id}_")
            and f.filename.endswith(".md")
        }

        logger.debug(
            f"Found {len(existing_ticket_numbers)} existing tickets in database for source {groove_source.source_id}"
        )

        new_tickets = []
        updated_tickets = []

        # Check for new tickets
        for ticket_number in range(start_ticket, end_ticket + 1):
            if ticket_number not in existing_ticket_numbers:
                new_tickets.append(ticket_number)
        if groove_source.re_ingest_updated_tickets:
            since_time = groove_source.last_monitored or (
                datetime.utcnow() - timedelta(hours=24)
            )

            all_updated_tickets = _check_ticket_updated(api_key, since_time)

            # Only include tickets that exist in our database (we don't re-ingest tickets we've never seen)
            updated_tickets = [
                ticket
                for ticket in all_updated_tickets
                if ticket in existing_ticket_numbers
            ]

        logger.info(
            f"Found {len(new_tickets)} new tickets, {len(updated_tickets)} updated tickets in source {groove_source.source_id}"
        )

        # Process new tickets with deduplication
        for ticket_number in new_tickets:
            logger.info(f"Processing new ticket: {ticket_number}")

            # Check if this ticket is already being processed
            try:
                from app.api.deps import get_redis_client_sync

                redis_client = get_redis_client_sync()
                dedup_key = f"groove_ticket_processing:{groove_source.source_id}:{ticket_number}"

                # Check if already being processed
                if redis_client.exists(dedup_key):
                    logger.info(
                        f"Ticket {ticket_number} already being processed, skipping"
                    )
                    continue

            except Exception as e:
                logger.warning(f"Failed to check deduplication: {str(e)}")

            celery.send_task(
                "tasks.process_new_groove_ticket_task",
                kwargs={
                    "source_id": str(groove_source.source_id),
                    "ticket_number": ticket_number,
                    "event_time": datetime.utcnow().isoformat(),
                    "dataset_id": str(dataset.id),
                    "chunking_config": (
                        chunking_config_dict if chunking_config_dict else None
                    ),
                    "organization_id": str(dataset.workspace_id),
                },
                queue="file_pull_queue",
            )

        # Process updated tickets based on source-level setting
        if updated_tickets:
            if groove_source.re_ingest_updated_tickets:
                logger.info(
                    f"Re-ingestion enabled for source {groove_source.source_id}: Processing {len(updated_tickets)} updated tickets"
                )
                for ticket_number in updated_tickets:
                    logger.info(f"Re-ingesting updated ticket: {ticket_number}")

                    # Check if this ticket is already being processed
                    try:
                        from app.api.deps import get_redis_client_sync

                        redis_client = get_redis_client_sync()
                        dedup_key = f"groove_ticket_update_processing:{groove_source.source_id}:{ticket_number}"

                        # Check if already being processed
                        if redis_client.exists(dedup_key):
                            logger.info(
                                f"Updated ticket {ticket_number} already being processed, skipping"
                            )
                            continue

                    except Exception as e:
                        logger.warning(f"Failed to check deduplication: {str(e)}")

                    celery.send_task(
                        "tasks.process_updated_groove_ticket_task",
                        kwargs={
                            "source_id": str(groove_source.source_id),
                            "ticket_number": ticket_number,
                            "event_time": datetime.utcnow().isoformat(),
                            "dataset_id": str(dataset.id),
                            "chunking_config": (
                                chunking_config_dict if chunking_config_dict else None
                            ),
                            "organization_id": str(dataset.workspace_id),
                        },
                        queue="file_pull_queue",
                    )
            else:
                logger.info(
                    f"Re-ingestion disabled for source {groove_source.source_id}: Logging {len(updated_tickets)} updated tickets without re-ingestion"
                )
                for ticket_number in updated_tickets:
                    logger.info(
                        f"Updated ticket detected (re-ingestion disabled for source {groove_source.source_id}): ticket_{ticket_number}.md"
                    )

        # Update monitoring metadata
        groove_source.last_monitored = datetime.utcnow()
        groove_source.last_ticket_number = latest_ticket_number
        db_session.commit()

        if new_tickets:
            logger.info(
                f"Successfully queued {len(new_tickets)} new tickets for processing in source {groove_source.source_id}"
            )

        if updated_tickets:
            if groove_source.re_ingest_updated_tickets:
                logger.info(
                    f"Successfully queued {len(updated_tickets)} updated tickets for re-ingestion in source {groove_source.source_id}"
                )
            else:
                logger.info(
                    f"Logged {len(updated_tickets)} updated tickets without re-ingestion in source {groove_source.source_id}"
                )

        # NOTE: Ingestion is now handled individually by each process_new_groove_ticket_task
        # This provides better error isolation and follows the Azure pattern

    except Exception as e:
        logger.error(f"Error monitoring source {groove_source.source_id}: {str(e)}")


def _get_latest_ticket_number_from_groove(api_key: str) -> Optional[int]:
    """Get the latest ticket number from Groove API."""
    query = """
    query {
      conversations(first: 1, orderBy: {field: CREATED_AT, direction: DESC}) {
        edges {
          node {
            number
          }
        }
      }
    }
    """
    try:
        response = requests.post(
            settings.GROOVE_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            logger.error(f"API error getting latest ticket: {data['errors']}")
            return None

        edges = data.get("data", {}).get("conversations", {}).get("edges", [])
        if edges:
            return edges[0]["node"]["number"]
        return None
    except Exception as e:
        logger.error(f"Failed to get latest ticket number: {e}")
        return None


def _check_ticket_updated(api_key: str, last_time: datetime) -> List[int]:
    """
    Get list of ticket numbers that have been updated since the given time.
    Uses pagination to fetch all updated tickets

    Args:
        api_key: GrooveHQ API token
        last_time: datetime object to check updates since (from DB; naive or aware)
    Returns:
        list: List of ticket numbers (integers) that were updated
    """

    def normalize_datetime_to_utc(dt: Optional[datetime]) -> datetime:
        """Normalize datetime to UTC-aware datetime"""
        if dt is None:
            logger.info("datetime is None, defaulting to 24h ago")
            return datetime.now(timezone.utc) - timedelta(hours=24)

        if dt.tzinfo is None:
            # Assume naive datetime is UTC
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC
            return dt.astimezone(timezone.utc)

    def parse_groove_datetime(date_str: str) -> datetime:
        """Parse Groove API datetime string - EXACTLY like original code"""
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # Normalize to UTC if needed (keeping original logic)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def process_conversation_batch(
        edges: List[dict], last_time_utc: datetime
    ) -> List[int]:
        """Process a batch of conversations and return ticket numbers that were updated"""
        ticket_numbers = []

        for edge in edges:
            try:
                node = edge["node"]
                ticket_number = node["number"]

                # Parse timestamps
                created_at = parse_groove_datetime(node["createdAt"])
                updated_at = parse_groove_datetime(node["updatedAt"])

                # Check if ticket was actually updated (not just created) and after our cutoff
                if updated_at != created_at and updated_at >= last_time_utc:
                    ticket_numbers.append(ticket_number)
                    logger.debug(
                        f"Ticket #{ticket_number}: updated_at={updated_at} >= last_time={last_time_utc}"
                    )
                else:
                    logger.debug(
                        f"Ticket #{ticket_number}: skipped (created_at={created_at}, updated_at={updated_at})"
                    )

            except KeyError as e:
                logger.error(f"Missing required field in conversation data: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing conversation: {e}")
                continue

        return ticket_numbers

    def should_stop_pagination(last_edge: dict, last_time_utc: datetime) -> bool:
        """Check if we should stop pagination based on the last conversation's update time"""
        try:
            node = last_edge["node"]
            updated_at = parse_groove_datetime(node["updatedAt"])

            # If the most recent conversation in this batch is older than our cutoff,
            # all subsequent pages will be even older
            return updated_at < last_time_utc
        except Exception as e:
            logger.debug(f"Could not determine if pagination should stop: {e}")
            return False  # Continue pagination to be safe

    # Main function logic starts here
    GRAPHQL_ENDPOINT = settings.GROOVE_API_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Normalize last_time to UTC-aware
    last_time_utc = normalize_datetime_to_utc(last_time)
    logger.info(f"Checking for tickets updated since: {last_time_utc}")

    all_ticket_numbers = []
    cursor = None
    batch_size = 200  # Smaller batches for better performance
    total_processed = 0

    while True:
        # Build query with pagination support
        query = """
        query GetRecentlyUpdatedConversations($first: Int!, $orderBy: ConversationOrder!, $after: String) {
          conversations(first: $first, orderBy: $orderBy, after: $after) {
            totalCount
            edges {
              node {
                id
                number
                createdAt
                updatedAt
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """

        variables = {
            "first": batch_size,
            "orderBy": {"field": "UPDATED_AT", "direction": "DESC"},
        }
        if cursor:
            variables["after"] = cursor

        try:
            response = requests.post(
                GRAPHQL_ENDPOINT,
                headers=headers,
                json={"query": query, "variables": variables},
                timeout=10,
            )
            response.raise_for_status()

        except requests.exceptions.Timeout:
            logger.error("Timeout fetching updated tickets")
            break  # Return partial results
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching updated tickets: {e}")
            break  # Return partial results

        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Invalid JSON response: {e}")
            break

        if "errors" in data:
            logger.error("GraphQL Errors: %s", data["errors"])
            break  # Return partial results

        # Extract conversation data
        conversations = data.get("data", {}).get("conversations", {})
        edges = conversations.get("edges", [])
        page_info = conversations.get("pageInfo", {})

        if not edges:
            logger.info("No more conversations found")
            break

        # Process this batch
        batch_tickets = process_conversation_batch(edges, last_time_utc)
        all_ticket_numbers.extend(batch_tickets)
        total_processed += len(edges)

        logger.debug(
            f"Processed {len(edges)} conversations, found {len(batch_tickets)} updated tickets"
        )

        # Check if we should continue
        if not page_info.get("hasNextPage", False):
            logger.info("Reached end of conversations")
            break

        cursor = page_info.get("endCursor")
        if not cursor:
            logger.warning("No end cursor provided but hasNextPage=true")
            break

        # Early termination optimization: if we're getting conversations older than our cutoff,
        # we can stop since they're ordered by updatedAt DESC
        if edges and should_stop_pagination(edges[-1], last_time_utc):
            logger.info(
                "Reached conversations older than cutoff time, stopping pagination"
            )
            break

        # Safety check to prevent runaway requests
        if total_processed > 10000:
            logger.warning("Hit safety limit of 10000 conversations processed")
            break

    logger.info(
        f"Found {len(all_ticket_numbers)} updated tickets from {total_processed} conversations processed"
    )
    return sorted(set(all_ticket_numbers))  # Remove duplicates and sort
