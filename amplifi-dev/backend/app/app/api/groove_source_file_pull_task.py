import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import requests
from celery import shared_task
from sqlalchemy.orm import Session

from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.source_connector_crud import CRUDSource
from app.db.session import SyncSessionLocal
from app.models.file_model import File
from app.models.groove_source_model import GrooveSource
from app.models.pull_status_model import PullStatusEnum
from app.models.source_model import Source, SourcePullStatus
from app.schemas.file_schema import FileStatusEnum

crud = CRUDSource(Source)


@shared_task(name="tasks.pull_files_from_groove_source_task", bind=True, max_retries=3)
def pull_files_from_groove_source_task(
    self,
    workspace_id: UUID,
    user_id: UUID,
    source_id: UUID,
    api_key: str,
    ticket_numbers: Optional[List[int]] = None,
    batch_size: int = 50,
    max_batches: Optional[int] = None,
    use_batch_mode: bool = True,
) -> List[Dict[str, Any]]:
    """
    Enhanced Celery task to fetch Groove tickets in batches and save them as files.
    """
    logger.info("Starting enhanced pull_files_from_groove_source_task")
    logger.info(f"Source ID: {source_id}")
    logger.info(f"Batch mode: {use_batch_mode}, Batch size: {batch_size}")
    if ticket_numbers:
        logger.info(f"Processing specific tickets: {ticket_numbers}")
    if max_batches:
        logger.info(f"Max batches limit: {max_batches}")

    db_session: Session = SyncSessionLocal()
    GROOVE_SOURCE_DIR = Path(settings.GROOVE_SOURCE_DIR)
    GROOVE_API_URL = settings.GROOVE_API_URL
    pulled_files: List[File] = []  # Keep as List[File] as requested

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

    def fetch_tickets_batch(after_cursor: Optional[str] = None, batch_size: int = 50):
        """Fetch a batch of tickets with pagination"""
        query = """
        query getBatch($first: Int!, $after: String) {
          conversations(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
            pageInfo {
              hasNextPage
              endCursor
            }
            edges {
              node {
                id
                number
                subject
                state
                createdAt
              }
            }
          }
        }
        """

        variables = {"first": batch_size}
        if after_cursor:
            variables["after"] = after_cursor

        try:
            logger.info(f"Fetching batch (size={batch_size}, cursor={after_cursor})")
            resp = requests.post(
                GROOVE_API_URL,
                headers=headers,
                json={"query": query, "variables": variables},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                logger.error(f"Batch fetch API error: {data['errors']}")
                return None, None, False

            conversations_data = data.get("data", {}).get("conversations", {})
            edges = conversations_data.get("edges", [])
            page_info = conversations_data.get("pageInfo", {})

            logger.info(f"Fetched {len(edges)} tickets in batch")
            return (
                edges,
                page_info.get("endCursor"),
                page_info.get("hasNextPage", False),
            )

        except Exception as e:
            logger.error(f"Failed to fetch batch: {e}")
            return None, None, False

    def parse_groove_datetime(date_str: str) -> datetime:
        """Parse Groove API datetime string - same logic as monitor task"""
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # Normalize to UTC if needed
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def is_ticket_within_timeframe(
        ticket_data: Dict[str, Any], cutoff_time: datetime
    ) -> bool:
        """Check if ticket is within the desired timeframe (last year)"""
        try:
            ticket = ticket_data.get("node", ticket_data)
            created_at_str = ticket.get("createdAt")
            if not created_at_str:
                return False

            created_at = parse_groove_datetime(created_at_str)
            return created_at >= cutoff_time
        except Exception as e:
            logger.warning(f"Could not parse ticket creation time: {e}")
            return False  # Skip tickets we can't parse

    def fetch_ticket_by_number(ticket_number: int):
        """Fetch individual ticket details (legacy method)"""
        query = """
        query q($n: Int!) {
          conversation(number: $n) {
            id
            number
            subject
            state
            createdAt
          }
        }
        """
        try:
            resp = requests.post(
                GROOVE_API_URL,
                headers=headers,
                json={"query": query, "variables": {"n": ticket_number}},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch ticket #{ticket_number}: {e}")
            return None

        data = resp.json()
        if "errors" in data:
            logger.error(f"API error for ticket #{ticket_number}: {data['errors']}")
            return None
        return data.get("data", {}).get("conversation")

    def fetch_conversation_history(conversation_id: str):
        """Fetch conversation history for a ticket"""
        query = """
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
                json={"query": query, "variables": {"id": conversation_id}},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch conversation history {conversation_id}: {e}")
            return []

        data = resp.json()
        if "errors" in data:
            logger.error(
                f"API error on conversation history {conversation_id}: {data['errors']}"
            )
            return []
        return data.get("data", {}).get("eventGroups", {}).get("edges", [])

    def process_ticket_from_data(ticket_data: Dict[str, Any]) -> Optional[File]:
        """Process a ticket from batch data and create file record"""
        nonlocal pulled_files

        # Handle both batch format and individual ticket format
        if "node" in ticket_data:
            ticket = ticket_data["node"]
        else:
            ticket = ticket_data

        ticket_number = ticket.get("number")
        conversation_id = ticket.get("id")

        if not ticket_number:
            logger.warning("Ticket missing number, skipping")
            return None

        logger.info(f"Processing ticket #{ticket_number}...")

        # Create filename and path
        filename = f"ticket_{source_id}_{ticket_number}.md"
        file_path = GROOVE_SOURCE_DIR / filename

        # Ensure directory exists
        GROOVE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

        # Create DB record first
        file = File(
            filename=filename,
            mimetype="text/markdown",
            size=0,  # Will be updated after writing
            file_path=str(file_path),
            status=FileStatusEnum.Uploading,
            source_id=source_id,
            workspace_id=workspace_id,
        )
        db_session.add(file)
        db_session.commit()
        db_session.refresh(file)
        logger.info(f"File record created for {filename}")

        try:
            # Generate markdown content
            content = f"# Ticket #{ticket_number}\n\n"
            content += f"**Subject:** {ticket.get('subject', 'No subject')}\n"
            content += f"**State:** {ticket.get('state', 'Unknown')}\n"
            content += f"**Created:** {ticket.get('createdAt', 'Unknown')}\n"
            content += f"**ID:** {conversation_id}\n\n"
            content += "---\n\n## Conversation History\n\n"

            # Fetch conversation history
            history = fetch_conversation_history(conversation_id)

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
                            content += f"``````\n{cleaned_body}\n``````\n\n"
                        elif change.get("__typename") == "StateChanged":
                            content += f"> *State changed from **{change.get('from')}** to **{change.get('to')}***\n\n"
                        else:
                            content += f"> *Action: {change.get('__typename')}*\n\n"
                        content += "---\n\n"
            else:
                content += "No conversation history found.\n"

            # Write content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Update file record with actual size
            file.size = len(content.encode("utf-8"))
            file.status = FileStatusEnum.Uploaded
            db_session.commit()

            # Add File object to pulled_files list
            pulled_files.append(file)
            logger.info(f"File {filename} created and saved successfully")
            return file

        except Exception as e:
            file.status = FileStatusEnum.Failed
            logger.error(f"Error processing ticket {ticket_number}: {e}")
            db_session.commit()
            raise e

    def process_ticket_by_number(ticket_number: int):
        """Process a single ticket by number (legacy method)"""
        # Fetch ticket details
        ticket = fetch_ticket_by_number(ticket_number)
        if not ticket:
            logger.warning(f"Ticket #{ticket_number} not found, skipping.")
            return None

        return process_ticket_from_data(ticket)

    def get_latest_ticket_number():
        """Fetch the latest ticket number from Groove API (fallback method)"""
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
            resp = requests.post(
                GROOVE_API_URL, headers=headers, json={"query": query}, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
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

    # Main execution starts here
    try:
        if not api_key:
            raise ValueError("API key is required")

        # Update pull status to STARTED
        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.STARTED})
        db_session.commit()

        # Process tickets based on input method
        if ticket_numbers:
            # Process specific tickets provided (legacy mode)
            logger.info(
                f"Processing {len(ticket_numbers)} specific tickets (legacy mode)"
            )
            for ticket_number in ticket_numbers:
                try:
                    process_ticket_by_number(ticket_number)
                except Exception as e:
                    logger.error(f"Failed to process ticket {ticket_number}: {e}")
                    if self.request.retries >= self.max_retries:
                        logger.error(f"Max retries reached for ticket {ticket_number}")
                    else:
                        self.retry(exc=e, countdown=60)

        elif use_batch_mode:
            # New batch processing mode with time filtering
            logger.info("Using batch processing mode with last year filter")

            # Calculate cutoff time for last year
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            cutoff_time = today_start - timedelta(days=settings.GROOVE_DAYS_TO_INGEST)
            logger.info(f"Only processing tickets created after: {cutoff_time}")

            cursor = None
            batch_number = 0
            total_processed = 0
            total_skipped_old = 0
            last_ticket_number = None

            while True:
                # Check batch limit
                if max_batches and batch_number >= max_batches:
                    logger.info(f"Reached max batches limit: {max_batches}")
                    break

                batch_number += 1
                logger.info(f"Processing batch #{batch_number}")

                # Fetch batch
                result = fetch_tickets_batch(cursor, batch_size)
                if not result or result[0] is None:
                    logger.error(f"Failed to fetch batch #{batch_number}")
                    break

                tickets_batch, next_cursor, has_next_page = result

                if not tickets_batch:
                    logger.info(f"No tickets in batch #{batch_number}, stopping")
                    break

                # Check if we should stop based on ticket age (early termination optimization)
                if tickets_batch and not is_ticket_within_timeframe(
                    tickets_batch[0], cutoff_time
                ):
                    logger.info(
                        f"Reached tickets older than 1 year in batch #{batch_number}, stopping"
                    )
                    break

                logger.info(
                    f"Processing {len(tickets_batch)} tickets in batch #{batch_number}"
                )

                # Process each ticket in the batch with time filtering
                batch_success = 0
                batch_failed = 0
                batch_skipped_old = 0

                for ticket_data in tickets_batch:
                    # Check if ticket is within our timeframe
                    if not is_ticket_within_timeframe(ticket_data, cutoff_time):
                        batch_skipped_old += 1
                        total_skipped_old += 1
                        continue

                    try:
                        file_result = process_ticket_from_data(ticket_data)
                        if file_result:
                            batch_success += 1
                            total_processed += 1
                            ticket = ticket_data.get("node", ticket_data)
                            ticket_number = ticket.get("number")
                            if ticket_number and (
                                last_ticket_number is None
                                or ticket_number > last_ticket_number
                            ):
                                last_ticket_number = ticket_number

                        else:
                            batch_failed += 1
                    except Exception as e:
                        batch_failed += 1
                        logger.error(f"Error in batch processing: {e}")
                        if self.request.retries >= self.max_retries:
                            logger.error("Max retries reached in batch processing")
                        else:
                            self.retry(exc=e, countdown=60)

                logger.info(
                    f"Batch #{batch_number} complete: {batch_success} success, {batch_failed} failed, {batch_skipped_old} skipped (too old)"
                )
                logger.info(
                    f"Total processed so far: {total_processed}, Total skipped (old): {total_skipped_old}"
                )

                # Check if we should continue
                if not has_next_page:
                    logger.info("No more pages available, batch processing complete")
                    break

                cursor = next_cursor

                # Small delay between batches to be nice to the API
                time.sleep(0.1)

            logger.info(
                f"Batch processing complete. Total processed: {total_processed}, Total skipped (old): {total_skipped_old}"
            )

        else:
            # Fallback to original behavior: Process last 10 tickets
            logger.info("Using fallback mode: processing last 10 tickets")
            latest_ticket_number = get_latest_ticket_number()
            if not latest_ticket_number:
                latest_ticket_number = 712356  # Fallback
                logger.warning(
                    f"Could not fetch latest ticket number, using fallback: {latest_ticket_number}"
                )
            else:
                logger.info(f"Latest ticket number found: {latest_ticket_number}")
            last_ticket_number = latest_ticket_number
            # Process last 10 tickets (configurable)
            ticket_count = 100
            for ticket_number in range(
                latest_ticket_number, latest_ticket_number - ticket_count, -1
            ):
                try:
                    process_ticket_by_number(ticket_number)
                except Exception as e:
                    logger.error(
                        f"Failed to process ticket {ticket_number} in fallback mode: {e}"
                    )
                    if self.request.retries >= self.max_retries:
                        logger.error(f"Max retries reached for ticket {ticket_number}")
                    else:
                        self.retry(exc=e, countdown=60)

        # Update pull status to SUCCESS
        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.SUCCESS})
        db_session.commit()

        db_session.query(GrooveSource).filter(
            GrooveSource.source_id == source_id
        ).update(
            {
                "last_monitored": datetime.utcnow(),
                "last_ticket_number": last_ticket_number,
            }
        )
        db_session.commit()

        logger.info(
            f"Completed Groove data fetch task. Processed {len(pulled_files)} tickets."
        )

    except Exception as e:
        logger.error(f"Error in pull_files_from_groove_source_task: {e}")
        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.FAILED})
        db_session.commit()

        if self.request.retries >= self.max_retries:
            logger.error("Max retries reached for task. Status set to Stopped.")
        raise self.retry(exc=e, countdown=60)
    finally:
        db_session.close()

    # Convert File objects to dictionaries for return (as expected by API)
    return pulled_files
