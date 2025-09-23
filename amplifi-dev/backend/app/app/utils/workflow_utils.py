from datetime import datetime

# R2R functionality disabled - removed unused imports
from uuid import UUID

from sqlalchemy_celery_beat.models import (
    CrontabSchedule,
    PeriodicTask,
)

from app import crud
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.file_split_model import FileSplit
from app.models.transferred_files_model import TransferredFiles
from app.schemas.long_task_schema import ITaskType
from app.schemas.workflow_schema import WorkflowRunStatusType
from app.utils.destination_writer import DestinationWriterFactory


def schedule_workflow(workflow, user_id, schedule_config, celery_session):
    try:
        logger.info("starting workflow scheduling")
        logger.debug(
            f"Workflow ID: {getattr(workflow, 'id', 'unknown')}, Cron: {getattr(schedule_config, 'cron_expression', 'unknown')}"
        )
        cron_expression = schedule_config.cron_expression
        workflow_id = workflow.id

        # Parse the cron expression
        cron_parts = cron_expression.split()
        minute, hour, day_of_month, month, day_of_week = cron_parts

        logger.debug(cron_parts)

        # Create CrontabSchedule
        schedule = CrontabSchedule(
            minute=minute,
            hour=hour,
            day_of_month=day_of_month,
            month_of_year=month,
            day_of_week=day_of_week,
            timezone="UTC",
        )
        celery_session.add(schedule)
        celery_session.commit()

        # Create PeriodicTask
        periodic_task = PeriodicTask(
            schedule_model=schedule,
            name=f"workflow_{workflow_id}",
            args=f'["{workflow.organization_id}", "{user_id}", "{workflow.id}"]',
            task="tasks.execute_workflow_task",
            one_off=False,
        )
        logger.debug(f"Periodic Task: {periodic_task}")
        celery_session.add(periodic_task)
        celery_session.commit()

        logger.debug(
            f"Workflow {workflow_id} scheduled with cron expression: {cron_expression}"
        )
    except Exception as e:
        logger.error(f"Error in schedule_workflow: {e}")
        raise
    finally:
        celery_session.close()


def publish_workflow_status(user_id, workflow_id, workflow_run_id):
    from app.api.deps import publish_websocket_message

    with SyncSessionLocal() as db:
        workflow_run = crud.workflow.get_workflow_run_by_id_sync(
            workflow_run_id, workflow_id, db
        )
        channel_name = f"{user_id}:{ITaskType.execute_workflow}"
        publish_websocket_message(channel_name, workflow_run.model_dump_json())


def check_and_fetch_file_splits(
    organization_id: UUID, file_id: UUID, dataset_id: UUID, db_session
):
    """
    Check if a file has splits and fetch their vectors.

    Args:
        organization_id: Organization ID
        file_id: Original file ID
        dataset_id: Dataset ID
        db_session: Database session

    Returns:
        Tuple of (has_splits: bool, split_vectors: list)
    """
    # Check if the file has splits in this dataset
    splits_query = (
        db_session.query(FileSplit)
        .filter(
            FileSplit.original_file_id == file_id,
            FileSplit.dataset_id == dataset_id,
        )
        .all()
    )

    # If no splits, return False and empty list
    if not splits_query:
        logger.debug(f"No splits found for file {file_id} in dataset {dataset_id}")
        return False, []

    logger.info(
        f"Found {len(splits_query)} splits for file {file_id} in dataset {dataset_id}"
    )

    # Initialize clients and variables
    # R2R functionality disabled
    logger.warning("R2R functionality disabled - skipping vector fetch for splits")
    return []


# R2R functionality disabled
def fetch_vectors_from_r2r(
    organization_id: UUID, file_id: UUID, dataset_id: UUID, db_session
):
    # R2R functionality disabled
    logger.warning("R2R functionality disabled - fetch_vectors_from_r2r called")
    return []


def vectors_to_destination(
    vectors,
    workflow_run,
    workflow_id,
    dataset_id,
    db_session,
    file_id,
    destination_writer,
):
    new_transferred_file = None
    try:
        # Strip out any split metadata from vectors before sending to destination
        processed_vectors = []
        for vector_data in vectors:
            processed_vector = {
                "file_name": vector_data["file_name"],
                "vector": vector_data["vector"],
            }
            processed_vectors.append(processed_vector)

        # Write vectors to the destination
        destination_writer.write(processed_vectors)

        # Mark the file as transferred
        new_transferred_file = TransferredFiles(
            file_id=file_id,
            workflow_run_id=workflow_run["run_id"],
            dataset_id=dataset_id,
            workflow_id=workflow_id,
            status="transferred",
        )
        db_session.add(new_transferred_file)

        # Commit the transaction if successful
        db_session.commit()
    except Exception as e:
        logger.error(f"Error in transferring vectors to destination database: {e}")
        db_session.rollback()
        # Update the status of the file transfer to "failed"
        new_transferred_file = TransferredFiles(
            file_id=file_id,
            workflow_run_id=workflow_run["run_id"],
            dataset_id=dataset_id,
            workflow_id=workflow_id,
            status="failed",
        )
        db_session.add(new_transferred_file)
        db_session.commit()
        raise  # Re-raise the exception to be caught by the caller


def process_files_for_dataset(
    workflow_with_schedule, workflow_run, db_session, vectors
):
    backup_data = False
    try:
        # Retrieve the workflow from the database
        workflow = workflow_with_schedule["workflow"]
        destination_id = workflow["destination_id"]
        organization_id = workflow["organization_id"]
        dataset_id = workflow["dataset_id"]

        # Retrieve list of files associated with the dataset_id
        dataset = crud.dataset_v2.get_dataset_sync(
            dataset_id=dataset_id, db_session=db_session
        )
        dataset_name = dataset.name

        # Retrieve the destination details from the database
        connection_details = (
            crud.destination_crud.get_destination_connection_details_by_id_sync(
                destination_id=destination_id,
                organization_id=organization_id,
                db_session=db_session,
            )
        )

        # Add dataset_name to connection_details
        connection_details["dataset_name"] = dataset_name

        logger.debug(f"Retrieved connection details for destination {destination_id}")

        # Get the appropriate destination writer
        destination_writer = DestinationWriterFactory.get(connection_details)
        logger.debug(f"Destination writer: {destination_writer}")

        # Backup the vectors table
        backup_data = destination_writer.backup()

        # Process all current files
        for vector_data in vectors:
            file_id = vector_data["file_id"]
            file_vectors = vector_data["vectors"]
            vectors_to_destination(
                file_vectors,
                workflow_run,
                workflow["id"],
                dataset_id,
                db_session,
                file_id,
                destination_writer,
            )

        # Delete the backup table after processing all files
        destination_writer.delete_backup()

        # Update workflow run status and finish time
        crud.workflow.update_workflow_run_attribute_sync(
            workflow_run["run_id"], "status", WorkflowRunStatusType.Success, db_session
        )
        crud.workflow.update_workflow_run_attribute_sync(
            workflow_run["run_id"], "finished_at", datetime.utcnow(), db_session
        )
    except Exception as e:
        logger.error(f"Error in processing files for dataset: {e}")
        # Restore the vectors table from the backup data
        if backup_data:
            restore_successful = destination_writer.restore()
            if restore_successful:
                logger.info(
                    f"Successfully restored vectors table from backup for dataset {dataset_name}"
                )
            else:
                logger.error(
                    f"Failed to restore vectors table from backup for dataset {dataset_name}"
                )
        raise
