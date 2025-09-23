from datetime import datetime

from celery import chain

from app import crud
from app.be_core.celery import celery
from app.be_core.logger import logger
from app.crud.ingest_crud import ingestion_crud
from app.db.session import SyncSessionLocal as db_session
from app.deps.celery_deps import get_job_db
from app.models.workflow_run_model import WorkflowRun
from app.schemas.workflow_schema import WorkflowRunStatusType
from app.utils.workflow_utils import (  # R2R functionality disabled; fetch_vectors_from_r2r,
    process_files_for_dataset,
    publish_workflow_status,
    schedule_workflow,
)


@celery.task(name="tasks.execute_workflow_task", bind=True)
def execute_workflow_task(self, organization_id, user_id, workflow_id):
    logger.info(
        f"Starting execute_workflow_task for Organization ID: {organization_id}, Workflow ID: {workflow_id}"
    )
    logger.debug(
        f"Task Details - Organization ID: {organization_id}, Workflow ID: {workflow_id}, Task Name: {self.name}, Task ID: {self.request.id}"
    )

    workflow_run = None
    try:
        with db_session() as session:

            # Check if any previous runs are in progress
            in_progress_runs = (
                session.query(WorkflowRun)
                .filter(
                    WorkflowRun.workflow_id == workflow_id,
                    WorkflowRun.status == WorkflowRunStatusType.Processing,
                )
                .all()
            )

            if in_progress_runs:
                logger.warning(
                    f"There are {len(in_progress_runs)} workflow runs in progress for workflow ID {workflow_id}"
                )

            workflow_with_schedule = crud.workflow.get_workflow_by_id_sync(
                workflow_id=workflow_id,
                organization_id=organization_id,
                db_session=session,
            )

            workflow = workflow_with_schedule.workflow
            schedule_config = workflow_with_schedule.schedule_config

            logger.debug(f"Workflow: {workflow}")
            logger.debug(f"Schedule Config: {schedule_config}")

            if (
                schedule_config
                and schedule_config.cron_expression
                and not workflow.scheduled
            ):
                celery_session = next(get_job_db())
                schedule_workflow(workflow, user_id, schedule_config, celery_session)
                workflow.scheduled = True
                session.commit()
                session.refresh(workflow)
            else:
                workflow_with_schedule_dict = {
                    "workflow": workflow.dict(),
                    "schedule_config": (
                        schedule_config.dict() if schedule_config else None
                    ),
                }

                workflow_run_id = crud.workflow.create_workflow_run_sync(
                    workflow_id, session
                )
                workflow_run = dict(
                    crud.workflow.get_workflow_run_by_id_sync(
                        workflow_run_id, workflow_id, session
                    )
                )
                logger.debug(workflow_run)

                logger.debug(workflow_with_schedule_dict)

                chain(
                    fetch_vectors_task.s(
                        organization_id,
                        user_id,
                        workflow_id,
                        workflow_run,
                        workflow_with_schedule_dict,
                    ),
                    store_vectors_task.s(),
                ).apply_async()
    except Exception as e:
        logger.error(f"Error in execute workflow task: {e}")
        if workflow_run:
            crud.workflow.update_workflow_run_attribute_sync(
                workflow_run["run_id"], "status", WorkflowRunStatusType.Failed, session
            )
            publish_workflow_status(user_id, workflow_id, workflow_run["run_id"])
    finally:
        session.close()


@celery.task(name="tasks.fetch_vectors_task")
def fetch_vectors_task(
    organization_id, user_id, workflow_id, workflow_run, workflow_with_schedule
):
    logger.info(
        f"Starting fetch_vectors_task. Organization ID: {organization_id}, Workflow ID: {workflow_id}, Workflow Run ID: {workflow_run['run_id']}"
    )

    vectors = []
    try:
        with db_session() as session:

            # Set the workflow_run status to Processing
            if workflow_run:
                crud.workflow.update_workflow_run_attribute_sync(
                    workflow_run["run_id"],
                    "status",
                    WorkflowRunStatusType.Processing,
                    session,
                )
                publish_workflow_status(user_id, workflow_id, workflow_run["run_id"])

            workflow = workflow_with_schedule["workflow"]
            dataset_id = workflow["dataset_id"]

            # Retrieve list of files associated with the dataset_id
            current_files = ingestion_crud.get_files_for_dataset_sync(
                dataset_id=dataset_id, db_session=session
            )

            logger.info(
                f"Processing {len(current_files)} files for dataset {dataset_id}"
            )

            # Fetch vectors for each file
            for file in current_files:
                logger.info(f"Fetching vectors for file {file.id}")

                # Check if file requires splitting by checking the requires_splitting flag
                if hasattr(file, "requires_splitting") and file.requires_splitting:
                    logger.info(
                        f"File {file.id} requires splitting, will handle split vectors"
                    )

                # Fetch vectors using the updated function that handles splits
                # R2R functionality disabled
                logger.warning("R2R functionality disabled - skipping vector fetch")
                continue

                # file_vectors = fetch_vectors_from_r2r(
                #    organization_id, file.id, dataset_id, session
                # )

                # R2R functionality disabled - skip vector collection
                # vectors.append({"file_id": file.id, "vectors": file_vectors})

            return {
                "organization_id": organization_id,
                "user_id": user_id,
                "workflow_id": workflow_id,
                "workflow_run": workflow_run,
                "workflow_with_schedule": workflow_with_schedule,
                "vectors": vectors,
            }
    except Exception as e:
        logger.error(f"Error in fetch_vectors_task: {e}")
        if workflow_run:
            crud.workflow.update_workflow_run_attribute_sync(
                workflow_run["run_id"], "status", WorkflowRunStatusType.Failed, session
            )
            publish_workflow_status(user_id, workflow_id, workflow_run["run_id"])
    finally:
        session.close()


@celery.task(name="tasks.store_vectors_task")
def store_vectors_task(result):
    logger.info(
        f"Starting store_vectors_task. Organization ID: {result['organization_id']}, Workflow ID: {result['workflow_id']}, Workflow Run ID: {result['workflow_run']['run_id']}"
    )
    session = None

    try:
        if "vectors" not in result or not isinstance(result["vectors"], list):
            logger.error(
                f"❌ Missing or invalid 'vectors' key in store_vectors_task: {result}"
            )
            return

        vectors = result["vectors"]

        session = db_session()
        process_files_for_dataset(
            result["workflow_with_schedule"], result["workflow_run"], session, vectors
        )
        logger.info("✅ Successfully processed vectors in store_vectors_task")

        crud.workflow.update_workflow_run_attribute_sync(
            result["workflow_run"]["run_id"],
            "status",
            WorkflowRunStatusType.Success,
            session,
        )
        crud.workflow.update_workflow_run_attribute_sync(
            result["workflow_run"]["run_id"], "finished_at", datetime.utcnow(), session
        )
        publish_workflow_status(
            result["user_id"], result["workflow_id"], result["workflow_run"]["run_id"]
        )
        logger.info(
            f"✅ Updated workflow_run {result['workflow_run']['run_id']} to SUCCESS."
        )
    except Exception as e:
        logger.error(f"Error in store_vectors_task: {e}", exc_info=True)
        if result["workflow_run"]:
            with db_session() as temp_session:
                crud.workflow.update_workflow_run_attribute_sync(
                    result["workflow_run"]["run_id"],
                    "status",
                    WorkflowRunStatusType.Failed,
                    temp_session,
                )
            publish_workflow_status(
                result["user_id"],
                result["workflow_id"],
                result["workflow_run"]["run_id"],
            )
    finally:
        if session:
            session.close()
