import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Page, Params, paginate
from sqlalchemy import asc, desc
from sqlalchemy.orm import Session
from sqlalchemy_celery_beat.models import (
    PeriodicTask,
)
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app import crud
from app.api.workflow_exec_task import execute_workflow_task
from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.deps.celery_deps import get_job_db
from app.models import ScheduleConfig, WorkflowRun
from app.models.dataset_model import Dataset
from app.models.destination_model import Destination
from app.models.file_model import File
from app.models.workflow_model import Workflow
from app.schemas.common_schema import IOrderEnum
from app.schemas.workflow_response_schema import (
    workflow_response_scheduleConfig,
    workflow_response_schema,
)
from app.schemas.workflow_schema import (
    IWorkflowCreate,
    IWorkflowRunRead,
    IWorkflowUpdate,
    WorkflowRunStatusType,
)


class WorkflowWithScheduleConfig:
    def __init__(self, workflow: Workflow, schedule_config: Optional[ScheduleConfig]):
        self.workflow = workflow
        self.schedule_config = schedule_config


class CRUDWorkflow(CRUDBase[Workflow, IWorkflowCreate, IWorkflowUpdate]):
    async def _get_workflow(
        self,
        workflow_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession,
    ) -> Workflow:
        result = await db_session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.organization_id == organization_id,
                Workflow.deleted_at.is_(None),
            )
        )
        workflow = result.scalar_one_or_none()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow

    def _get_workflow_sync(
        self,
        workflow_id: UUID,
        organization_id: UUID,
        db_session: Session | None = None,
    ) -> Workflow:
        result = db_session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.organization_id == organization_id,
                Workflow.deleted_at.is_(None),
            )
        )
        workflow = result.scalar_one_or_none()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow

    async def _soft_delete(self, entity, db_session: AsyncSession):
        entity.deleted_at = datetime.now()
        await db_session.commit()
        await db_session.refresh(entity)

    def _format_schedule_config(self, schedule_config: ScheduleConfig):
        if not schedule_config:
            return None
        return workflow_response_scheduleConfig(
            cron_expression=schedule_config.cron_expression,
        )

    async def _get_dataset_and_destination_names(
        self,
        dataset_id: UUID,
        destination_id: UUID,
        db_session: AsyncSession,
    ) -> tuple[str, str]:
        dataset_name = await db_session.execute(
            select(Dataset.name).where(Dataset.id == dataset_id)
        )
        dataset_name = dataset_name.scalar_one_or_none()

        destination_name = await db_session.execute(
            select(Destination.name).where(Destination.id == destination_id)
        )
        destination_name = destination_name.scalar_one_or_none()

        if not dataset_name or not destination_name:
            raise HTTPException(
                status_code=404, detail="Dataset or Destination not found"
            )

        return dataset_name, destination_name

    async def create_workflow(
        self,
        *,
        obj_in: IWorkflowCreate,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> workflow_response_schema:
        db_session = db_session or super().get_db().session
        new_workflow = Workflow(organization_id=organization_id, **obj_in.dict())
        db_session.add(new_workflow)
        await db_session.flush()  # Ensure new_workflow.id is available

        schedule_config = None
        if obj_in.schedule_config:
            schedule_config_data = obj_in.schedule_config.dict()
            schedule_config = ScheduleConfig(
                workflow_id=new_workflow.id, **schedule_config_data
            )
            db_session.add(schedule_config)
            await db_session.flush()  # Ensure schedule_config is created

        await db_session.commit()
        await db_session.refresh(new_workflow)
        await db_session.refresh(schedule_config)

        dataset_name, destination_name = await self._get_dataset_and_destination_names(
            dataset_id=new_workflow.dataset_id,
            destination_id=new_workflow.destination_id,
            db_session=db_session,
        )

        return workflow_response_schema(
            **new_workflow.dict(),
            schedule_config=self._format_schedule_config(schedule_config),
            dataset_name=dataset_name,
            destination_name=destination_name,
        )

    async def get_all_workflows(
        self,
        *,
        organization_id: UUID,
        params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[workflow_response_schema]:
        db_session = db_session or super().get_db().session
        columns = Workflow.__table__.columns
        if order_by not in columns:
            order_by = "created_at"

        order_clause = asc(
            columns[order_by]
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = (
            select(Workflow)
            .where(
                Workflow.organization_id == organization_id,
                Workflow.deleted_at.is_(None),
            )
            .order_by(order_clause)
        )

        result = await db_session.execute(query)
        workflows = result.scalars().all()

        workflow_responses = []
        for workflow in workflows:
            dataset_name, destination_name = (
                await self._get_dataset_and_destination_names(
                    dataset_id=workflow.dataset_id,
                    destination_id=workflow.destination_id,
                    db_session=db_session,
                )
            )
            workflow_responses.append(
                workflow_response_schema(
                    **workflow.dict(),
                    schedule_config=self._format_schedule_config(
                        workflow.schedule_configs
                    ),
                    dataset_name=dataset_name,
                    destination_name=destination_name,
                )
            )

        return paginate(workflow_responses, params)

    async def update_workflow(
        self,
        *,
        workflow_id: UUID,
        organization_id: UUID,
        obj_in: IWorkflowUpdate,
        db_session: AsyncSession | None = None,
    ) -> workflow_response_schema:
        db_session = db_session or super().get_db().session
        workflow = await self._get_workflow(workflow_id, organization_id, db_session)

        # Update all fields of the workflow except schedule_config
        for key, value in obj_in.dict(exclude={"schedule_config"}).items():
            setattr(workflow, key, value)

        schedule_config = None
        if obj_in.schedule_config:
            schedule_config = await db_session.execute(
                select(ScheduleConfig).where(ScheduleConfig.workflow_id == workflow_id)
            )
            schedule_config = schedule_config.scalar_one_or_none()
            schedule_config_data = obj_in.schedule_config.dict(exclude_unset=True)
            if schedule_config:
                for key, value in schedule_config_data.items():
                    setattr(schedule_config, key, value)
            else:
                schedule_config = ScheduleConfig(
                    workflow_id=workflow_id, **schedule_config_data
                )
                db_session.add(schedule_config)

            # Update the existing PeriodicTask with the new cron expression using celery_session
            celery_session = next(get_job_db())
            periodic_task = celery_session.execute(
                select(PeriodicTask).where(
                    PeriodicTask.name == f"workflow_{workflow_id}"
                )
            )
            periodic_task = periodic_task.scalar_one_or_none()
            if periodic_task:
                crontab_schedule = periodic_task.schedule_model
                cron_parts = schedule_config.cron_expression.split()
                (
                    crontab_schedule.minute,
                    crontab_schedule.hour,
                    crontab_schedule.day_of_month,
                    crontab_schedule.month_of_year,
                    crontab_schedule.day_of_week,
                ) = cron_parts
                celery_session.add(crontab_schedule)
                celery_session.commit()

        await db_session.commit()
        await db_session.refresh(workflow)
        await db_session.refresh(schedule_config)

        dataset_name, destination_name = await self._get_dataset_and_destination_names(
            dataset_id=workflow.dataset_id,
            destination_id=workflow.destination_id,
            db_session=db_session,
        )

        return workflow_response_schema(
            **workflow.dict(),
            schedule_config=self._format_schedule_config(schedule_config),
            dataset_name=dataset_name,
            destination_name=destination_name,
        )

    async def stop_workflow(
        self,
        *,
        workflow_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> None:
        db_session = db_session or super().get_db().session
        celery_session = next(get_job_db())
        try:
            workflow = await db_session.execute(
                select(Workflow).where(Workflow.id == workflow_id)
            )
            workflow = workflow.scalar_one_or_none()
            logger.info(f"workflow: {workflow}")
            if workflow:
                periodic_task = celery_session.execute(
                    select(PeriodicTask).where(
                        PeriodicTask.name == f"workflow_{workflow_id}"
                    )
                )
                periodic_task = periodic_task.scalar_one_or_none()
                logger.info(f"Periodic Task: {periodic_task}")
                if periodic_task:
                    periodic_task.enabled = False
                    celery_session.add(periodic_task)
                    celery_session.commit()

                # set is_active to False for the workflow
                workflow.is_active = False
                await db_session.commit()
                await db_session.refresh(workflow)
            else:
                raise HTTPException(status_code=404, detail="Workflow not found")

        finally:
            celery_session.close()

    async def start_workflow(
        self,
        *,
        workflow_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> None:
        db_session = db_session or super().get_db().session
        celery_session = next(get_job_db())
        try:
            workflow = await db_session.execute(
                select(Workflow).where(Workflow.id == workflow_id)
            )
            workflow = workflow.scalar_one_or_none()

            if workflow:
                periodic_task = celery_session.execute(
                    select(PeriodicTask).where(
                        PeriodicTask.name == f"workflow_{workflow_id}"
                    )
                )
                periodic_task = periodic_task.scalar_one_or_none()
                logger.info(f"Periodic Task: {periodic_task}")
                if periodic_task:
                    periodic_task.enabled = True
                    celery_session.add(periodic_task)
                    celery_session.commit()

                # set is_active to True for the workflow
                workflow.is_active = True
                await db_session.commit()
                await db_session.refresh(workflow)
            else:
                raise HTTPException(status_code=404, detail="Workflow not found")

        finally:
            celery_session.close()

    async def get_workflow_by_id(
        self,
        *,
        workflow_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> workflow_response_schema:
        db_session = db_session or super().get_db().session
        workflow = await self._get_workflow(workflow_id, organization_id, db_session)
        dataset_name, destination_name = await self._get_dataset_and_destination_names(
            dataset_id=workflow.dataset_id,
            destination_id=workflow.destination_id,
            db_session=db_session,
        )
        return workflow_response_schema(
            **workflow.dict(),
            schedule_config=self._format_schedule_config(workflow.schedule_configs),
            dataset_name=dataset_name,
            destination_name=destination_name,
        )

    def get_workflow_by_id_sync(
        self,
        *,
        workflow_id: UUID,
        organization_id: UUID,
        db_session: Session | None = None,
    ) -> WorkflowWithScheduleConfig:
        db_session = db_session or super().get_db().session
        workflow = self._get_workflow_sync(workflow_id, organization_id, db_session)
        logger.info(f"Retrieved workflow {workflow_id}")
        schedule_config = db_session.execute(
            select(ScheduleConfig).where(ScheduleConfig.workflow_id == workflow_id)
        ).scalar_one_or_none()
        logger.info(f"Retrieved schedule config for workflow {workflow_id}")
        return WorkflowWithScheduleConfig(workflow, schedule_config)

    async def remove_workflow(
        self,
        *,
        workflow_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> workflow_response_schema:
        db_session = db_session or super().get_db().session
        workflow = await self._get_workflow(workflow_id, organization_id, db_session)
        logger.debug(f"Removing workflow: {workflow}")

        # Stop the workflow if it is scheduled
        if workflow.scheduled:
            await self.stop_workflow(workflow_id=workflow_id)
            logging.debug(f"Stopped workflow: {workflow}")

        dataset_name, destination_name = await self._get_dataset_and_destination_names(
            dataset_id=workflow.dataset_id,
            destination_id=workflow.destination_id,
            db_session=db_session,
        )

        # Soft delete the workflow
        await self._soft_delete(workflow, db_session)
        logging.debug(f"deleted workflow: {workflow}")

        schedule_config = await db_session.execute(
            select(ScheduleConfig).where(ScheduleConfig.workflow_id == workflow_id)
        )
        schedule_config = schedule_config.scalar_one_or_none()

        return workflow_response_schema(
            **workflow.dict(),
            schedule_config=self._format_schedule_config(schedule_config),
            dataset_name=dataset_name,
            destination_name=destination_name,
        )

    async def execute_workflow(
        self,
        *,
        workflow_id: UUID,
        organization_id: UUID,
        user_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> str:

        # Trigger the workflow execution task
        execute_workflow_task.delay(organization_id, user_id, workflow_id)

        return WorkflowRunStatusType.Processing

    async def get_workflow_run_history(
        self,
        *,
        workflow_id: UUID,
        organization_id: UUID,
        params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[IWorkflowRunRead]:
        db_session = db_session or super().get_db().session
        columns = WorkflowRun.__table__.columns
        if order_by not in columns:
            order_by = "created_at"

        order_clause = asc(
            columns[order_by]
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = (
            select(WorkflowRun)
            .where(
                WorkflowRun.workflow_id == workflow_id,
            )
            .order_by(order_clause)
        )

        result = await db_session.execute(query)
        workflow_runs = result.scalars().all()

        return paginate(
            [
                IWorkflowRunRead(
                    run_id=workflow_run.id,
                    workflow_id=workflow_run.workflow_id,
                    status=workflow_run.status,
                    created_at=workflow_run.created_at.isoformat(),
                    finished_at=(
                        workflow_run.finished_at.isoformat()
                        if workflow_run.finished_at
                        else None
                    ),
                )
                for workflow_run in workflow_runs
            ],
            params,
        )

    def get_workflow_run_by_id_sync(
        self,
        workflow_run_id: UUID,
        workflow_id: UUID,
        db_session: Session | None = None,
    ) -> IWorkflowRunRead:
        db_session = db_session or super().get_db().session
        result = db_session.execute(
            select(WorkflowRun).where(
                WorkflowRun.id == workflow_run_id,
                WorkflowRun.workflow_id == workflow_id,
            )
        )
        workflow_run = result.scalar_one_or_none()
        return IWorkflowRunRead(
            run_id=workflow_run.id,
            workflow_id=workflow_run.workflow_id,
            status=workflow_run.status,
            created_at=workflow_run.created_at.isoformat(),
            finished_at=(
                workflow_run.finished_at.isoformat()
                if workflow_run.finished_at
                else None
            ),
        )

    def create_workflow_run_sync(
        self,
        workflow_id: UUID,
        db_session: Session | None = None,
    ) -> UUID:
        db_session = db_session or super().get_db().session
        workflow_run = WorkflowRun(workflow_id=workflow_id)
        db_session.add(workflow_run)
        db_session.commit()
        db_session.refresh(workflow_run)
        return workflow_run.id

    def update_workflow_run_attribute_sync(
        self,
        workflow_run_id: UUID,
        attribute_name: str,
        attribute_value,
        db_session: Session | None = None,
    ):
        db_session = db_session or super().get_db().session
        workflow_run = db_session.execute(
            select(WorkflowRun).where(WorkflowRun.id == workflow_run_id)
        ).scalar_one_or_none()
        if workflow_run:
            setattr(workflow_run, attribute_name, attribute_value)
            db_session.commit()
            db_session.refresh(workflow_run)

    async def get_files_by_workflow_id(
        self,
        *,
        workflow_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> List[File]:
        db_session = db_session or super().get_db().session

        # Retrieve the workflow using the workflow_id
        workflow = await self._get_workflow(
            workflow_id=workflow_id,
            organization_id=organization_id,
            db_session=db_session,
        )

        # Get the dataset_id from the workflow
        dataset_id = workflow.dataset_id

        # Retrieve the files associated with the dataset_id
        files = await crud.CRUDIngestion.get_files_for_dataset(
            self, dataset_id=dataset_id, db_session=db_session
        )

        return files


workflow = CRUDWorkflow(Workflow)
