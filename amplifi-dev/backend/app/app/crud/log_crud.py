from typing import List
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Params
from sqlalchemy import or_
from sqlalchemy.orm import selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.models import Workspace
from app.models.audit_log_model import AuditLog, EntityType, OperationType
from app.schemas.common_schema import IOrderEnum


class CRUDLog(CRUDBase[AuditLog, None, None]):
    async def get_workspace_logs(
        self, workspace_id: UUID, params: Params, db_session: AsyncSession | None = None
    ) -> List[str]:
        db_session = db_session or super().get_db().session
        workspace = await db_session.execute(
            select(Workspace)
            .options(
                selectinload(Workspace.sources),
                selectinload(Workspace.files),
                selectinload(Workspace.datasets),
            )
            .where(Workspace.id == workspace_id)
        )
        workspace = workspace.scalar_one_or_none()

        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        source_ids = [source.id for source in workspace.sources]
        logger.info(f"source_ids:{source_ids}")
        file_ids = [file.id for file in workspace.files]
        logger.info(f"file_ids:{file_ids}")
        dataset_ids = [dataset.id for dataset in workspace.datasets]
        logger.info(f"dataset_ids:{dataset_ids}")
        related_ids = source_ids + file_ids + dataset_ids
        query = (
            select(AuditLog)
            .where(
                or_(
                    AuditLog.entity_id == workspace_id,
                    AuditLog.entity_id.in_(related_ids),
                )
            )
            .order_by(AuditLog.logged_at.desc())
        )
        paginated_logs = await self.get_multi_paginated_ordered(
            params=params,
            order_by="logged_at",
            order=IOrderEnum.descendent,
            query=query,
        )
        # Format the logs
        formatted_logs = [
            f"{log.user_name} ({log.user_id}) {log.operation.value}d {log.entity.value} {log.entity_name} ({log.entity_id}) at {log.logged_at}"
            for log in paginated_logs.items
        ]
        return formatted_logs

    async def log_operation(
        self,
        *,
        operation: OperationType,
        entity: EntityType,
        entity_id: UUID,
        entity_name: str,
        user_id: UUID,
        user_name: str,
        db_session: AsyncSession | None = None,
    ) -> None:
        db_session = db_session or super().get_db().session

        new_log = AuditLog(
            operation=operation,
            entity=entity,
            entity_id=entity_id,
            entity_name=entity_name,
            user_id=user_id,
            user_name=user_name,
        )
        db_session.add(new_log)
        await db_session.commit()


log_crud = CRUDLog(AuditLog)
