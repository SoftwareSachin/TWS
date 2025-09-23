from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import and_, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models import VannaTraining


class CRUDVannaTraining:
    async def create(self, db: AsyncSession, obj_in: Dict[str, Any]) -> VannaTraining:
        db_obj = VannaTraining(**obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def get_max_version_by_dataset(
        self, db: AsyncSession, dataset_id: UUID
    ) -> List[VannaTraining]:
        # Get the maximum version for the dataset
        max_version_subquery = (
            select(func.max(VannaTraining.version_id)).where(
                and_(
                    VannaTraining.dataset_id == dataset_id,
                    VannaTraining.deleted_at.is_(None),
                )
            )
        ).scalar_subquery()

        # Get all records with the maximum version
        result = await db.execute(
            select(VannaTraining).where(
                and_(
                    VannaTraining.dataset_id == dataset_id,
                    VannaTraining.version_id == max_version_subquery,
                    VannaTraining.deleted_at.is_(None),
                )
            )
        )
        return result.scalars().all()

    async def get_by_dataset(
        self, db: AsyncSession, dataset_id: UUID
    ) -> List[VannaTraining]:
        result = await db.execute(
            select(VannaTraining).where(
                and_(
                    VannaTraining.dataset_id == dataset_id,
                    VannaTraining.deleted_at.is_(None),
                )
            )
        )
        return result.scalars().all()

    async def soft_delete_by_dataset(self, db: AsyncSession, dataset_id: UUID) -> int:
        result = await db.execute(
            update(VannaTraining)
            .where(
                and_(
                    VannaTraining.dataset_id == dataset_id,
                    VannaTraining.deleted_at.is_(None),
                )
            )
            .values(deleted_at=datetime.utcnow())
        )
        await db.commit()
        return result.rowcount

    async def get_next_version_for_dataset(
        self, db: AsyncSession, dataset_id: UUID
    ) -> int:
        result = await db.execute(
            select(func.max(VannaTraining.version_id)).where(
                VannaTraining.dataset_id == dataset_id
            )
        )
        max_version = result.scalar()
        return (max_version + 1) if max_version else 1

    async def delete_by_dataset(self, db: AsyncSession, dataset_id: UUID) -> int:
        """Hard delete - kept for backward compatibility"""
        result = await db.execute(
            VannaTraining.__table__.delete().where(
                VannaTraining.dataset_id == dataset_id
            )
        )
        await db.commit()
        return result.rowcount

    async def get_all_by_dataset_including_deleted(
        self, db: AsyncSession, dataset_id: UUID
    ) -> List[VannaTraining]:
        """Get all training records for a dataset, including soft-deleted ones"""
        result = await db.execute(
            select(VannaTraining).where(VannaTraining.dataset_id == dataset_id)
        )
        return result.scalars().all()


vanna_training = CRUDVannaTraining()
