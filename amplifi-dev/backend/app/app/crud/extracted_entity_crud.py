from typing import List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.crud.base_crud import CRUDBase
from app.models.extracted_entity_model import ExtractedEntity
from app.schemas.graph_schema import ExtractedEntity as ExtractedEntitySchema
from app.schemas.graph_schema import (
    ExtractedEntityRead,
)


class CRUDExtractedEntity(CRUDBase[ExtractedEntity, ExtractedEntity, ExtractedEntity]):
    async def create_entity(
        self,
        *,
        graph_id: UUID,
        item: ExtractedEntitySchema,
        db_session: AsyncSession | None = None,
    ) -> ExtractedEntity:
        """Create a single entity for a graph from ExtractedEntitySchema."""
        db_session = db_session or super().get_db().session
        entity = ExtractedEntity(
            graph_id=graph_id,
            name=item.name,
            entity_type=item.type,
            description=item.description,
        )
        db_session.add(entity)
        await db_session.commit()
        await db_session.refresh(entity)
        return entity

    async def create_batch(
        self,
        *,
        graph_id: UUID,
        items: List[ExtractedEntitySchema],
        db_session: AsyncSession | None = None,
    ) -> List[ExtractedEntity]:
        """
        Create many entities for a graph in one transaction from ExtractedEntitySchema list.
        """
        db_session = db_session or super().get_db().session

        to_persist: list[ExtractedEntity] = []
        for item in items:
            to_persist.append(
                ExtractedEntity(
                    graph_id=graph_id,
                    name=item.name,
                    entity_type=item.type,
                    description=item.description,
                )
            )

        if not to_persist:
            return []

        db_session.add_all(to_persist)
        await db_session.commit()

        # Refresh each to ensure ids are populated
        for entity in to_persist:
            await db_session.refresh(entity)

        return to_persist

    def create_batch_sync(
        self,
        *,
        graph_id: UUID,
        items: List[ExtractedEntitySchema],
        db_session: Session,
    ) -> List[ExtractedEntity]:
        """
        Sync version of create_batch - Create many entities for a graph in one transaction from ExtractedEntitySchema list.
        """
        to_persist: list[ExtractedEntity] = []
        for item in items:
            to_persist.append(
                ExtractedEntity(
                    graph_id=graph_id,
                    name=item.name,
                    entity_type=item.type,
                    description=item.description,
                )
            )

        if not to_persist:
            return []

        db_session.add_all(to_persist)
        db_session.commit()

        # Refresh each to ensure ids are populated
        for entity in to_persist:
            db_session.refresh(entity)

        return to_persist

    async def get_entity_type_counts(
        self,
        *,
        graph_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> list[dict]:
        """
        Returns a list of dicts: {"entity_type": str, "count": int}
        for the specified graph.
        """
        db_session = db_session or super().get_db().session
        query = (
            select(ExtractedEntity.entity_type, func.count(ExtractedEntity.id))
            .where(ExtractedEntity.graph_id == graph_id)
            .group_by(ExtractedEntity.entity_type)
            .order_by(func.count(ExtractedEntity.id).desc())
        )
        result = await db_session.execute(query)
        rows = result.all()
        return [
            {"entity_type": entity_type, "count": int(count)}
            for entity_type, count in rows
        ]

    async def delete_entity(
        self, *, entity_id: UUID, db_session: AsyncSession | None = None
    ) -> None:
        db_session = db_session or super().get_db().session
        await db_session.execute(
            delete(ExtractedEntity).where(ExtractedEntity.id == entity_id)
        )
        await db_session.commit()

    async def delete_by_type(
        self,
        *,
        graph_id: UUID,
        entity_type: str,
        db_session: AsyncSession | None = None,
    ) -> int:
        """
        Delete all entities of a given type for a specific graph.
        Returns the number of rows deleted.
        """
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            delete(ExtractedEntity).where(
                ExtractedEntity.graph_id == graph_id,
                ExtractedEntity.entity_type == entity_type,
            )
        )
        await db_session.commit()
        # result.rowcount may be None depending on dialect; coerce to int
        return int(result.rowcount or 0)

    async def get_entities_info(
        self,
        *,
        graph_id: UUID,
        entity_type: Optional[str] = None,
        db_session: AsyncSession | None = None,
    ) -> list[ExtractedEntitySchema]:
        """
        Return all entities for a graph with just the info: name, type, description.
        """
        db_session = db_session or super().get_db().session
        query = select(
            ExtractedEntity.id,
            ExtractedEntity.name,
            ExtractedEntity.entity_type,
            ExtractedEntity.description,
        ).where(ExtractedEntity.graph_id == graph_id)
        if entity_type:
            query = query.where(ExtractedEntity.entity_type == entity_type)
        result = await db_session.execute(query)
        rows = result.all()
        return [
            ExtractedEntityRead(
                id=row_id, name=name, type=entity_type, description=description
            )
            for row_id, name, entity_type, description in rows
        ]

    def get_entities_info_sync(
        self,
        *,
        graph_id: UUID,
        entity_type: Optional[str] = None,
        db_session: Session,
    ) -> list[ExtractedEntitySchema]:
        """
        Sync version of get_entities_info - Return all entities for a graph with just the info: name, type, description.
        """
        query = select(
            ExtractedEntity.id,
            ExtractedEntity.name,
            ExtractedEntity.entity_type,
            ExtractedEntity.description,
        ).where(ExtractedEntity.graph_id == graph_id)
        if entity_type:
            query = query.where(ExtractedEntity.entity_type == entity_type)
        result = db_session.execute(query)
        rows = result.all()
        return [
            ExtractedEntityRead(
                id=row_id, name=name, type=entity_type, description=description
            )
            for row_id, name, entity_type, description in rows
        ]


extracted_entity_crud = CRUDExtractedEntity(ExtractedEntity)
