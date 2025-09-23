from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.crud.base_crud import CRUDBase
from app.models.graph_model import Graph
from app.schemas.graph_schema import GraphCreate, GraphRead


class CRUDGraph(CRUDBase[Graph, GraphCreate, GraphCreate]):
    async def create_graph(
        self,
        *,
        obj_in: GraphCreate,
        db_session: Optional[AsyncSession] = None,
    ) -> GraphRead:
        db_session = db_session or super().get_db().session
        db_obj = Graph(**obj_in.dict())
        db_session.add(db_obj)
        await db_session.commit()
        await db_session.refresh(db_obj)
        return GraphRead.model_validate(db_obj)

    async def get_graph(
        self,
        *,
        graph_id: UUID,
        db_session: Optional[AsyncSession] = None,
    ) -> Optional[GraphRead]:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Graph).where(Graph.id == graph_id, Graph.deleted_at.is_(None))
        )
        db_obj = result.scalar_one_or_none()
        if not db_obj:
            raise HTTPException(status_code=404, detail="Graph not found")
        return GraphRead.model_validate(db_obj)

    async def get_graphs_by_dataset(
        self,
        *,
        dataset_id: UUID,
        db_session: Optional[AsyncSession] = None,
    ) -> List[GraphRead]:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Graph).where(
                Graph.dataset_id == dataset_id, Graph.deleted_at.is_(None)
            )
        )
        db_objs = result.scalars().all()
        return [GraphRead.model_validate(obj) for obj in db_objs]

    async def update_graph_status(
        self,
        *,
        graph_id: UUID,
        new_status: str,
        field: Literal["relationships", "entities"],
        db_session: Optional[AsyncSession] = None,
    ) -> GraphRead:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Graph).where(Graph.id == graph_id, Graph.deleted_at.is_(None))
        )
        db_obj = result.scalar_one_or_none()
        if not db_obj:
            raise HTTPException(status_code=404, detail="Graph not found")
        if field == "relationships":
            db_obj.relationships_status = new_status
        elif field == "entities":
            db_obj.entities_status = new_status
        else:
            raise HTTPException(status_code=400, detail="Invalid field")
        await db_session.commit()
        await db_session.refresh(db_obj)
        return GraphRead.model_validate(db_obj)

    def update_graph_status_sync(
        self,
        *,
        graph_id: UUID,
        new_status: str,
        field: Literal["relationships", "entities"],
        db_session: Session,
    ) -> GraphRead:
        """Sync version of update_graph_status"""
        result = db_session.execute(
            select(Graph).where(Graph.id == graph_id, Graph.deleted_at.is_(None))
        )
        db_obj = result.scalar_one_or_none()
        if not db_obj:
            raise HTTPException(status_code=404, detail="Graph not found")
        if field == "relationships":
            db_obj.relationships_status = new_status
        elif field == "entities":
            db_obj.entities_status = new_status
        else:
            raise HTTPException(status_code=400, detail="Invalid field")
        db_session.commit()
        db_session.refresh(db_obj)
        return GraphRead.model_validate(db_obj)

    async def soft_delete_graph(
        self,
        *,
        graph_id: UUID,
        db_session: Optional[AsyncSession] = None,
    ) -> GraphRead:
        """Soft delete a graph by setting its deleted_at timestamp and cleaning up Kuzu database files."""
        db_session = db_session or super().get_db().session

        result = await db_session.execute(
            select(Graph).where(Graph.id == graph_id, Graph.deleted_at.is_(None))
        )
        db_obj = result.scalar_one_or_none()

        if not db_obj:
            raise HTTPException(
                status_code=404, detail="Graph not found or already deleted"
            )

        # Soft delete the graph in the database
        db_obj.deleted_at = datetime.utcnow()
        db_session.add(db_obj)
        await db_session.commit()
        await db_session.refresh(db_obj)
        # Skip deletion of kuzu files, to allow for re-use of graph if needed.
        # # Clean up the Kuzu database files
        # try:
        #     kuzu_manager = KuzuManager()
        #     kuzu_manager.delete_graph(str(graph_id))
        # except Exception as e:
        #     # Log the error but don't fail the soft delete operation
        #     logger.warning(
        #         f"Failed to delete Kuzu database files for graph {graph_id}: {str(e)}"
        #     )

        return GraphRead.model_validate(db_obj)

    async def get_most_recent_graph(
        self,
        *,
        dataset_id: UUID,
        db_session: Optional[AsyncSession] = None,
    ) -> GraphRead:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Graph)
            .where(Graph.dataset_id == dataset_id, Graph.deleted_at.is_(None))
            .order_by(Graph.created_at.desc())
            .limit(1)
        )
        db_obj = result.scalar_one_or_none()
        if not db_obj:
            raise HTTPException(status_code=404, detail="Graph not found")
        return GraphRead.model_validate(db_obj)


crud_graph = CRUDGraph(Graph)
