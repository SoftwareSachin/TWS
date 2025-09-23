from datetime import datetime
from typing import List, Optional, Sequence
from uuid import UUID

from fastapi import HTTPException, status
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import and_, func, or_
from sqlalchemy.exc import IntegrityError
from sqlmodel import asc, delete, desc, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.crud.chat_app_crud import chatapp
from app.crud.user_crud import user
from app.models.user_model import User
from app.models.user_workspace_link_model import UserWorkspaceLink
from app.models.workspace_model import Workspace
from app.schemas.common_schema import IOrderEnum
from app.schemas.role_schema import IRoleEnum
from app.schemas.workspace_schema import IWorkspaceCreate, IWorkspaceUpdate


class CRUDWorkspace(CRUDBase[Workspace, IWorkspaceCreate, IWorkspaceUpdate]):
    async def create_workspace(
        self,
        *,
        obj_in: IWorkspaceCreate,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Workspace:
        db_session = db_session or super().get_db().session
        workspace = Workspace(
            name=obj_in.name,
            description=obj_in.description,
            is_active=obj_in.is_active,
            organization_id=organization_id,
        )
        # Check if a workspace with the same name already exists in this org
        result = await db_session.execute(
            select(Workspace).where(
                Workspace.organization_id == organization_id,
                Workspace.name.ilike(obj_in.name),
                Workspace.deleted_at.is_(None),  # Only consider non-deleted workspaces
            )
        )
        existing_workspace = result.scalars().first()

        if existing_workspace:
            raise HTTPException(
                status_code=400,
                detail=f"A workspace with name '{obj_in.name}' already exists in this organization.",
            )

        db_session.add(workspace)
        await db_session.commit()
        await db_session.refresh(workspace)

        return workspace

    async def update_workspace(
        self,
        *,
        workspace_id: UUID,
        organization_id: UUID,
        obj_in: IWorkspaceUpdate,
        db_session: AsyncSession | None = None,
    ) -> Workspace:
        db_session = db_session or super().get_db().session

        # Get existing workspace
        workspace = await db_session.get(Workspace, workspace_id)
        if not workspace or workspace.organization_id != organization_id:
            return None

        # Check duplicate name
        if obj_in.name:
            result = await db_session.execute(
                select(Workspace).where(
                    Workspace.organization_id == organization_id,
                    Workspace.name.ilike(obj_in.name),
                    Workspace.id != workspace_id,
                    Workspace.deleted_at.is_(None),
                )
            )
            existing_workspace = result.scalars().first()

            if existing_workspace:
                raise HTTPException(
                    status_code=400,
                    detail=f"A workspace with name '{obj_in.name}' already exists in this organization.",
                )

        # Update fields
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(workspace, field, value)

        db_session.add(workspace)
        await db_session.commit()
        await db_session.refresh(workspace)
        return workspace

    async def get_workspace(
        self,
        *,
        workspace_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Workspace:
        db_session = db_session or super().get_db().session
        query = select(Workspace).where(
            and_(
                Workspace.id == workspace_id,
                Workspace.organization_id == organization_id,
                Workspace.deleted_at.is_(None),
            )
        )

        result = await db_session.execute(query)
        workspace = result.unique().scalar_one_or_none()

        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found or has been soft-deleted.",
            )

        return workspace

    async def get_organization_id_of_workspace(
        self,
        *,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> UUID:
        db_session = db_session or super().get_db().session

        query = select(Workspace).where(
            and_(
                Workspace.id == workspace_id,
                Workspace.deleted_at.is_(None),
            )
        )

        result = await db_session.execute(query)
        workspace = result.unique().scalar_one_or_none()

        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found",
            )

        return workspace.organization_id

    async def get_workspaces(
        self,
        *,
        organization_id: UUID,
        params: Params | None = Params(),
        order: IOrderEnum = IOrderEnum.ascendent,
        user_id: UUID,
        user_role: List[str],
        order_by: str = "created_at",
        search: Optional[str] = None,
        db_session: AsyncSession | None = None,
    ) -> Page[Workspace]:
        logger.info(f"CRUD: Starting get_workspaces for org {organization_id}")
        db_session = db_session or super().get_db().session
        columns = Workspace.__table__.columns

        # Validate `order_by` column
        if order_by not in columns:
            logger.warning(
                f"Invalid order_by column '{order_by}', defaulting to 'created_at'"
            )
            order_by = "created_at"

        # # Determine order clause
        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )
        logger.info(
            f"CRUD: Building query filters - search: '{search}', order: {order}"
        )
        # Get all workspaces in organization if user is Admin
        filters = [
            Workspace.organization_id == organization_id,
            Workspace.deleted_at.is_(None),
        ]

        if search:
            search_pattern = f"%{search.lower()}%"
            filters.append(
                or_(
                    func.lower(Workspace.name).like(search_pattern),
                )
            )
            logger.info(f"CRUD: Added search filter for '{search}'")
        try:
            if IRoleEnum.admin in user_role:
                logger.info("CRUD: User is admin - fetching all workspaces")
                # Query with filter for soft deletion and order
                query = select(Workspace).where(*filters).order_by(order_clause)
            else:
                logger.info("CRUD: User is not admin - fetching user workspaces")
                workspace_ids = await user.get_workspace_ids_for_user(user_id=user_id)
                logger.info(f"CRUD: User has access to {len(workspace_ids)} workspaces")
                filters.append(Workspace.id.in_(workspace_ids))
                query = select(Workspace).where(*filters).order_by(order_clause)

            logger.info("CRUD: Executing database query")
            result = await paginate(db_session, query, params)
            logger.info(
                f"CRUD: Query completed - found {len(result.items)} workspaces, total: {result.total}"
            )

            return result

        except Exception as e:
            logger.error(f"CRUD: Error in get_workspaces: {str(e)}", exc_info=True)
            raise

    async def get_workspace_ids_of_organization(
        self,
        *,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> List[UUID]:
        db_session = db_session or super().get_db().session

        query = select(Workspace.id).where(
            and_(
                Workspace.organization_id == organization_id,
                Workspace.deleted_at.is_(None),
            )
        )

        result = await db_session.execute(query)
        workspace_ids = result.scalars().all()

        return workspace_ids

    async def _soft_delete(self, entity: Workspace, db_session: AsyncSession):
        entity.deleted_at = datetime.now()
        entity.is_active = False
        await db_session.commit()
        await db_session.refresh(entity)

    async def delete_workspace(
        self,
        *,
        workspace_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Workspace:
        db_session = db_session or super().get_db().session
        workspace = await db_session.get(Workspace, workspace_id)
        if not workspace:
            raise ValueError(f"Workspace with ID {workspace_id} not found.")
        await chatapp.delete_chatapps_by_workspace_id(
            workspace_id=workspace_id, db_session=db_session
        )

        # Clean up all video segments for the workspace when workspace is deleted
        await self._cleanup_all_workspace_video_segments(workspace_id, db_session)

        await self._soft_delete(workspace, db_session)
        return workspace

    async def _cleanup_all_workspace_video_segments(
        self, workspace_id: UUID, db_session: AsyncSession
    ) -> None:
        """
        Clean up ALL video segments for a workspace when the workspace is deleted.

        This forcefully removes all video segment directories for the workspace
        since the workspace itself is being deleted.
        """
        try:
            import os

            from app.be_core.config import settings

            logger.info(
                f"Cleaning up all video segments for deleted workspace {workspace_id}"
            )

            # Get the workspace video segments directory
            workspace_video_dir = os.path.join(
                settings.VIDEO_SEGMENTS_DIR, str(workspace_id)
            )

            if os.path.exists(workspace_video_dir):
                import shutil

                # Calculate total size before deletion for logging
                total_size = 0
                total_files = 0
                for root, _dirs, files in os.walk(workspace_video_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                            total_files += 1
                        except (OSError, IOError):
                            pass

                # Remove the entire workspace video directory
                shutil.rmtree(workspace_video_dir)

                size_mb = total_size / (1024 * 1024)
                logger.info(
                    f"Deleted all video segments for workspace {workspace_id}: "
                    f"{size_mb:.1f}MB, {total_files} files"
                )
            else:
                logger.debug(
                    f"No video segments directory found for workspace {workspace_id}"
                )

        except Exception as e:
            # Don't fail workspace deletion if video cleanup fails
            logger.error(
                f"Error cleaning up video segments for workspace {workspace_id}: {e}",
                exc_info=True,
            )

    async def user_belongs_to_organization(
        self,
        *,
        user_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> bool:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(User).where(
                User.id == user_id, User.organization_id == organization_id
            )
        )
        user = result.scalar_one_or_none()
        return user is not None

    async def user_belongs_to_workspace(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> bool:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(UserWorkspaceLink).where(
                UserWorkspaceLink.user_id == user_id,
                UserWorkspaceLink.workspace_id == workspace_id,
            )
        )
        link = result.scalar_one_or_none()
        return link is not None

    async def add_user_in_workspace(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ):
        db_session = db_session or super().get_db().session
        link = UserWorkspaceLink(user_id=user_id, workspace_id=workspace_id)
        db_session.add(link)
        try:
            await db_session.commit()
        except IntegrityError:
            await db_session.rollback()
            raise HTTPException(
                status_code=400, detail="User is already in the workspace"
            )

    async def get_users_in_workspace(
        self,
        *,
        workspace_id: UUID,
        params: Params | None = Params(),
        order: IOrderEnum = IOrderEnum.ascendent,
        order_by: str = "created_at",
        db_session: AsyncSession | None = None,
    ) -> Page[User]:
        db_session = db_session or super().get_db().session
        columns = User.__table__.columns
        if order_by not in columns:
            logger.warning("given order_by column invalid. defaulting to created_at")
            order_by = "created_at"
        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )
        query = (
            select(User)
            .join(UserWorkspaceLink, User.id == UserWorkspaceLink.user_id)
            .where(
                and_(
                    UserWorkspaceLink.workspace_id == workspace_id,
                    User.deleted_at.is_(None),
                )
            )
            .order_by(order_clause)
        )

        return await paginate(db_session, query, params)

    async def delete_users_from_workspaces(
        self,
        *,
        workspace_id: UUID,
        user_ids: List[UUID],
        db_session: AsyncSession | None = None,
    ) -> Sequence[UUID]:
        db_session = db_session or super().get_db().session

        result = await db_session.execute(
            select(UserWorkspaceLink.user_id).where(
                UserWorkspaceLink.workspace_id == workspace_id,
                UserWorkspaceLink.user_id.in_(user_ids),
            )
        )
        existing_user_ids = result.scalars().all()

        if not existing_user_ids:
            raise HTTPException(
                status_code=404, detail="No matching users found in workspace"
            )

        logger.info(f"Deleting {len(existing_user_ids)} users from workspace")
        await db_session.execute(
            delete(UserWorkspaceLink).where(
                UserWorkspaceLink.workspace_id == workspace_id,
                UserWorkspaceLink.user_id.in_(existing_user_ids),
            )
        )
        await db_session.commit()
        return existing_user_ids


workspace = CRUDWorkspace(Workspace)
