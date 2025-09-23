from datetime import datetime
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func
from sqlmodel import and_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.crud.base_crud import CRUDBase
from app.crud.workspace_crud import workspace
from app.db.session import SyncSessionLocal
from app.models.organization_model import Organization
from app.schemas.organization_schema import (
    IOrganizationCreate,
    IOrganizationUpdate,
)


class CRUDOrganization(
    CRUDBase[Organization, IOrganizationCreate, IOrganizationUpdate]
):
    async def get_organization_by_id(
        self,
        *,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Organization:
        db_session = db_session or super().get_db().session
        query = select(Organization).where(
            and_(Organization.id == organization_id, Organization.deleted_at.is_(None))
        )
        response = await db_session.execute(query)
        organization = response.scalar_one_or_none()
        return organization

    @staticmethod
    def get_organization_by_id_sync(
        organization_id: UUID,
    ) -> Organization:
        with SyncSessionLocal() as db_session:
            org = db_session.execute(
                select(Organization).where(Organization.id == organization_id)
            ).scalar_one_or_none()
            return org

    async def create_organization(
        self,
        *,
        obj_in: IOrganizationCreate,
        db_session: AsyncSession | None = None,
    ) -> Organization | None:
        db_session = db_session or super().get_db().session

        org = Organization(
            name=obj_in.name,
            description=obj_in.description,
            domain=obj_in.domain,
            r2r_user="r2r_disabled@amplifi.com",
        )

        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        return org

    async def get_organizations(
        self,
        *,
        db_session: AsyncSession | None = None,
    ) -> list[Organization]:
        db_session = (db_session or super().get_db().session,)

        result = await db_session.execute(select(Organization))
        return result.scalars().all()

    async def count_organizations(db: AsyncSession) -> int:
        return await db.execute(select(func.count(Organization.id))).scalar()

    async def get_orgid_from_name(
        self, *, org_id: str, db_session: AsyncSession | None = None
    ):
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Organization.id).where(Organization.name == org_id)
        )
        orgid = result.scalar()
        return orgid

    async def update_organization(
        self,
        *,
        organization_id: UUID,
        obj_in: IOrganizationUpdate,
        db_session: AsyncSession | None = None,
    ) -> Organization:
        db_session = db_session or super().get_db().session
        organization = await self.get_organization_by_id(
            organization_id=organization_id,
            db_session=db_session,
        )

        if not organization:
            raise ValueError(
                f"Embedding configuration with ID {organization_id} not found."
            )

        if obj_in.name is not None:
            organization.name = obj_in.name
        if obj_in.description is not None:
            organization.description = obj_in.description

        # Update the fields
        for field, value in obj_in.dict(exclude_unset=True).items():
            setattr(organization, field, value)

        await db_session.commit()
        await db_session.refresh(organization)

        return organization

    async def soft_delete_organization(
        self,
        *,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Organization:
        db_session = db_session or super().get_db().session
        organization = await self.get_organization_by_id(
            organization_id=organization_id,
            db_session=db_session,
        )

        if not organization:
            raise HTTPException(
                status_code=404,
                detail=f"Organization with ID {organization_id} not found.",
            )

        organization.deleted_at = datetime.utcnow()
        await db_session.commit()
        await db_session.refresh(organization)
        workspace_ids = await workspace.get_workspace_ids_of_organization(
            organization_id=organization_id
        )
        for workspace_id in workspace_ids:
            workspace.delete_workspace(
                workspace_id=workspace_id, organization_id=organization_id
            )
        return organization


organization = CRUDOrganization(Organization)
