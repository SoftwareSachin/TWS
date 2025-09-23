from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.crud.base_crud import CRUDBase
from app.models.platform_model import DeploymentInfo
from app.schemas.platform_schema import IDeploymentInfoCreate, IDeploymentInfoRead


class CRUDDeploymentInfo(
    CRUDBase[DeploymentInfo, IDeploymentInfoRead, IDeploymentInfoCreate]
):
    async def get_latest(
        self, db_session: AsyncSession | None = None
    ) -> DeploymentInfo | None:
        db_session = db_session or super().get_db().session
        response = await db_session.execute(select(DeploymentInfo).limit(1))
        return response.scalar_one_or_none()

    async def create(self, obj_in: IDeploymentInfoCreate, db_session: AsyncSession):
        """
        Create a new deployment record.
        """
        db_obj = DeploymentInfo(
            version=obj_in.version,
            product_documentation_link=str(obj_in.product_documentation_link),
            technical_documentation_link=str(obj_in.technical_documentation_link),
        )
        db_session.add(db_obj)
        await db_session.commit()
        await db_session.refresh(db_obj)
        return db_obj

    async def get_by_version(self, version: str, db_session: AsyncSession):
        """
        Retrieve a deployment by its version.
        """
        query = select(DeploymentInfo).where(DeploymentInfo.version == version)
        result = await db_session.execute(query)
        return result.scalars().first()


deployment_info = CRUDDeploymentInfo(DeploymentInfo)
