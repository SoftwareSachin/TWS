from typing import Any, List
from uuid import UUID

from fastapi_pagination import Page, Params, paginate
from pydantic.networks import EmailStr
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.be_core.security import get_password_hash, verify_password
from app.crud.base_crud import CRUDBase
from app.crud.user_follow_crud import user_follow as UserFollowCRUD
from app.models import UserWorkspaceLink
from app.models.image_media_model import ImageMedia
from app.models.media_model import Media
from app.models.user_model import User
from app.schemas.media_schema import IMediaCreate
from app.schemas.user_schema import IUserCreate, IUserUpdate


class CRUDUser(CRUDBase[User, IUserCreate, IUserUpdate]):
    async def get_by_email(
        self, *, email: str, db_session: AsyncSession | None = None
    ) -> User | None:
        db_session = db_session or super().get_db().session
        users = await db_session.execute(select(User).where(User.email == email))
        return users.scalar_one_or_none()

    async def get_by_id_active(self, *, id: UUID) -> User | None:
        user = await super().get(id=id)
        if not user:
            return None
        if user.is_active is False:
            return None

        return user

    async def create_with_role(
        self, *, obj_in: IUserCreate, db_session: AsyncSession | None = None
    ) -> User:
        db_session = db_session or super().get_db().session
        db_obj = User.model_validate(obj_in)
        db_obj.hashed_password = get_password_hash(obj_in.password)
        db_session.add(db_obj)
        await db_session.commit()
        await db_session.refresh(db_obj)
        return db_obj

    async def update_is_active(
        self, *, db_obj: list[User], obj_in: int | str | dict[str, Any]
    ) -> User | None:
        response = None
        db_session = super().get_db().session
        for x in db_obj:
            x.is_active = obj_in.is_active
            db_session.add(x)
            await db_session.commit()
            await db_session.refresh(x)
            response.append(x)
        return response

    async def authenticate(self, *, email: EmailStr, password: str) -> User | None:
        user = await self.get_by_email(email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def update_photo(
        self,
        *,
        user: User,
        image: IMediaCreate,
        heigth: int,
        width: int,
        file_format: str,
    ) -> User:
        db_session = super().get_db().session
        user.image = ImageMedia(
            media=Media.model_validate(image),
            height=heigth,
            width=width,
            file_format=file_format,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        return user

    async def remove(
        self, *, id: UUID | str, db_session: AsyncSession | None = None
    ) -> User:
        db_session = db_session or super().get_db().session
        response = await db_session.execute(
            select(self.model).where(self.model.id == id)
        )
        obj = response.scalar_one()

        followings = await UserFollowCRUD.get_follow_by_user_id(user_id=obj.id)
        if followings:
            for following in followings:
                user = await self.get(id=following.target_user_id)
                user.follower_count -= 1
                db_session.add(user)
                await db_session.delete(following)

        followeds = await UserFollowCRUD.get_follow_by_target_user_id(
            target_user_id=obj.id
        )
        if followeds:
            for followed in followeds:
                user = await self.get(id=followed.user_id)
                user.following_count -= 1
                db_session.add(user)
                await db_session.delete(followed)

        await db_session.delete(obj)
        await db_session.commit()
        return obj

    async def get_users(
        self,
        *,
        organization_id: UUID,
        params: Params = Params(),
        db_session: AsyncSession | None = None,
    ) -> Page[User]:
        db_session = db_session or super().get_db().session
        query = select(User).where(User.organization_id == organization_id)
        results = await db_session.execute(query)
        files = results.scalars().all()
        return paginate(files, params)

    async def get_workspace_ids_for_user(
        self,
        *,
        user_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> List[UUID]:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(UserWorkspaceLink.workspace_id).where(
                UserWorkspaceLink.user_id == user_id
            )
        )
        workspace_ids = result.scalars().all()

        if not workspace_ids:
            workspace_ids = []

        return workspace_ids


user = CRUDUser(User)
