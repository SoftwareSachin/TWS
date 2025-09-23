from datetime import datetime
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Page, Params, paginate
from sqlalchemy import and_, asc, desc
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.crud.base_crud import CRUDBase
from app.models.chat_app_model import ChatApp
from app.models.chat_session_model import ChatSession
from app.schemas.chat_schema import (
    IChatSessionCreate,
)
from app.schemas.common_schema import IOrderEnum


class CRUDChatSession(CRUDBase[ChatSession, IChatSessionCreate, None]):
    async def get_chat_session_by_id(
        self,
        *,
        chat_session_id: UUID,
        chatapp_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> ChatSession:
        db_session = db_session or super().get_db().session
        chatsession_record = await db_session.execute(
            select(ChatSession).where(
                ChatSession.id == chat_session_id,
                ChatSession.chatapp_id == chatapp_id,
                ChatSession.deleted_at.is_(None),
            )
        )
        return chatsession_record.scalars().first()

    async def create_chat_session(
        self,
        *,
        obj_in: IChatSessionCreate,
        chatapp_id: UUID,
        user_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> ChatSession:
        db_session = db_session or super().get_db().session
        query = select(ChatApp.id).filter(ChatApp.id == chatapp_id)
        result = await db_session.execute(query)
        chat_app = result.scalar_one_or_none()
        if not chat_app:
            raise ValueError(f"ChatApp with id {chatapp_id} does not exist.")
        chat_session = ChatSession(
            title=obj_in.title,
            chatapp_id=chatapp_id,
            user_id=user_id,
        )
        db_session.add(chat_session)
        await db_session.commit()
        await db_session.refresh(chat_session)
        return chat_session

    async def update_chat_session(
        self,
        *,
        chat_session_id: UUID,
        chatapp_id: UUID,
        obj_in: IChatSessionCreate,
        db_session: AsyncSession | None = None,
    ) -> ChatSession:
        db_session = db_session or super().get_db().session
        chat_session: ChatSession = await self.get_chat_session_by_id(
            chat_session_id=chat_session_id,
            chatapp_id=chatapp_id,
            db_session=db_session,
        )

        if not chat_session:
            raise HTTPException(
                status_code=404,
                detail=f"ChatSession with ID {chat_session_id} not found.",
            )

        for field, value in obj_in.model_dump(exclude_unset=True).items():
            setattr(chat_session, field, value)

        chat_session.updated_at = datetime.utcnow()

        await db_session.commit()
        await db_session.refresh(chat_session)
        return chat_session

    async def get_chat_sessions(
        self,
        *,
        chatapp_id: UUID,
        user_id: UUID,
        pagination_params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[ChatSession]:
        db_session = db_session or super().get_db().session
        columns = ChatSession.__table__.columns
        if order_by not in columns:
            order_by = "created_at"
        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )
        query = (
            select(ChatSession)
            .where(
                and_(
                    ChatSession.chatapp_id == chatapp_id,
                    ChatSession.deleted_at.is_(None),
                    ChatSession.user_id == user_id,
                )
            )
            .order_by(order_clause)
        )
        result = await db_session.execute(query)
        return paginate(result.scalars().all(), pagination_params)

    async def get_chatapp_id_of_chatsession(
        self,
        *,
        chat_session_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> UUID | None:
        db_session = db_session or super().get_db().session

        chatsession_record = await db_session.execute(
            select(ChatSession).where(
                ChatSession.id == chat_session_id,
                ChatSession.deleted_at.is_(None),
            )
        )
        # return chatsession_record.scalars().first()

        chatsession: ChatSession | None = chatsession_record.scalars().first()

        return chatsession.chatapp_id if chatsession else None

    async def soft_delete_chatapp_by_id(
        self,
        *,
        chat_session_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> None:
        db_session = db_session or super().get_db().session
        chatsession_record = await db_session.execute(
            select(ChatSession).where(
                ChatSession.id == chat_session_id,
                ChatSession.deleted_at.is_(None),
            )
        )
        chatsession: ChatSession | None = chatsession_record.scalars().first()
        if not chatsession:
            raise HTTPException(
                status_code=404,
                detail=f"ChatSession with ID {chat_session_id} not found.",
            )
        chatsession.deleted_at = datetime.utcnow()
        await db_session.commit()
        await db_session.refresh(chatsession)


chatsession = CRUDChatSession(ChatSession)
