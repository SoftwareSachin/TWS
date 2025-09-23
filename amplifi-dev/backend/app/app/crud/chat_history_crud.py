from typing import List
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Page, Params, paginate
from sqlmodel import insert, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.models.chat_app_model import ChatApp
from app.models.chat_history_model import ChatHistory
from app.models.chat_session_model import ChatSession
from app.schemas.chat_schema import IChatHistoryCreate


class CRUDChatHistory(CRUDBase[ChatHistory, IChatHistoryCreate, None]):
    async def get_chat_history_by_session_id(
        self,
        *,
        chat_session_id: UUID,
        pagination_params: Params = Params(),
        db_session: AsyncSession | None = None,
    ) -> Page[ChatHistory]:
        db_session = db_session or super().get_db().session
        query = (
            select(ChatHistory)
            .where(ChatHistory.chat_session_id == chat_session_id)
            .order_by(ChatHistory.created_at.asc())
        )
        result = await db_session.execute(query)
        return paginate(result.scalars().all(), pagination_params)

    async def get_chat_history_by_session_id_no_pagination(
        self,
        *,
        chat_session_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> List[ChatHistory]:
        db_session = db_session or super().get_db().session
        query = (
            select(ChatHistory)
            .where(ChatHistory.chat_session_id == chat_session_id)
            .order_by(ChatHistory.created_at.asc())
        )
        result = await db_session.execute(query)
        return result.scalars().all()

    async def add_chat_session_history(
        self,
        *,
        chatapp_id: UUID,
        chat_session_id: UUID,
        obj_in: IChatHistoryCreate,
        db_session: AsyncSession | None = None,
    ):
        db_session = db_session or super().get_db().session

        ## checking that chat_session id belongs in the chatapp id. raise an exception if it odnt
        query = select(ChatSession).where(
            ChatSession.id == chat_session_id, ChatSession.chatapp_id == chatapp_id
        )

        result = await db_session.execute(query)
        chat_session = result.scalars().first()

        if not chat_session:
            logger.debug("chat session id doesnt belong in chatapp")
            raise HTTPException(
                status_code=400,
                detail=f"Chat session {chat_session_id} does not belong to chat app {chatapp_id}",
            )

        history_data = [
            {
                "chat_session_id": chat_session_id,
                "user_query": history.user_query,
                "contexts": history.contexts,
                "llm_response": history.llm_response,
                "llm_model": history.llm_model,
                "input_tokens": history.input_tokens,
                "output_tokens": history.output_tokens,
                "total_tokens": history.total_tokens,
                "pydantic_message": history.pydantic_message,
            }
            for history in obj_in.histories
        ]

        await db_session.execute(insert(ChatHistory).values(history_data))

        new_input_tokens = sum(
            history["input_tokens"]
            for history in history_data
            if history["input_tokens"] is not None
        )
        if new_input_tokens is None:
            new_input_tokens = 0
        new_output_tokens = sum(
            history["output_tokens"]
            for history in history_data
            if history["output_tokens"] is not None
        )
        if new_output_tokens is None:
            new_output_tokens = 0
        new_total_tokens = sum(
            history["total_tokens"]
            for history in history_data
            if history["total_tokens"] is not None
        )
        if new_total_tokens is None:
            new_total_tokens = 0

        await db_session.execute(
            update(ChatSession)
            .where(ChatSession.id == chat_session_id)
            .values(
                input_tokens_per_chat_session=ChatSession.input_tokens_per_chat_session
                + new_input_tokens,
                output_tokens_per_chat_session=ChatSession.output_tokens_per_chat_session
                + new_output_tokens,
                total_tokens_per_chat_session=ChatSession.total_tokens_per_chat_session
                + new_total_tokens,
            )
        )

        await db_session.execute(
            update(ChatApp)
            .where(ChatApp.id == chatapp_id)
            .values(
                input_tokens_per_chat_app=ChatApp.input_tokens_per_chat_app
                + new_input_tokens,
                output_tokens_per_chat_app=ChatApp.output_tokens_per_chat_app
                + new_output_tokens,
                total_tokens_per_chat_app=ChatApp.total_tokens_per_chat_app
                + new_total_tokens,
            )
        )

        await db_session.commit()


chathistory = CRUDChatHistory(ChatHistory)
