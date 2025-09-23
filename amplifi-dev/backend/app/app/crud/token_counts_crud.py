from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.crud.base_crud import CRUDBase
from app.models.chat_app_model import ChatApp
from app.models.chat_session_model import ChatSession
from app.models.dataset_model import Dataset, DatasetFileLink
from app.models.token_counts_model import (
    DatasetEmbeddingTokenCount,
    OrganizationEmbeddingTokenCount,
    WorkspaceEmbeddingTokenCount,
)
from app.models.workspace_model import Workspace
from app.schemas.token_counts_schema import (
    ChatAppTokenCountLLMResponse,
    ChatSessionTokenCountLLMResponse,
    ITokenCountCreate,
    OrganizationTokenCountLLMResponse,
    WorkspaceTokenCountLLMResponse,
)


class CRUDTokenCount(
    CRUDBase[OrganizationEmbeddingTokenCount, ITokenCountCreate, None]
):

    async def workspace_exists_in_org_include_soft_delete(
        self,
        workspace_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> None:
        result = await db_session.execute(
            select(Workspace.id).where(
                Workspace.id == workspace_id,
                Workspace.organization_id == organization_id,
            )
        )
        if (
            result.scalar_one_or_none() is None
        ):  ## same exception workflow as in workspace crud but including soft deleted worspaces
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace {workspace_id} does not exist in this organization",
            )

    async def dataset_exists_in_org_include_soft_delete(
        self,
        dataset_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> None:
        result = await db_session.execute(
            select(Dataset.id)
            .join(Workspace, Dataset.workspace_id == Workspace.id)
            .where(
                Dataset.id == dataset_id,
                Workspace.organization_id == organization_id,
            )
        )
        if (
            result.scalar_one_or_none() is None
        ):  ## same exception workflow as in workspace crud but including soft deleted worspaces
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"dataset {dataset_id} does not exist in this organization",
            )

    async def get_dataset_ids_by_org_no(
        self, organization_id: UUID, db_session: AsyncSession | None = None
    ) -> List[UUID]:
        db_session = db_session or self.db.session
        query = (
            select(Dataset.id)
            .join(Workspace, Dataset.workspace_id == Workspace.id)
            .where(Workspace.organization_id == organization_id)
        )
        result = await db_session.execute(query)
        return result.scalars().all()

    async def get_datasets_for_workspace_no(
        self, workspace_id: UUID, db_session: AsyncSession | None = None
    ) -> List[UUID]:
        db_session = db_session or self.db.session
        query = select(Dataset.id).where(Dataset.workspace_id == workspace_id)
        result = await db_session.execute(query)
        return result.scalars().all()

    async def calculate_dataset_token_counts(
        self,
        dataset_ids: List[UUID],
        r2r_token_map: Dict[UUID, int],
        db_session: AsyncSession | None = None,
    ) -> Tuple[Dict, int]:

        db_session = db_session or super().get_db().session

        dataset_token_counts = {}
        total = 0

        for dataset_id in dataset_ids:
            query = select(DatasetFileLink.r2r_id).where(
                DatasetFileLink.dataset_id == dataset_id
            )
            result = await db_session.execute(query)
            r2r_ids = result.scalars().all()

            tokens = sum(r2r_token_map.get(r2r_id, 0) for r2r_id in r2r_ids)
            dataset_token_counts[dataset_id] = tokens
            total += tokens

        return dataset_token_counts, total

    async def create_token_count(
        self,
        *,
        obj_in: ITokenCountCreate,
        db_session: AsyncSession | None = None,
    ) -> OrganizationEmbeddingTokenCount:

        db_session = db_session or super().get_db().session

        org_record = OrganizationEmbeddingTokenCount(
            organization_id=obj_in.organization_id,
            org_level_token_count=obj_in.org_level_tokens,
        )
        db_session.add(org_record)
        await db_session.commit()

        await db_session.refresh(org_record)

        if obj_in.workspace_map:
            for workspace_id, token_count in obj_in.workspace_map.items():
                workspace_record = WorkspaceEmbeddingTokenCount(
                    workspace_id=workspace_id,
                    org_token_count_id=org_record.id,
                    workspace_level_token_count=token_count,
                )
                db_session.add(workspace_record)

        if obj_in.dataset_map:
            for dataset_id, token_count in obj_in.dataset_map.items():
                query = select(Dataset.workspace_id).where(Dataset.id == dataset_id)
                result = await db_session.execute(query)
                workspace_id = result.scalar_one()

                dataset_record = DatasetEmbeddingTokenCount(
                    dataset_id=dataset_id,
                    org_token_count_id=org_record.id,
                    workspace_id=workspace_id,
                    dataset_level_token_count=token_count,
                )
                db_session.add(dataset_record)

        await db_session.commit()

        return obj_in

    async def get_workspace_llm_token_counts(
        self,
        workspace_id: UUID,
        start_date: datetime,
        db_session: Optional[AsyncSession] = None,
        chat_app_type: Optional[str] = None,
        end_date: Optional[datetime] = None,
    ) -> WorkspaceTokenCountLLMResponse:

        db_session = db_session or super().get_db().session
        query = select(
            func.sum(ChatApp.input_tokens_per_chat_app).label("input_tokens"),
            func.sum(ChatApp.output_tokens_per_chat_app).label("output_tokens"),
            func.sum(ChatApp.total_tokens_per_chat_app).label("total_tokens"),
        ).where(
            ChatApp.workspace_id == workspace_id,
            (ChatApp.created_at >= start_date) | (ChatApp.updated_at >= start_date),
        )

        if end_date:
            end_date += timedelta(days=1)
            query = query.where(
                (ChatApp.created_at < end_date) | (ChatApp.updated_at < end_date)
            )

        if chat_app_type:
            query = query.where(ChatApp.chat_app_type == chat_app_type)

        result = await db_session.execute(query)
        row = result.first()

        if not row or row.total_tokens is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No LLM token usage found for workspace {workspace_id}.",
            )

        return WorkspaceTokenCountLLMResponse(
            workspace_id=workspace_id,
            input_tokens=row.input_tokens if row and row.input_tokens else 0,
            output_tokens=row.output_tokens if row and row.output_tokens else 0,
            total_tokens=row.total_tokens if row and row.total_tokens else 0,
        )

    async def get_organization_llm_token_counts(
        self,
        organization_id: UUID,
        start_date: datetime,
        db_session: Optional[AsyncSession] = None,
        chat_app_type: Optional[str] = None,
        end_date: Optional[datetime] = None,
    ) -> OrganizationTokenCountLLMResponse:

        db_session = db_session or super().get_db().session
        query = (
            select(
                func.sum(ChatApp.input_tokens_per_chat_app).label("input_tokens"),
                func.sum(ChatApp.output_tokens_per_chat_app).label("output_tokens"),
                func.sum(ChatApp.total_tokens_per_chat_app).label("total_tokens"),
            )
            .join(Workspace, ChatApp.workspace_id == Workspace.id)
            .where(
                Workspace.organization_id == organization_id,
                (ChatApp.created_at >= start_date) | (ChatApp.updated_at >= start_date),
            )
        )

        if end_date:
            end_date += timedelta(days=1)
            query = query.where(
                (ChatApp.created_at < end_date) | (ChatApp.updated_at < end_date)
            )

        if chat_app_type:
            query = query.where(ChatApp.chat_app_type == chat_app_type)

        result = await db_session.execute(query)
        row = result.first()

        if not row or row.total_tokens is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No LLM token usage found for organization {organization_id}.",
            )

        return OrganizationTokenCountLLMResponse(
            organization_id=organization_id,
            input_tokens=row.input_tokens if row and row.input_tokens else 0,
            output_tokens=row.output_tokens if row and row.output_tokens else 0,
            total_tokens=row.total_tokens if row and row.total_tokens else 0,
        )

    async def get_chatapp_llm_token_counts(
        self,
        chatapp_id: UUID,
        start_date: datetime,
        db_session: Optional[AsyncSession] = None,
        chat_app_type: Optional[str] = None,
        end_date: Optional[datetime] = None,
    ) -> ChatAppTokenCountLLMResponse:

        db_session = db_session or super().get_db().session
        query = select(ChatApp).where(ChatApp.id == chatapp_id)

        query = query.where(
            ChatApp.created_at >= start_date,
            ChatApp.updated_at >= start_date,
        )

        if end_date:
            end_date += timedelta(days=1)
            query = query.where(
                (ChatApp.created_at < end_date) | (ChatApp.updated_at < end_date)
            )

        if chat_app_type:
            query = query.where(ChatApp.chat_app_type == chat_app_type)

        result = await db_session.execute(query)
        chat_app = result.scalar_one_or_none()

        if not chat_app:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ChatApp {chatapp_id} not found in the specified date range.",
            )

        if chat_app:
            return ChatAppTokenCountLLMResponse(
                chatapp_id=chatapp_id,
                input_tokens=chat_app.input_tokens_per_chat_app or 0,
                output_tokens=chat_app.output_tokens_per_chat_app or 0,
                total_tokens=chat_app.total_tokens_per_chat_app or 0,
            )

    async def get_chat_session_llm_token_counts(
        self,
        chat_session_id: UUID,
        db_session: Optional[AsyncSession] = None,
    ) -> ChatSessionTokenCountLLMResponse:

        db_session = db_session or super().get_db().session

        query = select(ChatSession).where(ChatSession.id == chat_session_id)
        result = await db_session.execute(query)
        chat_session = result.scalar_one_or_none()

        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {chat_session_id} not found.",
            )

        if chat_session:
            return ChatSessionTokenCountLLMResponse(
                chat_session_id=chat_session_id,
                input_tokens=chat_session.input_tokens_per_chat_session or 0,
                output_tokens=chat_session.output_tokens_per_chat_session or 0,
                total_tokens=chat_session.total_tokens_per_chat_session or 0,
            )


token_count = CRUDTokenCount(OrganizationEmbeddingTokenCount)
