from datetime import datetime
from uuid import UUID

from fastapi import HTTPException, status
from fastapi_pagination import Page, Params, paginate
from sqlalchemy import and_, asc, desc, func
from sqlmodel import insert, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.be_core.config import settings
from app.crud.base_crud import CRUDBase
from app.crud.user_crud import user as user_crud
from app.models import Agent, Source
from app.models.chat_app_datsets_model import LinkChatAppDatasets
from app.models.chat_app_generation_config_model import (
    ChatAppGenerationConfig,
    ChatAppGenerationConfigBase,
)
from app.models.chat_app_model import ChatApp
from app.models.dataset_model import Dataset
from app.schemas.chat_schema import (
    IChatAppCreate,
    IChatAppRead,
    IChatAppTypeEnum,
    IChatAppV2Create,
    IChatAppV2Read,
    ISqlChatAppCreate,
    ISqlChatAppRead,
)
from app.schemas.common_schema import IOrderEnum


class CRUDChatApp(CRUDBase[ChatApp, IChatAppCreate, None]):

    async def _get_chatapp_by_id(
        self,
        *,
        chatapp_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> any:
        chatapp = await super().get(id=chatapp_id, db_session=db_session)
        if not chatapp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ChatApp not found: {chatapp_id} ",
            )
        if chatapp.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ChatApp {chatapp_id} has been deleted",
            )
        return chatapp

    async def get_chatapp_by_id(
        self,
        *,
        chatapp_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IChatAppRead:
        db_session = db_session or super().get_db().session
        chatapp = await self._get_chatapp_by_id(
            chatapp_id=chatapp_id, db_session=db_session
        )
        return IChatAppRead(
            **chatapp.model_dump(),
            datasets=[dataset.id for dataset in chatapp.datasets],
            generation_config=(
                ChatAppGenerationConfigBase(
                    **chatapp.chat_app_generation_config.model_dump()
                )
                if chatapp.chat_app_generation_config
                else None
            ),
        )

    async def get_chatapp_v2_by_id(
        self,
        *,
        chatapp_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IChatAppV2Read:
        db_session = db_session or super().get_db().session
        chatapp = await self._get_chatapp_by_id(
            chatapp_id=chatapp_id, db_session=db_session
        )
        return IChatAppV2Read(
            **chatapp.model_dump(),
        )

    async def get_sql_chatapp_by_id(
        self,
        *,
        chatapp_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> ISqlChatAppRead:

        db_session = db_session or super().get_db().session
        chatapp = await self.get_chatapp_by_id(
            chatapp_id=chatapp_id, db_session=db_session
        )
        return ISqlChatAppRead(
            **chatapp.model_dump(),
            generation_config=ChatAppGenerationConfigBase(
                **chatapp.chat_app_generation_config.model_dump()
            ),
        )

    async def get_chatapps(
        self,
        *,
        workspace_id: UUID,
        pagination_params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[ChatApp]:
        db_session = db_session or super().get_db().session
        columns = ChatApp.__table__.columns
        if order_by not in columns:
            order_by = "created_at"
        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )
        query = (
            select(ChatApp)
            .where(
                and_(
                    ChatApp.workspace_id == workspace_id,
                    ChatApp.deleted_at.is_(None),
                )
            )
            .order_by(order_clause)
        )
        result = await db_session.execute(query)
        chatapp_results = result.scalars().all()
        return paginate(chatapp_results, pagination_params)

    async def get_chatapps_by_user_id(
        self,
        *,
        user_id: UUID,
        pagination_params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[ChatApp]:
        db_session = db_session or super().get_db().session
        columns = ChatApp.__table__.columns
        if order_by not in columns:
            order_by = "created_at"
        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )
        user_workspace_ids = await user_crud.get_workspace_ids_for_user(user_id=user_id)
        query = (
            select(ChatApp)
            .where(
                and_(
                    ChatApp.workspace_id.in_(user_workspace_ids),
                    ChatApp.deleted_at.is_(None),
                )
            )
            .order_by(order_clause)
        )
        result = await db_session.execute(query)
        chatapp_results = result.scalars().all()
        return paginate(chatapp_results, pagination_params)

    async def create_unstructured_chatapp(
        self,
        *,
        obj_in: IChatAppCreate,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IChatAppRead:
        db_session = db_session or super().get_db().session
        await self._validate_unstructured_datasets(dataset_ids=obj_in.datasets)
        await self._validate_chat_app_name(workspace_id, obj_in.name, db_session)
        chat_app = ChatApp(
            chat_app_type=obj_in.chat_app_type,
            name=obj_in.name,
            description=obj_in.description,
            system_prompt=obj_in.system_prompt,
            chat_retention_days=obj_in.chat_retention_days,
            workspace_id=workspace_id,
            voice_enabled=obj_in.voice_enabled,
            graph_enabled=obj_in.graph_enabled,
        )
        db_session.add(chat_app)
        generation_config = ChatAppGenerationConfig(
            llm_model=obj_in.generation_config.llm_model,
            max_chunks_retrieved=obj_in.generation_config.max_chunks_retrieved,
            temperature=obj_in.generation_config.temperature,
            top_p=obj_in.generation_config.top_p,
            max_tokens_to_sample=obj_in.generation_config.max_tokens_to_sample,
            chatapp_id=chat_app.id,
        )
        db_session.add(generation_config)
        dataset_links = [
            {"chatapp_id": chat_app.id, "dataset_id": dataset_id}
            for dataset_id in obj_in.datasets
        ]
        await db_session.execute(insert(LinkChatAppDatasets).values(dataset_links))
        await db_session.commit()
        await db_session.refresh(chat_app)
        return await self.get_chatapp_by_id(chatapp_id=chat_app.id)

    async def create_sql_chatapp(
        self,
        *,
        obj_in: ISqlChatAppCreate,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IChatAppRead:
        db_session = db_session or super().get_db().session
        await self._validate_sql_dataset(dataset_id=obj_in.dataset_id)
        await self._validate_chat_app_name(workspace_id, obj_in.name, db_session)

        chat_app = ChatApp(
            chat_app_type=obj_in.chat_app_type,
            name=obj_in.name,
            description=obj_in.description,
            # system_prompt=obj_in.system_prompt,
            # chat_retention_days=obj_in.chat_retention_days,
            workspace_id=workspace_id,
            voice_enabled=obj_in.voice_enabled,
        )
        db_session.add(chat_app)
        generation_config = ChatAppGenerationConfig(
            llm_model=obj_in.generation_config.llm_model,
            # max_chunks_retrieved=obj_in.generation_config.max_chunks_retrieved,
            # temperature=obj_in.generation_config.temperature,
            # top_p=obj_in.generation_config.top_p,
            # max_tokens_to_sample=obj_in.generation_config.max_tokens_to_sample,
            chatapp_id=chat_app.id,
        )
        db_session.add(generation_config)
        # dataset_links = [
        #     {"chatapp_id": chat_app.id, "dataset_id": dataset_id}
        #     for dataset_id in obj_in.dataset_id
        # ]
        dataset_links = [{"chatapp_id": chat_app.id, "dataset_id": obj_in.dataset_id}]
        await db_session.execute(insert(LinkChatAppDatasets).values(dataset_links))
        await db_session.commit()
        await db_session.refresh(chat_app)
        return await self.get_chatapp_by_id(chatapp_id=chat_app.id)

    async def create_chatapp_v2(
        self,
        *,
        obj_in: IChatAppV2Create,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> ChatApp:
        db_session = db_session or super().get_db().session
        await self._validate_agent(obj_in.agent_id)
        await self._validate_chat_app_name(workspace_id, obj_in.name, db_session)
        chat_app = ChatApp(
            chat_app_type=IChatAppTypeEnum.agentic,
            name=obj_in.name,
            description=obj_in.description,
            workspace_id=workspace_id,
            voice_enabled=obj_in.voice_enabled,
            agent_id=obj_in.agent_id,
        )
        db_session.add(chat_app)
        await db_session.commit()
        await db_session.refresh(chat_app)
        return chat_app

    async def _validate_unstructured_datasets(
        self, dataset_ids: list[UUID], db_session: AsyncSession | None = None
    ):
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Dataset.id, Source.source_type)
            .join(Source, Dataset.source_id == Source.id, isouter=True)
            .where(Dataset.id.in_(dataset_ids))
        )
        dataset_info = result.fetchall()

        # Check for missing datasets
        existing_ids = {row[0] for row in dataset_info}
        missing_ids = set(dataset_ids) - existing_ids
        if missing_ids:
            raise ValueError(
                f"The following datasets do not exist: {[str(ds) for ds in missing_ids]}"
            )

        # Check for invalid source types
        invalid_ids = [
            str(row[0])
            for row in dataset_info
            if row[1] is not None and row[1] not in settings.UNSTRUCTURED_SOURCES
        ]
        if invalid_ids:
            raise ValueError(
                f"The following datasets are not from unstructured sources: {invalid_ids}"
            )

    async def _validate_sql_dataset(
        self, dataset_id: UUID, db_session: AsyncSession | None = None
    ):
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Dataset.id, Source.source_type)
            .join(Source, Dataset.source_id == Source.id, isouter=True)
            .where(Dataset.id == dataset_id)
        )
        row = result.first()

        if not row or not row[0]:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset with id {dataset_id} does not exist.",
            )

        if row[1] not in settings.STRUCTURED_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset {dataset_id} is not from a structured source.",
            )

    async def _validate_chat_app_name(
        self,
        workspace_id: UUID,
        chat_app_name: str,
        db_session: AsyncSession | None = None,
    ):
        existing_query = select(ChatApp).where(
            func.lower(func.trim(ChatApp.name)) == func.lower(func.trim(chat_app_name)),
            ChatApp.workspace_id == workspace_id,
            ChatApp.deleted_at.is_(None),
        )
        existing_result = await db_session.execute(existing_query)
        if existing_result.scalars().first():
            raise HTTPException(
                status_code=400,
                detail=f"A chat app with name '{chat_app_name}' already exists in this workspace.",
            )

    async def _validate_agent(
        self, agent_id: UUID, db_session: AsyncSession | None = None
    ):
        db_session = db_session or super().get_db().session
        result = await db_session.execute(select(Agent.id).where(Agent.id == agent_id))
        row = result.first()

        if not row or not row[0]:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with id {agent_id} does not exist.",
            )

    async def update_unstructured_chatapp(
        self,
        *,
        chatapp_id: UUID,
        obj_in: IChatAppCreate,
        db_session: AsyncSession | None = None,
    ) -> IChatAppRead:
        db_session = db_session or super().get_db().session
        await self._validate_unstructured_datasets(dataset_ids=obj_in.datasets)

        query = select(ChatApp).where(ChatApp.id == chatapp_id)
        result = await db_session.execute(query)
        chat_app: ChatApp = result.scalar_one_or_none()

        if not chat_app:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ChatApp with id {chatapp_id} not found",
            )

        # Name uniqueness check (excluding deleted apps)
        if obj_in.name and obj_in.name != chat_app.name:
            await self._validate_chat_app_name(
                chat_app.workspace_id, obj_in.name, db_session
            )

        chat_app.name = obj_in.name
        chat_app.description = obj_in.description
        chat_app.system_prompt = obj_in.system_prompt
        chat_app.chat_retention_days = obj_in.chat_retention_days
        chat_app.voice_enabled = obj_in.voice_enabled
        chat_app.graph_enabled = obj_in.graph_enabled
        db_session.add(chat_app)

        # Update generation config
        gen_config_query = select(ChatAppGenerationConfig).where(
            ChatAppGenerationConfig.chatapp_id == chatapp_id
        )
        gen_config_result = await db_session.execute(gen_config_query)
        generation_config: ChatAppGenerationConfig = (
            gen_config_result.scalar_one_or_none()
        )
        if generation_config:
            generation_config.llm_model = obj_in.generation_config.llm_model
            generation_config.max_chunks_retrieved = (
                obj_in.generation_config.max_chunks_retrieved
            )
            generation_config.temperature = obj_in.generation_config.temperature
            generation_config.top_p = obj_in.generation_config.top_p
            generation_config.max_tokens_to_sample = (
                obj_in.generation_config.max_tokens_to_sample
            )
            db_session.add(generation_config)

        # Update datasets
        old_links_query = select(LinkChatAppDatasets).where(
            LinkChatAppDatasets.chatapp_id == chatapp_id
        )
        old_links_result = await db_session.execute(old_links_query)
        old_links = old_links_result.scalars().all()
        for link in old_links:
            await db_session.delete(link)

        dataset_links = [
            {"chatapp_id": chatapp_id, "dataset_id": dataset_id}
            for dataset_id in obj_in.datasets
        ]
        await db_session.execute(insert(LinkChatAppDatasets).values(dataset_links))

        # Commit changes
        await db_session.commit()
        await db_session.refresh(chat_app)

        return await self.get_chatapp_by_id(chatapp_id=chat_app.id)

    async def update_sql_chatapp(
        self,
        *,
        chatapp_id: UUID,
        obj_in: ISqlChatAppCreate,
        db_session: AsyncSession | None = None,
    ) -> IChatAppRead:
        db_session = db_session or super().get_db().session
        await self._validate_sql_dataset(dataset_id=obj_in.dataset_id)

        query = select(ChatApp).where(ChatApp.id == chatapp_id)
        result = await db_session.execute(query)
        chat_app: ChatApp = result.scalar_one_or_none()

        if not chat_app:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ChatApp with id {chatapp_id} not found",
            )

        # Name uniqueness check (excluding deleted apps)
        if obj_in.name and obj_in.name != chat_app.name:
            await self._validate_chat_app_name(
                chat_app.workspace_id, obj_in.name, db_session
            )

        # Update ChatApp fields
        chat_app.name = obj_in.name
        chat_app.description = obj_in.description
        chat_app.system_prompt = obj_in.system_prompt
        chat_app.chat_retention_days = obj_in.chat_retention_days
        chat_app.voice_enabled = obj_in.voice_enabled
        db_session.add(chat_app)

        # Update generation config (only llm_model)
        gen_config_query = select(ChatAppGenerationConfig).where(
            ChatAppGenerationConfig.chatapp_id == chatapp_id
        )
        gen_config_result = await db_session.execute(gen_config_query)
        generation_config: ChatAppGenerationConfig = (
            gen_config_result.scalar_one_or_none()
        )

        if generation_config:
            generation_config.llm_model = obj_in.generation_config.llm_model
            db_session.add(generation_config)

        # Update dataset link
        old_link_query = select(LinkChatAppDatasets).where(
            LinkChatAppDatasets.chatapp_id == chatapp_id
        )
        old_link_result = await db_session.execute(old_link_query)
        old_link = old_link_result.scalar_one_or_none()

        if old_link:
            await db_session.delete(old_link)

        new_link = LinkChatAppDatasets(
            chatapp_id=chatapp_id, dataset_id=obj_in.dataset_id
        )
        db_session.add(new_link)

        # Commit changes
        await db_session.commit()
        await db_session.refresh(chat_app)

        return await self.get_chatapp_by_id(chatapp_id=chat_app.id)

    async def get_workspace_id_of_chatapp(
        self,
        *,
        chatapp_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> UUID:
        db_session = db_session or super().get_db().session

        query = select(ChatApp).where(
            and_(
                ChatApp.id == chatapp_id,
                ChatApp.deleted_at.is_(None),
            )
        )

        result = await db_session.execute(query)
        chatapp: ChatApp = result.unique().scalar_one_or_none()

        if not chatapp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ChatApp {chatapp_id} not found",
            )

        return chatapp.workspace_id

    async def delete_chatapp_by_id(
        self,
        *,
        chatapp_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> IChatAppRead:
        db_session = db_session or super().get_db().session

        query = select(ChatApp).where(
            and_(
                ChatApp.id == chatapp_id,
                ChatApp.deleted_at.is_(None),
            )
        )

        result = await db_session.execute(query)
        chatapp: ChatApp = result.unique().scalar_one_or_none()

        if not chatapp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ChatApp {chatapp_id} not found",
            )

        chatapp.deleted_at = datetime.utcnow()
        db_session.add(chatapp)
        await db_session.commit()

        return IChatAppRead(
            **chatapp.model_dump(),
            datasets=[dataset.id for dataset in chatapp.datasets],
            generation_config=(
                ChatAppGenerationConfigBase(
                    **chatapp.chat_app_generation_config.model_dump()
                )
                if chatapp.chat_app_generation_config
                else None
            ),
        )

    async def delete_chatapps_by_workspace_id(
        self, *, workspace_id: UUID, db_session: AsyncSession | None = None
    ) -> list[IChatAppRead]:
        db_session = db_session or super().get_db().session

        query = select(ChatApp).where(
            and_(ChatApp.workspace_id == workspace_id, ChatApp.deleted_at.is_(None))
        )

        result = await db_session.execute(query)
        chatapps: list[ChatApp] = result.scalars().all()

        deleted_chatapps = []

        for chatapp in chatapps:
            chatapp.deleted_at = datetime.utcnow()
            db_session.add(chatapp)
            deleted_chatapps.append(
                IChatAppRead(
                    **chatapp.model_dump(),
                    datasets=[dataset.id for dataset in chatapp.datasets],
                    generation_config=(
                        ChatAppGenerationConfigBase(
                            **chatapp.chat_app_generation_config.model_dump()
                        )
                        if chatapp.chat_app_generation_config
                        else None
                    ),
                )
            )

        await db_session.commit()
        return deleted_chatapps

    async def get_dataset_id_by_chatapp_id(
        self, *, chatapp_id: UUID, db_session: AsyncSession | None = None
    ) -> UUID | None:
        db_session = db_session or super().get_db().session
        query = select(LinkChatAppDatasets.dataset_id).where(
            LinkChatAppDatasets.chatapp_id == chatapp_id
        )
        result = await db_session.execute(query)
        dataset_id = result.scalar_one_or_none()
        return dataset_id

    async def get_chatapps_v2(
        self,
        *,
        workspace_id: UUID,
        pagination_params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[ChatApp]:
        db_session = db_session or super().get_db().session

        columns = ChatApp.__table__.columns
        if order_by not in columns:
            order_by = "created_at"

        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = (
            select(ChatApp)
            .where(
                and_(
                    ChatApp.workspace_id == workspace_id,
                    ChatApp.deleted_at.is_(None),
                    ChatApp.agent_id.is_not(None),  # Only V2 apps (agent-based)
                )
            )
            .order_by(order_clause)
        )

        result = await db_session.execute(query)
        chatapp_results = result.scalars().all()
        return paginate(chatapp_results, pagination_params)

    async def update_chatapp_v2(
        self,
        *,
        chatapp_id: UUID,
        workspace_id: UUID,
        obj_in: IChatAppV2Create,
        db_session: AsyncSession | None = None,
    ) -> IChatAppV2Read:
        db_session = db_session or super().get_db().session

        # Validate agent existence
        await self._validate_agent(obj_in.agent_id, db_session)

        # Fetch chatapp using direct SQL query
        query = select(ChatApp).where(ChatApp.id == chatapp_id)
        result = await db_session.execute(query)
        chatapp: ChatApp | None = result.scalar_one_or_none()

        if not chatapp:
            raise HTTPException(
                status_code=404,
                detail=f"ChatApp with id {chatapp_id} not found",
            )

        # Check name conflict (exclude self)
        name_conflict_query = select(ChatApp).where(
            func.lower(func.trim(ChatApp.name)) == func.lower(func.trim(obj_in.name)),
            ChatApp.workspace_id == workspace_id,
            ChatApp.deleted_at.is_(None),
            ChatApp.id != chatapp_id,
        )
        existing_result = await db_session.execute(name_conflict_query)
        if existing_result.scalars().first():
            raise HTTPException(
                status_code=400,
                detail=f"A chat app with name '{obj_in.name}' already exists in this workspace.",
            )

        # Update fields
        chatapp.name = obj_in.name
        chatapp.description = obj_in.description
        chatapp.voice_enabled = obj_in.voice_enabled
        chatapp.agent_id = obj_in.agent_id
        chatapp.chat_app_type = IChatAppTypeEnum.agentic

        db_session.add(chatapp)
        await db_session.commit()
        await db_session.refresh(chatapp)

        return IChatAppV2Read.model_validate(chatapp)


chatapp = CRUDChatApp(ChatApp)
