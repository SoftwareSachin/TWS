from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.crud.base_crud import CRUDBase
from app.models.embeddingConfig_model import EmbeddingConfig
from app.schemas.embeddingConfig_schema import (
    IEmbeddingConfigCreate,
    IEmbeddingConfigUpdate,
)


class CRUDEmbeddingConfig(
    CRUDBase[EmbeddingConfig, IEmbeddingConfigCreate, IEmbeddingConfigUpdate]
):

    # Create an Embedding Configuration in the dataset
    async def create_embedding_config(
        self,
        *,
        obj_in: IEmbeddingConfigCreate,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> EmbeddingConfig:
        db_session = db_session or super().get_db().session
        embedding_config = EmbeddingConfig(
            dataset_id=dataset_id,
            name=obj_in.name,
            provider=obj_in.provider,
            base_model=obj_in.base_model,
            rerank_model=obj_in.rerank_model,
            base_dimension=obj_in.base_dimension,
            batch_size=obj_in.batch_size,
            add_title_as_prefix=obj_in.add_title_as_prefix,
        )

        db_session.add(embedding_config)
        await db_session.commit()
        await db_session.refresh(embedding_config)

        return embedding_config

    # Get Embedding Configuration by ID
    async def get_embedding_config(
        self,
        *,
        embedding_config_id: UUID,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> EmbeddingConfig | None:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(EmbeddingConfig).where(EmbeddingConfig.id == embedding_config_id)
        )
        return result.scalar_one_or_none()

    # Update Embedding Configuration
    async def update_embedding_config(
        self,
        *,
        embedding_config_id: UUID,
        dataset_id: UUID,
        obj_in: IEmbeddingConfigUpdate,
        db_session: AsyncSession | None = None,
    ) -> EmbeddingConfig:
        db_session = db_session or super().get_db().session
        embedding_config = await self.get_embedding_config(
            dataset_id=dataset_id,
            embedding_config_id=embedding_config_id,
            db_session=db_session,
        )

        if not embedding_config:
            raise ValueError(
                f"Embedding configuration with ID {embedding_config_id} not found."
            )

        if obj_in.name is not None:
            embedding_config.name = obj_in.name
        if obj_in.is_active is not None:
            embedding_config.is_active = obj_in.is_active

        # Update the fields
        for field, value in obj_in.dict(exclude_unset=True).items():
            setattr(embedding_config, field, value)

        await db_session.commit()
        await db_session.refresh(embedding_config)

        return embedding_config

    # Delete Embedding Configuration
    async def delete_embedding_config(
        self,
        *,
        embedding_config_id: UUID,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> EmbeddingConfig:
        db_session = db_session or super().get_db().session
        embedding_config = await self.get_embedding_config(
            embedding_config_id=embedding_config_id,
            dataset_id=dataset_id,
            db_session=db_session,
        )

        if not embedding_config:
            raise ValueError(
                f"Embedding configuration with ID {embedding_config_id} not found."
            )

        await db_session.delete(embedding_config)
        await db_session.commit()

        return embedding_config


embedding_config = CRUDEmbeddingConfig(EmbeddingConfig)
