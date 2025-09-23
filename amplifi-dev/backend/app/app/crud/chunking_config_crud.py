from datetime import datetime
from typing import Union
from uuid import UUID

from sqlalchemy import delete
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.crud.base_crud import CRUDBase
from app.models.chunking_config_model import ChunkingConfig
from app.models.r2r_provider_chunking_config_model import (
    R2RProviderChunkingConfig,
)
from app.models.unstructured_provider_chunking_config_model import (
    UnstructuredProviderChunkingConfig,
)
from app.schemas.chunking_config_response_schema import (
    R2RProviderChunkingConfigResponse,
    UnstructuredProviderChunkingConfigResponse,
)
from app.schemas.chunking_config_schema import (
    IChunkingConfigCreate,
    IChunkingConfigUpdate,
    R2RChunkingConfig,
    UnstructuredChunkingConfig,
)
from app.utils.exceptions import IdNotFoundException

ChunkingConfigModelUnion = Union[
    ChunkingConfig, UnstructuredChunkingConfig, R2RChunkingConfig
]


class CRUDChunkingConfig(
    CRUDBase[ChunkingConfigModelUnion, IChunkingConfigCreate, IChunkingConfigUpdate]
):
    async def create_or_update_chunking_config(
        self,
        *,
        obj_in: UnstructuredChunkingConfig | R2RChunkingConfig,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Union[
        UnstructuredProviderChunkingConfigResponse,
        R2RProviderChunkingConfigResponse,
    ]:
        db_session = db_session or super().get_db().session

        # Check if an active ChunkingConfig already exists for the dataset
        result = await db_session.execute(
            select(ChunkingConfig).where(
                ChunkingConfig.dataset_id == dataset_id,
                ChunkingConfig.deleted_at.is_(None),
            )
        )
        existing_chunking_config = result.scalar_one_or_none()

        if existing_chunking_config:
            # Delete the existing provider-specific chunking config
            if existing_chunking_config.provider == "r2r":
                await db_session.execute(
                    delete(R2RProviderChunkingConfig).where(
                        R2RProviderChunkingConfig.chunking_config_id
                        == existing_chunking_config.id
                    )
                )
            elif existing_chunking_config.provider == "unstructured_local":
                await db_session.execute(
                    delete(UnstructuredProviderChunkingConfig).where(
                        UnstructuredProviderChunkingConfig.chunking_config_id
                        == existing_chunking_config.id
                    )
                )
            else:
                raise ValueError("Invalid provider value")

            # Update the existing ChunkingConfig
            existing_chunking_config.name = obj_in.name
            existing_chunking_config.provider = obj_in.provider
        else:
            # Create a new ChunkingConfig
            existing_chunking_config = ChunkingConfig(
                dataset_id=dataset_id,
                provider=obj_in.provider,
                name=obj_in.name,
            )
            db_session.add(existing_chunking_config)
            await db_session.commit()
            await db_session.refresh(existing_chunking_config)

        # Create the new provider-specific chunking config
        if obj_in.provider == "r2r":
            provider_chunking_config = R2RProviderChunkingConfig(
                chunking_config_id=existing_chunking_config.id,
                **obj_in.model_dump(),
            )
            db_session.add(provider_chunking_config)
            response = R2RProviderChunkingConfigResponse(
                **provider_chunking_config.dict(),
                id=existing_chunking_config.id,
                name=existing_chunking_config.name,
                provider=existing_chunking_config.provider,
            )
        elif obj_in.provider == "unstructured_local":
            provider_chunking_config = UnstructuredProviderChunkingConfig(
                chunking_config_id=existing_chunking_config.id,
                **obj_in.model_dump(),
            )
            db_session.add(provider_chunking_config)
            response = UnstructuredProviderChunkingConfigResponse(
                **provider_chunking_config.dict(),
                id=existing_chunking_config.id,
                name=existing_chunking_config.name,
                provider=existing_chunking_config.provider,
            )
        else:
            raise ValueError("Invalid provider value")

        await db_session.commit()
        await db_session.refresh(provider_chunking_config)

        return response

    async def get_chunking_config(
        self,
        *,
        chunking_config_id: UUID,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Union[
        UnstructuredProviderChunkingConfigResponse,
        R2RProviderChunkingConfigResponse,
    ]:
        db_session = db_session or super().get_db().session

        # Query the ChunkingConfig table to get the provider field
        result = await db_session.execute(
            select(ChunkingConfig).where(
                ChunkingConfig.id == chunking_config_id,
                ChunkingConfig.dataset_id == dataset_id,
                ChunkingConfig.deleted_at.is_(None),
            )
        )
        chunking_config_result = result.scalar_one_or_none()

        if not chunking_config_result:
            raise IdNotFoundException(ChunkingConfig, chunking_config_id)

        # Based on the provider value, query the appropriate table
        if chunking_config_result.provider == "r2r":
            result = await db_session.execute(
                select(R2RProviderChunkingConfig).where(
                    R2RProviderChunkingConfig.chunking_config_id == chunking_config_id
                )
            )
            provider_chunking_config = result.scalar_one_or_none()
            response = R2RProviderChunkingConfigResponse(
                **chunking_config_result.dict(), **provider_chunking_config.dict()
            )
        elif chunking_config_result.provider == "unstructured_local":
            result = await db_session.execute(
                select(UnstructuredProviderChunkingConfig).where(
                    UnstructuredProviderChunkingConfig.chunking_config_id
                    == chunking_config_id
                )
            )
            provider_chunking_config = result.scalar_one_or_none()
            response = UnstructuredProviderChunkingConfigResponse(
                **chunking_config_result.dict(), **provider_chunking_config.dict()
            )
        else:
            raise ValueError("Invalid provider value")

        return response

    def get_chunking_config_sync(
        self,
        *,
        chunking_config_id: UUID,
        dataset_id: UUID,
        db_session,
    ) -> Union[
        UnstructuredProviderChunkingConfigResponse,
        R2RProviderChunkingConfigResponse,
    ]:
        """
        Synchronous version of get_chunking_config.
        """
        # Query the ChunkingConfig table to get the provider field
        chunking_config_result = (
            db_session.query(ChunkingConfig)
            .filter(
                ChunkingConfig.id == chunking_config_id,
                ChunkingConfig.dataset_id == dataset_id,
                ChunkingConfig.deleted_at.is_(None),
            )
            .first()
        )

        if not chunking_config_result:
            raise IdNotFoundException(ChunkingConfig, chunking_config_id)

        # Based on the provider value, query the appropriate table
        if chunking_config_result.provider == "r2r":
            provider_chunking_config = (
                db_session.query(R2RProviderChunkingConfig)
                .filter(
                    R2RProviderChunkingConfig.chunking_config_id == chunking_config_id
                )
                .first()
            )
            response = R2RProviderChunkingConfigResponse(
                **chunking_config_result.dict(), **provider_chunking_config.dict()
            )
        elif chunking_config_result.provider == "unstructured_local":
            provider_chunking_config = (
                db_session.query(UnstructuredProviderChunkingConfig)
                .filter(
                    UnstructuredProviderChunkingConfig.chunking_config_id
                    == chunking_config_id
                )
                .first()
            )
            response = UnstructuredProviderChunkingConfigResponse(
                **chunking_config_result.dict(), **provider_chunking_config.dict()
            )
        else:
            raise ValueError("Invalid provider value")

        return response

    async def delete_chunking_config(
        self,
        *,
        chunking_config_id: UUID,
        dataset_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Union[
        UnstructuredProviderChunkingConfigResponse,
        R2RProviderChunkingConfigResponse,
    ]:
        # global response
        db_session = db_session or super().get_db().session

        # Query the ChunkingConfig table to get the provider field
        result = await db_session.execute(
            select(ChunkingConfig).where(
                ChunkingConfig.id == chunking_config_id,
                ChunkingConfig.dataset_id == dataset_id,
            )
        )
        chunking_config_result = result.scalar_one_or_none()

        if not chunking_config_result:
            raise IdNotFoundException(ChunkingConfig, chunking_config_id)

        # Based on the provider value, delete from the appropriate table
        if chunking_config_result.provider == "r2r":
            result = await db_session.execute(
                select(R2RProviderChunkingConfig).where(
                    R2RProviderChunkingConfig.chunking_config_id == chunking_config_id
                )
            )
            provider_chunking_config = result.scalar_one_or_none()
            if provider_chunking_config:
                await db_session.delete(provider_chunking_config)
                response = R2RProviderChunkingConfigResponse(
                    **chunking_config_result.dict(), **provider_chunking_config.dict()
                )
        elif chunking_config_result.provider == "unstructured_local":
            result = await db_session.execute(
                select(UnstructuredProviderChunkingConfig).where(
                    UnstructuredProviderChunkingConfig.chunking_config_id
                    == chunking_config_id
                )
            )
            provider_chunking_config = result.scalar_one_or_none()
            if provider_chunking_config:
                await db_session.delete(provider_chunking_config)
                response = UnstructuredProviderChunkingConfigResponse(
                    **chunking_config_result.dict(), **provider_chunking_config.dict()
                )
        else:
            raise ValueError("Invalid provider value")

        # Soft delete the entry from the ChunkingConfig table
        chunking_config_result.deleted_at = datetime.utcnow()
        await db_session.commit()

        return response


chunking_config = CRUDChunkingConfig(ChunkingConfigModelUnion)
