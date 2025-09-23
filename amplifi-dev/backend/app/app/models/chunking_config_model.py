from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.dataset_model import Dataset
    from app.models.r2r_provider_chunking_config_model import (
        R2RProviderChunkingConfig,
    )
    from app.models.unstructured_provider_chunking_config_model import (
        UnstructuredProviderChunkingConfig,
    )


class ChunkingConfigBase(BaseUUIDModel):
    name: Optional[str] = Field(default=None)
    provider: str = Field(default="r2r", nullable=False)


class ChunkingConfig(ChunkingConfigBase, table=True):
    __tablename__ = "chunking_config"
    dataset_id: UUID = Field(foreign_key="datasets.id", nullable=False)
    dataset: "Dataset" = Relationship(back_populates="chunking_configs")

    r2r_provider_chunking_config: Optional["R2RProviderChunkingConfig"] = Relationship(
        back_populates="chunking_config"
    )

    unstructured_provider_chunking_config: Optional[
        "UnstructuredProviderChunkingConfig"
    ] = Relationship(back_populates="chunking_config")
