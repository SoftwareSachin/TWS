from typing import Optional
from uuid import UUID

from sqlalchemy import ARRAY
from sqlmodel import Column, Field, Integer, Relationship, SQLModel, String

from app.models.chunking_config_model import ChunkingConfig
from app.schemas.chunking_config_schema import IChunkingMethodEnum


class R2RProviderChunkingConfigBase(SQLModel):
    chunking_strategy: IChunkingMethodEnum = Field(
        default=IChunkingMethodEnum.recursive, sa_column=Column(String, nullable=False)
    )
    chunk_size: Optional[int] = Field(default=None, sa_column=Column(Integer))
    chunk_overlap: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # max_chunk_size: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # parser_overrides: Optional[dict[str, str]] = Field(
    #     default=None, sa_column=Column(String)
    # )
    excluded_parsers: Optional[list[str]] = Field(
        default=["mp4"], sa_column=Column(ARRAY(String()))
    )


class R2RProviderChunkingConfig(R2RProviderChunkingConfigBase, table=True):
    __tablename__ = "r2r_provider_chunking_config"
    chunking_config_id: UUID = Field(
        primary_key=True, foreign_key="chunking_config.id", nullable=False, unique=True
    )

    chunking_config: "ChunkingConfig" = Relationship(
        back_populates="r2r_provider_chunking_config"
    )
