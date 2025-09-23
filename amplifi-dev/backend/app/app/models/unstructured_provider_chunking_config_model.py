from typing import List, Optional
from uuid import UUID

from sqlmodel import (
    ARRAY,
    Boolean,
    Column,
    Field,
    Float,
    Integer,
    Relationship,
    SQLModel,
    String,
)

from app.models.chunking_config_model import ChunkingConfig
from app.schemas.chunking_config_schema import (
    IUnstructuredChunkingStrategyEnum,
    IUnstructuredStrategyEnum,
)


class UnstructuredProviderChunkingConfigBase(SQLModel):
    strategy: IUnstructuredStrategyEnum = Field(
        sa_column=Column(String, nullable=False)
    )
    chunking_strategy: IUnstructuredChunkingStrategyEnum = Field(
        sa_column=Column(String, nullable=False)
    )
    # max_chunk_size: Optional[int] = Field(default=None, sa_column=Column(Integer))
    combine_under_n_chars: Optional[int] = Field(default=128, sa_column=Column(Integer))
    max_characters: Optional[int] = Field(default=500, sa_column=Column(Integer))
    coordinates: Optional[bool] = Field(default=None, sa_column=Column(Boolean))
    chunk_overlap: Optional[int] = Field(default=None, sa_column=Column(Integer))
    encoding: Optional[str] = Field(default=None, sa_column=Column(String))
    gz_uncompressed_content_type: Optional[str] = Field(
        default=None, sa_column=Column(String)
    )
    hi_res_model_name: Optional[str] = Field(default=None, sa_column=Column(String))
    include_orig_elements: Optional[bool] = Field(
        default=None, sa_column=Column(Boolean)
    )
    include_page_breaks: Optional[bool] = Field(default=None, sa_column=Column(Boolean))
    multipage_sections: Optional[bool] = Field(default=None, sa_column=Column(Boolean))
    new_after_n_chars: Optional[int] = Field(default=1500, sa_column=Column(Integer))
    # output_format: Optional[str] = Field(
    #     default="application/json", sa_column=Column(String)
    # )
    overlap: int = Field(default=64, sa_column=Column(Integer))
    overlap_all: Optional[bool] = Field(default=None, sa_column=Column(Boolean))
    pdf_infer_table_structure: Optional[bool] = Field(
        default=None, sa_column=Column(Boolean)
    )
    similarity_threshold: Optional[float] = Field(default=None, sa_column=Column(Float))
    split_pdf_concurrency_level: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    split_pdf_page: Optional[bool] = Field(default=None, sa_column=Column(Boolean))
    starting_page_number: Optional[int] = Field(default=None, sa_column=Column(Integer))
    unique_element_ids: Optional[bool] = Field(default=None, sa_column=Column(Boolean))
    xml_keep_tags: Optional[bool] = Field(default=None, sa_column=Column(Boolean))
    extract_image_block_types: Optional[List[str]] = Field(
        default=None, sa_column=Column(ARRAY(String))
    )
    languages: List[str] = Field(default=None, sa_column=Column(ARRAY(String)))
    ocr_languages: Optional[List[str]] = Field(
        default=None, sa_column=Column(ARRAY(String))
    )
    skip_infer_table_types: Optional[List[str]] = Field(
        default=None, sa_column=Column(ARRAY(String))
    )


class UnstructuredProviderChunkingConfig(
    UnstructuredProviderChunkingConfigBase, table=True
):
    __tablename__ = "unstructured_provider_chunking_config"
    chunking_config_id: UUID = Field(
        primary_key=True, foreign_key="chunking_config.id", nullable=False, unique=True
    )
    chunking_config: "ChunkingConfig" = Relationship(
        back_populates="unstructured_provider_chunking_config"
    )
