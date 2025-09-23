from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel
from sqlmodel import Field

from app.schemas.chunking_config_schema import (
    IChunkingMethodEnum,
    IUnstructuredChunkingStrategyEnum,
    IUnstructuredStrategyEnum,
)


class ChunkingConfigResponseBase(BaseModel):
    name: Optional[str]
    provider: str
    id: UUID


class UnstructuredProviderChunkingConfigResponse(ChunkingConfigResponseBase):
    strategy: IUnstructuredStrategyEnum = Field(default="auto")
    chunking_strategy: IUnstructuredChunkingStrategyEnum = Field(default="by_title")
    combine_under_n_chars: Optional[int] = Field(None)
    max_characters: Optional[int] = Field(None)
    coordinates: Optional[bool] = Field(default=None)
    encoding: Optional[str] = None
    extract_image_block_types: Optional[List[str]] = Field(default=None)
    gz_uncompressed_content_type: Optional[str] = Field(None)
    hi_res_model_name: Optional[str] = Field(None)
    include_orig_elements: Optional[bool] = Field(default=None)
    include_page_breaks: Optional[bool] = Field(default=None)
    languages: Optional[List[str]] = Field(default=None)
    multipage_sections: Optional[bool] = Field(default=None)
    new_after_n_chars: Optional[int] = Field(default=512)
    ocr_languages: Optional[List[str]] = Field(default=None)
    # output_format: Optional[str] = Field(default="application/json")
    overlap: Optional[int] = Field(default=None)
    overlap_all: Optional[bool] = Field(default=None)
    pdf_infer_table_structure: Optional[bool] = Field(default=None)
    similarity_threshold: Optional[float] = Field(None)
    skip_infer_table_types: Optional[List[str]] = Field(default=None)
    split_pdf_concurrency_level: Optional[int] = Field(default=None)
    split_pdf_page: Optional[bool] = Field(default=None)
    starting_page_number: Optional[int] = Field(None)
    unique_element_ids: Optional[bool] = Field(default=None)
    xml_keep_tags: Optional[bool] = Field(default=None)


class R2RProviderChunkingConfigResponse(ChunkingConfigResponseBase):
    method: IChunkingMethodEnum = Field(default="recursive")
    chunk_size: Optional[int] = Field(default=1024)
    chunk_overlap: Optional[int] = Field(default=512)
    # max_chunk_size: Optional[int] = Field(default=None)
    # parser_overrides: Optional[dict[str, str]] = Field(default={"pdf": "zerox"})
    excluded_parsers: Optional[List[str]] = Field(default=["mp4"])
