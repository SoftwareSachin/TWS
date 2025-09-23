from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class IUnstructuredStrategyEnum(str, Enum):
    auto = "auto"
    fast = "fast"
    hi_res = "hi_res"


class IUnstructuredChunkingStrategyEnum(str, Enum):
    by_title = "by_title"
    basic = "basic"


class IChunkingMethodEnum(str, Enum):
    recursive = "recursive"
    by_title = "by_title"
    basic = "basic"


class ChunkingConfigBaseSchema(BaseModel):
    name: Optional[str] = None
    provider: str

    class Config:
        title = "ChunkingConfigBase"


class UnstructuredChunkingConfig(ChunkingConfigBaseSchema):
    strategy: IUnstructuredStrategyEnum = Field(default="auto")
    chunking_strategy: IUnstructuredChunkingStrategyEnum = Field(default="by_title")
    combine_under_n_chars: Optional[int] = Field(
        None, description="Combine chunks smaller than this number of characters."
    )
    max_characters: Optional[int] = Field(
        None, description="Maximum number of characters per chunk."
    )
    coordinates: Optional[bool] = Field(
        default=None, description="Whether to include coordinates in the output."
    )
    encoding: Optional[str] = None
    extract_image_block_types: Optional[List[str]] = Field(
        default=None, description="Types of image blocks to extract."
    )
    gz_uncompressed_content_type: Optional[str] = Field(
        None, description="Content type for uncompressed gzip files."
    )
    hi_res_model_name: Optional[str] = Field(
        None, description="Name of the high-resolution model to use."
    )
    include_orig_elements: Optional[bool] = Field(
        default=None, description="Whether to include original elements in the output."
    )
    include_page_breaks: Optional[bool] = Field(
        None, description="Whether to include page breaks in the output."
    )
    languages: Optional[List[str]] = Field(
        default=None, description="List of languages to consider for text processing."
    )
    multipage_sections: Optional[bool] = Field(
        default=None, description="Whether to allow sections to span multiple pages."
    )
    new_after_n_chars: Optional[int] = Field(
        None, description="Start a new chunk after this many characters."
    )
    ocr_languages: Optional[List[str]] = Field(
        default=None, description="Languages to use for OCR."
    )
    # output_format: str = Field(
    #     default="application/json", description="Format of the output."
    # )
    overlap: Optional[int] = Field(
        default=None, description="Number of characters to overlap between chunks."
    )
    overlap_all: Optional[bool] = Field(
        default=None, description="Whether to overlap all chunks."
    )
    pdf_infer_table_structure: Optional[bool] = Field(
        default=None, description="Whether to infer table structure in PDFs."
    )
    similarity_threshold: Optional[float] = Field(
        None, description="Threshold for considering chunks similar."
    )
    skip_infer_table_types: Optional[List[str]] = Field(
        default=None, description="Types of tables to skip inferring."
    )
    split_pdf_concurrency_level: Optional[int] = Field(
        default=None, description="Concurrency level for splitting PDFs."
    )
    split_pdf_page: Optional[bool] = Field(
        default=None, description="Whether to split PDFs by page."
    )
    starting_page_number: Optional[int] = Field(
        None, description="Page number to start processing from."
    )
    unique_element_ids: Optional[bool] = Field(
        default=None, description="Whether to generate unique IDs for elements."
    )
    xml_keep_tags: Optional[bool] = Field(
        default=None, description="Whether to keep XML tags in the output."
    )

    class Config:
        title = "UnstructuredChunkingConfig"


class R2RChunkingConfig(ChunkingConfigBaseSchema):
    method: IChunkingMethodEnum = IChunkingMethodEnum.recursive
    chunk_overlap: Optional[int] = None
    chunk_size: Optional[int] = Field(default=512)
    parser_overrides: Optional[dict[str, str]] = Field(default={"pdf": "zerox"})

    class Config:
        title = "DefaultChunkingConfig"


class IChunkingConfigCreate(ChunkingConfigBaseSchema):
    pass


class IChunkingConfigUpdate(ChunkingConfigBaseSchema):
    pass
