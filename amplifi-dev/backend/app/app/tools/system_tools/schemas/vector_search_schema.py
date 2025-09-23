from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.rag_generation_schema import IFileContextAggregation


class VectorSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language question or search query to find information within document content",
    )
    dataset_ids: List[UUID] = Field(
        ...,
        description="Dataset IDs to search within for content-based information retrieval",
    )
    file_ids: Optional[List[UUID]] = Field(
        None,
        description="Optional specific file IDs to focus search on. When provided, search will be limited to these files within the specified datasets for more targeted results.",
    )


class VectorSearchResult(BaseModel):
    chunk_id: UUID = Field(..., description="Unique identifier for the content chunk")
    source: Optional[str] = Field(
        None, description="Source file containing this content"
    )


class RagContext(BaseModel):
    text: str = Field(..., description="Actual text content from the document")
    file: Dict[str, str] = Field(..., description="File metadata containing this text")
    download_url: Optional[str] = Field(
        None, description="URL to download the source file"
    )
    chunk_id: Optional[UUID] = Field(None, description="Content chunk identifier")


class ChunkMetadata(BaseModel):
    match_type: Optional[str]
    base64: Optional[str] = None
    chunk_id: Optional[UUID]
    table_html: Optional[str] = None


class ImageSearchResult(BaseModel):
    file_path: str
    text: Optional[str]
    mimetype: Optional[str]
    file_id: UUID
    dataset_id: UUID
    search_score: Optional[float]
    chunk_metadata: Optional[ChunkMetadata] = None


class VectorSearchOutput(BaseModel):
    results: List[VectorSearchResult] = Field(
        ..., description="Content chunks matching the search query"
    )
    aggregated: List[IFileContextAggregation] = Field(
        ..., description="Organized content results by file"
    )
