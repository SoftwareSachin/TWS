from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

### pydantic classes

# class HybridSearchSettings(BaseModel):
#     full_text_weight: Optional[int] = 1
#     semantic_weight: Optional[int] = 5
#     full_text_limit: Optional[int] = 200
#     rrf_k: Optional[int] = 50


class VectorSearchSettings(BaseModel):
    search_limit: int = Field(default=3)  # default 10 in R2R API
    # filters: Dict[str, Dict[str, str]]
    # use_vector_search: bool
    search_index_type: Literal[
        "cosine_distance", "l2_distance", "max_inner_product"
    ] = Field(default="cosine_distance")
    probes: int = Field(default=10)  # default 10
    # hybrid_search_settings: Optional[HybridSearchSettings]
    ef_search: int = Field(default=40)  # default 40


class GraphSearchSettings(BaseModel):
    pass


class PerformSearchBase(BaseModel):
    query: str
    vector_search_settings: VectorSearchSettings = Field(default=VectorSearchSettings())
    graph_search_settings: GraphSearchSettings = Field(default=GraphSearchSettings())
    dataset_ids: List[str]  # since UUID is not serializable for celery
    # kg_search_settings: KGSearchSettings


class IWorkspacePerformSearchRequest(PerformSearchBase):
    perform_aggregate_search: bool = Field(
        default=True,
        description="Perform aggregate search over all datasets if true, perform dataset by dataset search if false.",
    )
    perform_graph_search: bool = Field(default=False)
    calculate_metrics: bool = False
    # perform search for each dataset individually
    # perform_dataset_search: bool = Field(default=False)


class SearchResult(BaseModel):
    text: str
    dataset_id: UUID
    search_score: float
    file_id: Optional[UUID] = None


class R2RVectorSearchResult(SearchResult):
    unstructured_page_number: Optional[int]
    chunk_order: int
    chunk_id: UUID


class BaseChunkMetadata(BaseModel):
    match_type: str
    base64: Optional[str] = None
    chunk_id: Optional[UUID]
    table_html: Optional[str] = None


class ImageSearchResult(SearchResult):
    file_path: str
    description: Optional[str]
    mimetype: str
    chunk_metadata: Optional[BaseChunkMetadata] = None


class R2RGraphSearchRelationshipResult(BaseModel):
    subject: str
    predicate: str
    object: str
    score: float
    description: Optional[str] = None


class IWorkspaceRagEvalData(BaseModel):
    ndcg_at_k: float
    precision_at_k: float
    latency: str


# test = IWorkspacePerformSearchRequest(
#     query="dafskhfasjf",
#     vector_search_settings=VectorSearchSettings(
#         search_limit=10, search_index_type="cosine_distance", probes=3, ef_search=3
#     ),
#     dataset_ids=[],
#     perform_aggregate_search=True,
# )
# perform_search_base = PerformSearchBase(**test.model_dump())

# print(test.vector_search_settings.model_dump())
# print(perform_search_base.model_dump())

# print(VectorSearchSettings())
