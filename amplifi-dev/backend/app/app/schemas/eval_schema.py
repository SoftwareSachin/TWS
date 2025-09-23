from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.search_schema import (
    ImageSearchResult,
    R2RGraphSearchRelationshipResult,
    R2RVectorSearchResult,
    SearchResult,
    VectorSearchSettings,
)


class SearchMetrics(BaseModel):
    precision: float = Field(..., description="Precision Score")
    ndcg_score: float = Field(..., description="NDCG Score")
    latency: float = Field(..., description="Search Retrieval Latency")


class SearchMetricsDataset(BaseModel):
    precision: float = Field(..., description="Precision Score")
    ndcg_score: float = Field(..., description="NDCG Score")


class RecallScores(BaseModel):
    recall: float = Field(..., description="Recall Score")


class Scores(BaseModel):
    precision_scores: SearchMetrics
    recall_scores: RecallScores


class EvalPrecisionRequest(BaseModel):
    query: str
    context: str | List[str]
    k: Optional[int] = 0

    @field_validator("k")
    def check_k(cls, value):
        if value < 0:
            raise ValueError("k must nonnegative")
        return value


class SearchResponseBase(BaseModel):
    query: str
    vector_search_results: List[Union[R2RVectorSearchResult, ImageSearchResult]]
    precision_scores: Optional[SearchMetrics] = Field(default=None)
    gpt_input_tokens: int = Field(default=-1)


class AggregateSearchEvalResult(SearchResponseBase):
    dataset_ids: List[UUID]


class DatasetSearchEvalResult(SearchResponseBase):
    dataset_id: UUID
    idx: Optional[int] = Field(default=None)
    true_answer: Optional[str] = Field(default=None)


class R2RGraphSearchEntityResult(BaseModel):
    # TODO: Implement when needed
    pass


class GraphSearchEvalResult(BaseModel):
    query: str
    graph_search_relationship_results: list[R2RGraphSearchRelationshipResult] = Field(
        default_factory=list
    )
    graph_search_entity_results: list[R2RGraphSearchEntityResult] = Field(
        default_factory=list
    )


class IWorkspaceSearchEvalResponse(BaseModel):
    aggregate_results: Optional[AggregateSearchEvalResult] = Field(default=None)
    dataset_results: Optional[List[DatasetSearchEvalResult]] = Field(default=None)
    graph_results: Optional[GraphSearchEvalResult] = Field(default=None)


class IWorkspaceRagEvalResponse(BaseModel):
    ndcg_at_k: float
    precision_at_k: float
    latency: str


class EvalProcessedQuestionsResponse(BaseModel):
    dataset_ids: List[UUID]
    workspace_id: UUID
    eval_task_id: UUID
    started_at: str


class RagEvalResult(BaseModel):
    query: str
    true_answer: str
    vector_search_results: List[R2RVectorSearchResult]


class DatasetEvalResults(BaseModel):
    avg_ndcg: float
    avg_precision: float
    avg_input_tokens: float
    avg_rag_latency_per_search: float
    total_time_taken_in_seconds: float
    rag_results: List[RagEvalResult]


class RagRequest(BaseModel):
    dataset_ids: List[str]
    vector_search_settings: VectorSearchSettings = Field(default=VectorSearchSettings())
    eval_file_id: str


class RagProcessedRequest(BaseModel):
    qa_pairs: List[Dict[str, str]]
    dataset_ids: List[str]
    vector_search_settings: VectorSearchSettings = Field(default=VectorSearchSettings())


class PerformRagEval(BaseModel):
    dataset_ids: List[str]
    vector_search_settings: VectorSearchSettings = Field(default=VectorSearchSettings())


class IWorkspaceEvalRequest(BaseModel):
    query: str
    contexts: List[SearchResult]


class IWorkspaceEvalResponse(BaseModel):
    query: str
    contexts: List[SearchResult]
    precision: float = Field(..., description="Precision Score")
    ndcg_score: float = Field(..., description="NDCG Score")
    in_tokens: int = Field(..., description="Input Tokens")
