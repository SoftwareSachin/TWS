from enum import Enum
from typing import Any, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, NonNegativeInt

from app.schemas.search_schema import ImageSearchResult, R2RVectorSearchResult


class CustomGenerationPrompt(BaseModel):
    pass


class ChatModelEnum(str, Enum):
    GPT35 = "GPT35"
    GPT4o = "GPT4o"
    GPT41 = "GPT41"
    GPTo3 = "o3-mini"
    GPT5 = "GPT5"


class GenerationSettings(BaseModel):
    chat_model: ChatModelEnum = Field(
        default=ChatModelEnum.GPT35, description="Model for chat completion"
    )

    chat_model_kwargs: Optional[dict] = Field(
        default={}, description="kwargs passed to chat model"
    )
    custom_prompt: Optional[CustomGenerationPrompt] = Field(
        default=None, description="Custom prompt; to be defined"
    )


class IWorkspacePerformGenerationRequest(BaseModel):
    query: str
    context: Optional[List[str]] = Field(
        default=None, description="Contexts, if None runs query with no context"
    )
    generation_settings: GenerationSettings


class RagContext(BaseModel):
    text: str
    file: Optional[dict[str, Any]] = None
    page_numb: Optional[NonNegativeInt] = None
    download_url: Optional[str] = None
    chunk_id: Optional[UUID] = None


class RagContextWrapper(BaseModel):
    rag_contexts: list[RagContext] = None
    raw_results: list[Union[R2RVectorSearchResult, ImageSearchResult]] = None


class IFileContextChunk(BaseModel):
    text: str
    chunk_id: Optional[UUID] = None
    page_number: Optional[int] = None
    search_score: float
    match_type: Optional[str] = None
    table_html: Optional[str] = None


class IFileContextAggregation(BaseModel):
    file_id: UUID
    file_name: Optional[str] = None
    max_search_score: Optional[float] = None
    mimetype: Optional[str] = None
    dataset_id: Optional[UUID] = None
    download_url: Optional[str] = None
    texts: List[IFileContextChunk]


class IWorkspaceGenerationResponse(BaseModel):
    answer: str
    full_response: str
    contexts_found: Optional[RagContextWrapper] = None
    file_contexts_aggregated: Optional[List[IFileContextAggregation]] = None
    ssml: str = ""
    input_tokens: Optional[int] = Field(default=-1)
    output_tokens: Optional[int] = Field(default=-1)


class SqlChatAppRAGGenerationRead(BaseModel):
    chat_app_type: str = "sql_chat_app"
    sql_prompt: str
    generated_sql: str
    answer: str
    plotly_code: str
    plotly_figure: str
    # chats: List[dict] = []


class IChatAppResponse(BaseModel):
    chats: Union[SqlChatAppRAGGenerationRead, IWorkspaceGenerationResponse]

    class Config:
        from_attributes = True
