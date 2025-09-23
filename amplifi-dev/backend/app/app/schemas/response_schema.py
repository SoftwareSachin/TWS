from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from math import ceil
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from fastapi_pagination import Page, Params
from fastapi_pagination.bases import AbstractPage, AbstractParams
from pydantic import BaseModel, Field

DataType = TypeVar("DataType")
T = TypeVar("T")


class PageBase(Page[T], Generic[T]):
    previous_page: int | None = Field(
        default=None, description="Page number of the previous page"
    )
    next_page: int | None = Field(
        default=None, description="Page number of the next page"
    )


class IResponseBase(BaseModel, Generic[T]):
    message: str = ""
    meta: dict | Any | None = {}
    data: T | None = None


class IGetResponsePaginated(AbstractPage[T], Generic[T]):
    message: str | None = ""
    meta: dict = {}
    data: PageBase[T]

    __params_type__ = Params  # Set params related to Page

    @classmethod
    def create(
        cls,
        items: Sequence[T],
        total: int,
        params: AbstractParams,
    ) -> PageBase[T] | None:
        if params.size is not None and total is not None and params.size != 0:
            pages = ceil(total / params.size)
        else:
            pages = 0

        return cls(
            data=PageBase[T](
                items=items,
                page=params.page,
                size=params.size,
                total=total,
                pages=pages,
                next_page=params.page + 1 if params.page < pages else None,
                previous_page=params.page - 1 if params.page > 1 else None,
            )
        )


class IGetResponseBase(IResponseBase[DataType], Generic[DataType]):
    message: str | None = "Data got correctly"


class IPostResponseBase(IResponseBase[DataType], Generic[DataType]):
    message: str | None = "Data created correctly"


class IPutResponseBase(IResponseBase[DataType], Generic[DataType]):
    message: str | None = "Data updated correctly"


class IDeleteResponseBase(IResponseBase[DataType], Generic[DataType]):
    message: str | None = "Data deleted correctly"


def create_response(
    data: DataType,
    message: str | None = None,
    meta: dict | Any | None = {},
) -> (
    IResponseBase[DataType]
    | IGetResponsePaginated[DataType]
    | IGetResponseBase[DataType]
    | IPutResponseBase[DataType]
    | IDeleteResponseBase[DataType]
    | IPostResponseBase[DataType]
):
    if isinstance(data, IGetResponsePaginated):
        data.message = "Data paginated correctly" if message is None else message
        data.meta = meta
        return data
    if message is None:
        return {"data": data, "meta": meta}
    return {"data": data, "message": message, "meta": meta}


class IWorkspacePerformSearchRead(BaseModel):
    vector_search_results: List[str]
    # kg_search_results: [List[str]] | None = []


class IPostResponseBase_IWorkspacePerformSearchRead(BaseModel):
    message: Optional[str] = Field(default="")
    meta: Dict = Field(default={})
    data: IWorkspacePerformSearchRead


# response model for ingestion api
class IIngestFilesOperationRead(BaseModel):
    file_id: UUID
    filename: str
    ingestion_id: Optional[UUID] = None
    # task_id: UUID
    chunking_config_id: Optional[UUID] = None
    # embedding_config_id: Optional[UUID] = None
    status: str
    created_at: str
    finished_at: Optional[str] = None


class IngestionStatusEnum(str, Enum):
    Processing = "Processing"
    Failed = "Failed"
    Success = "Success"
    Not_Started = "Not_Started"


class IKnowledgeGraphStatus(str, Enum):
    """Status values for knowledge graph building process"""

    PENDING = "pending"
    BUILDING = "building"
    SUCCESS = "success"
    FAILED = "failed"


class IKnowledgeGraphStatusResponse(BaseModel):
    """Schema for knowledge graph status response"""

    dataset_id: UUID = Field(..., description="Dataset UUID")
    has_knowledge_graph: bool = Field(
        False, description="Whether dataset has a knowledge graph"
    )
    status: str = Field(..., description="Current status of the graph building process")
    description: Optional[str] = Field(None, description="Description of the graph")
    created_at: Optional[datetime] = Field(
        None, description="When the graph was created"
    )
    updated_at: Optional[datetime] = Field(
        None, description="When the graph was last updated"
    )
