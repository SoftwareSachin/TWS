from typing import List
from uuid import UUID

from pydantic import BaseModel
from sqlmodel import Field


class IWorkspaceExtractTableRequest(BaseModel):
    file_ids: List[UUID] = Field(
        default=[],
        description="List of file ids that will be ingested. Ingest all in dataset if blank.",
    )


class IWorkspaceExtractTableResponse(BaseModel):
    file_names: List[str]
