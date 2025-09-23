from typing import Dict, Optional

from pydantic import BaseModel
from sqlmodel import Field


class IIngestDatasetCreate(BaseModel):
    name: Optional[str] = None
    metadata: Dict = Field(
        default={}, description="Metadata associated with the dataset"
    )
