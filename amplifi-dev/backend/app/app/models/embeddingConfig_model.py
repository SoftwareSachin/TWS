from typing import Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel  # Assuming this is your base model
from app.models.dataset_model import (  # Assuming you have a Dataset model defined
    Dataset,
)


class EmbeddingConfig(BaseUUIDModel, table=True):  # Inherits from BaseUUIDModel
    __tablename__ = "embedding_config"

    dataset_id: UUID = Field(foreign_key="datasets.id")  # Foreign key to Dataset
    name: Optional[str] = Field(default=None, index=True)
    provider: str = Field(default="r2r")  # Default value for provider
    base_model: str = Field(..., description="Base model used for embedding")
    rerank_model: str = Field(..., description="Rerank model used for embedding")
    base_dimension: Optional[int] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)
    add_title_as_prefix: bool = Field(default=False)
    is_active: bool = Field(default=True)

    # Relationship back to Dataset
    dataset: "Dataset" = Relationship(back_populates="embedding_configs")
