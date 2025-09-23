from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.utils.partial import optional  # noqa: F401


# Enum for the provider field
class IEmbeddingProviderEnum(str, Enum):
    litellm = "litellm"
    ollama = "ollama"
    openai = "openai"


# EmbeddingConfig Base Model
class EmbeddingConfigBase(BaseModel):
    name: str
    is_active: bool = False
    provider: Optional[IEmbeddingProviderEnum] = IEmbeddingProviderEnum.litellm
    base_model: str
    base_dimension: int = 512
    batch_size: int = 128
    rerank_model: Optional[str] = None
    add_title_as_prefix: bool = False


# IEmbeddingConfigCreate model (First Configuration)
class IEmbeddingConfigCreate(EmbeddingConfigBase):
    pass


# IEmbeddingConfigRead model
class IEmbeddingConfigRead(EmbeddingConfigBase):
    id: UUID  # UUID field


# IEmbeddingConfigUpdate model
class IEmbeddingConfigUpdate(EmbeddingConfigBase):
    pass
