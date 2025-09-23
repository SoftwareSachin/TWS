from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class IVannaTrainingCreate(BaseModel):
    dataset_id: UUID
    documentation: Optional[str] = None
    question_sql_pairs: Optional[List[Dict[str, Any]]] = None
    version_id: int = 1


class IVannaTrainingRead(BaseModel):
    id: UUID
    dataset_id: UUID
    documentation: Optional[str]
    question_sql_pairs: Optional[List[Dict[str, Any]]]
    version_id: int
    created_at: datetime

    class Config:
        orm_mode = True
