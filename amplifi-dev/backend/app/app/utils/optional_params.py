from typing import Optional

from fastapi import Query
from fastapi_pagination.bases import AbstractParams, RawParams
from pydantic import BaseModel


class OptionalParams(BaseModel, AbstractParams):
    page: Optional[int] = Query(None, ge=1, description="Page number")
    size: Optional[int] = Query(None, ge=1, le=100, description="Page size")

    def to_raw_params(self) -> RawParams:
        if self.page is not None and self.size is not None:
            return RawParams(
                limit=self.size,
                offset=self.size * (self.page - 1),
            )
        return RawParams(limit=None, offset=None)
