# tools/schemas/web_search.py
from typing import List

from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    results: List[WebSearchResult]
