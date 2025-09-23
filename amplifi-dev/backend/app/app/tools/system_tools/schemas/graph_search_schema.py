from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class GraphSearchInput(BaseModel):
    """Graph search request input"""

    query: str = Field(..., description="Natural language query to search in the graph")
    dataset_ids: List[UUID] = Field(..., description="List of dataset IDs to search in")
    limit: int = Field(
        default=20, ge=1, le=50, description="Maximum number of results to return"
    )


class GraphSearchResult(BaseModel):
    """Individual graph search result"""

    content: str = Field(..., description="Formatted content of the result")
    result_type: str = Field(
        ..., description="Type of result: entity, relationship, or general"
    )
    score: float = Field(..., description="Relevance score of the result")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the result"
    )


class GraphSearchOutput(BaseModel):
    """Graph search response output"""

    query: str = Field(..., description="Original search query")
    dataset_ids: List[UUID] = Field(..., description="Dataset IDs that were searched")
    cypher_query: Optional[str] = Field(
        None, description="Generated Cypher query (if LLM method was used)"
    )
    method: str = Field(
        ..., description="Search method used: llm_generated or fallback"
    )
    results: List[GraphSearchResult] = Field(..., description="List of search results")
    count: int = Field(..., description="Number of results returned")
    success: bool = Field(..., description="Whether the search was successful")
    error: Optional[str] = Field(None, description="Error message if search failed")
