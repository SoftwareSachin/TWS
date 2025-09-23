from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class AgentDeps(BaseModel):
    """Input schema for SQL chat app operations"""

    query: str
    llm_model: Optional[str] = "GPT4o"
    dataset_ids: list[UUID]


class SQLChatResponse(BaseModel):
    """Output schema for SQL chat app response"""

    generated_sql: str
    answer: str
    plotly_code: str
    plotly_figure: str


class SQLProcessResponse(BaseModel):
    """Output schema for SQL processing tool (without plotly)"""

    generated_sql: str
    answer: str
    table_data: Optional[List[Dict[str, Any]]]
    query: str
    csv_file_id: Optional[str] = None
    csv_file_name: Optional[str] = None


class PlotlyVisualizationInput(BaseModel):
    """Input schema for Plotly visualization generation"""

    query: str
    generated_sql: str
    table_data: List[Dict[str, Any]]
    llm_model: str
    dataset_id: Optional[UUID] = None


class PlotlyVisualizationResponse(BaseModel):
    """Output schema for Plotly visualization generation"""

    plotly_code: str
    plotly_figure: str
