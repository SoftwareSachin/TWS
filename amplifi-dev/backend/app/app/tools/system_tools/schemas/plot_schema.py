from typing import Optional, Union

from pydantic import BaseModel
from pydantic_ai.messages import ToolReturn


class PlotMockInput(BaseModel):
    query: str
    llm_model: Optional[str] = "gpt-4o"
    csv_file_name: Optional[str] = None


class PlotMockOutput(BaseModel):
    plot_code: str
    plot_html: str
    error: Optional[str] = None  # Optional for error messaging


# Union type that can handle both PlotMockOutput and ToolReturn
PlotOutput = Union[PlotMockOutput, ToolReturn]
