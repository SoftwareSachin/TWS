from typing import Any, Dict, Optional

from pydantic import BaseModel


class IEntitiesExtractionResponse(BaseModel):
    extracted_data: Optional[Dict[str, Any]] = None
