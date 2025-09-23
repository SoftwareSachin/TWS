from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.schemas.file_schema import FileStatusEnum


class FileMetadataRead(BaseModel):
    id: UUID
    filename: str
    mimetype: str
    size: int
    status: Optional[FileStatusEnum] = None

    rows: Optional[int] = 0
    columns: Optional[int] = 0

    class Config:
        from_attributes = True


class PGTableMetadataRead(BaseModel):
    filename: str
    mimetype: str
    size: int
    rows: int
    columns: int
    id: UUID
    status: Optional[FileStatusEnum] = None

    class Config:
        from_attributes = True
