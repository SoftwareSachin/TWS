from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class FileStatusEnum(str, Enum):
    Uploading = "Uploading"
    Uploaded = "Uploaded"
    Stopped = "Stopped"
    Failed = "Failed"
    Processing = "Processing"


class IFileUploadRead(BaseModel):
    filename: str
    mimetype: str
    size: int
    status: FileStatusEnum
    id: UUID  # File ID

    class Config:
        from_attributes = True


class IFileUploadFailed(IFileUploadRead):
    id: Optional[UUID] = None
    status: FileStatusEnum = FileStatusEnum.Failed
    error: str
