from enum import Enum

from pydantic import BaseModel


class ConnectorStatus(str, Enum):
    success = "success"
    failed = "failed"


class ISourceConnectorStatusRead(BaseModel):
    status: ConnectorStatus
    message: str


class IDestinationConnectorStatusRead(BaseModel):
    status: ConnectorStatus
    message: str
