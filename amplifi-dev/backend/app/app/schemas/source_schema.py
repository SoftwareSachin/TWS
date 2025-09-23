from enum import Enum
from typing import Any, Dict, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class IAzureStorageSource(BaseModel):
    source_type: str = "azure_storage"
    container_name: str
    sas_url: str

    class Config:
        from_attributes = True


class IAWSS3Source(BaseModel):
    source_type: str = "aws_s3"
    bucket_name: str
    access_id: str
    access_secret: str

    class Config:
        from_attributes = True


class IAzureFabricSource(BaseModel):
    source_type: str = "azure_fabric"
    container_name: str
    sas_url: str

    class Config:
        from_attributes = True


class IPGVectorSource(BaseModel):
    source_type: str = "pg_db"
    host: str
    port: int
    database_name: str
    username: str
    password: str

    class Config:
        from_attributes = True

    @field_validator("host", "database_name", "username", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v


class IGrooveSource(BaseModel):
    source_type: str = "groove_source"
    source_name: str
    groove_api_key: str
    # Auto-detection configuration
    auto_detection_enabled: Optional[bool] = False
    monitoring_frequency_minutes: Optional[int] = 30
    ticket_batch_size: Optional[int] = 10
    re_ingest_updated_tickets: Optional[bool] = False

    class Config:
        from_attributes = True

    @field_validator("groove_api_key", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v


# class IMySQLSource(BaseModel):
#     source_type: str = "mysql_db"
#     host: str
#     port: int
#     database_name: str
#     username: str
#     password: str

#     class Config:
#         from_attributes = True

#     @field_validator("host", "database_name", "username", mode="before")
#     @classmethod
#     def strip_whitespace(cls, v):
#         return v.strip() if isinstance(v, str) else v


class ISQLSource(BaseModel):
    source_type: Literal["pg_db", "mysql_db"]  # Enforces allowed values
    host: str
    port: int
    database_name: str
    username: str
    password: str

    # adding the SSL config
    ssl_mode: Optional[str] = "disabled"  # initial we have the SSL mode is disabled

    class Config:
        from_attributes = True

    @field_validator(
        "host",
        "database_name",
        "username",
        "ssl_mode",
        mode="before",
    )
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v


class ISourceUpdate(BaseModel):
    source_type: Optional[str] = None
    container_name: Optional[str] = None
    sas_url: Optional[str] = None
    bucket_name: Optional[str] = None
    access_id: Optional[str] = None
    access_secret: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    class Config:
        from_attributes = True


class ISourceResponse(BaseModel):
    message: str
    meta: Dict[str, Any]
    data: Dict[str, Any]

    @classmethod
    def from_orm(cls, obj: Any) -> "ISourceResponse":
        return cls(message="Data updated correctly", meta={}, data=obj)


class AzureStorageSourceResponse(BaseModel):
    source_id: UUID
    source_type: str = "azure_storage"
    container_name: str
    # sas_url: str

    class Config:
        from_attributes = True


class AWSS3SourceResponse(BaseModel):
    source_id: UUID
    source_type: str = "aws_s3"
    bucket_name: str
    # access_id: str

    class Config:
        from_attributes = True


class AzureFabricSourceResponse(BaseModel):
    source_id: UUID
    source_type: str = "azure_fabric"
    container_name: str
    # sas_url: str

    class Config:
        from_attributes = True


class PGVectorSourceResponse(BaseModel):
    source_id: UUID
    source_type: str = "pg_db"
    host: str
    database_name: str

    class Config:
        from_attributes = True


class MySQLSourceResponse(BaseModel):
    source_id: UUID
    source_type: str = "mysql_db"
    host: str
    database_name: str

    class Config:
        from_attributes = True


class GrooveSourceResponse(BaseModel):
    source_id: UUID
    source_type: str = "groove_source"
    source_name: str
    auto_detection_enabled: bool
    monitoring_frequency_minutes: int
    ticket_batch_size: int
    re_ingest_updated_tickets: bool

    class Config:
        from_attributes = True


class ISourceConnectorResponse(BaseModel):
    sources: Union[
        AzureStorageSourceResponse,
        AWSS3SourceResponse,
        AzureFabricSourceResponse,
        PGVectorSourceResponse,
        MySQLSourceResponse,  # Add for the mysql
        GrooveSourceResponse,
    ]

    class Config:
        from_attributes = True


class IAutoDetectionConfig(BaseModel):
    """Schema for auto-detection configuration request."""

    enabled: bool = Field(..., description="Whether to enable auto-detection")
    frequency_minutes: int = Field(
        default=5,
        ge=2,
        le=1440,
        description="Monitoring frequency in minutes (2-1440 minutes, default 5)",
    )
    re_ingest_updated_blobs: bool = Field(
        default=False, description="Whether to re-ingest updated blobs or just log them"
    )

    @field_validator("frequency_minutes")
    @classmethod
    def validate_frequency(cls, v):
        if v < 2:
            raise ValueError("Frequency must be at least 5 minutes")
        if v > 1440:  # 24 hours
            raise ValueError("Frequency cannot exceed 1440 minutes (24 hours)")
        return v


class IStorageAccountInfo(BaseModel):
    """Schema for storage account information."""

    storage_account_name: str
    container_name: str
    sas_url_configured: bool


class IFileStatistics(BaseModel):
    """Schema for file statistics."""

    total_files: int
    status_breakdown: Dict[str, int]
    files_last_24_hours: int
    successful_files: int
    failed_files: int
    processing_files: int


class MonitoringHealthStatusEnum(str, Enum):
    DISABLED = "disabled"
    PENDING_FIRST_SCAN = "pending_first_scan"
    OVERDUE = "overdue"
    DUE = "due"
    HEALTHY = "healthy"


class IMonitoringHealth(BaseModel):
    """Schema for monitoring health status."""

    status: MonitoringHealthStatusEnum = Field(
        ...,
        description="disabled, pending_first_scan, overdue, due, healthy",
    )
    message: str
    last_monitoring_age_minutes: Optional[float] = None
    minutes_overdue: Optional[float] = None
    next_monitoring_in_minutes: Optional[float] = None


class IAutoDetectionStatus(BaseModel):
    """Schema for auto-detection status response."""

    auto_detection_enabled: bool
    monitoring_frequency_minutes: int
    last_monitored: Optional[str] = None  # ISO format datetime
    detection_method: str = "polling"
    next_monitoring_time: Optional[str] = None  # ISO format datetime
    time_until_next_monitoring_minutes: Optional[float] = None
    re_ingest_updated_blobs: bool
    storage_account_info: Union[IStorageAccountInfo, Dict[str, str]]
    file_statistics: Union[IFileStatistics, Dict[str, str]]
    monitoring_health: IMonitoringHealth

    class Config:
        from_attributes = True
