from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


# PgVector schema
class PgVector(BaseModel):
    host: str
    port: int = 5432
    database_name: str
    table_name: str
    username: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "host": "localhost",
                "port": 5432,
                "database_name": "vector_db",
                "table_name": "my_table",
                "username": "admin",
                "password": "admin123",
            }
        }


# Databricks schema
class Databricks(BaseModel):
    workspace_url: str
    token: str
    warehouse_id: str
    database_name: str
    table_name: str

    class Config:
        json_schema_extra = {
            "example": {
                "workspace_url": "https://databricks.com/workspace",
                "token": "dapi1234567890abcdef1234567890",
                "warehouse_id": "0123-456789-abcdef",
                "database_name": "my_db",
                "table_name": "my_table",
            }
        }


# Base schema for Destination
class DestinationBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_active: bool = False
    active_workflows: Optional[int] = None


# Schema for creating a destination
class IDestinationCreate(DestinationBase):
    pg_vector: Optional[PgVector] = (
        None  # Optional PgVector for PostgreSQL-based destinations
    )
    databricks: Optional[Databricks] = (
        None  # Optional Databricks for Databricks-based destinations
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Destination",
                "description": "Destination for data storage",
                "is_active": True,
                "pg_vector": {
                    "host": "localhost",
                    "port": 5432,
                    "database_name": "vector_db",
                    "table_name": "my_table",
                    "username": "admin",
                    "password": "admin123",
                },
                "databricks": {
                    "workspace_url": "https://databricks.com/workspace",
                    "token": "dapi1234567890abcdef1234567890",
                    "Warehouse_id": "0123-456789-abcdef",
                    "database_name": "my_db",
                    "table_name": "my_table",
                },
            }
        }


# Schema for reading a destination (including ID)
class IDestinationRead(DestinationBase):
    id: UUID
    destination_type: str
    created_at: datetime
    pg_vector: Optional[PgVector] = None
    databricks: Optional[Databricks] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "sample_amplifi01",
                "description": "this is destination",
                "is_active": False,
                "destination_type": "pg_vector",
                "created_at": "2024-12-04T10:00:00Z",
                "pg_vector": {
                    "host": "amplifi01",
                    "port": 5432,
                    "database_name": "amplifi_db01",
                    "username": "amplifi1",
                    "password": "amplifi",
                },
                "databricks": None,
            }
        }


# Message schema
class Message(BaseModel):
    detail: str


# Meta schema for additional information like pagination
class Meta(BaseModel):
    pagination: Optional[dict] = None  # Optional meta data


# Schema for the delete response
class IDeleteResponseBase(BaseModel):
    message: Message
    meta: Meta
    data: Optional[IDestinationRead] = None  # Nullable field for deleted destination

    class Config:
        json_schema_extra = {
            "example": {
                "message": {"detail": "Destination deleted successfully"},
                "meta": {},
                "data": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "name": "Main Office",
                    "description": "Main office in New York",
                    "is_active": False,
                },
            }
        }


# Schema for the get response
class IGetResponseBase(BaseModel):
    message: Message
    meta: Meta
    data: Optional[IDestinationRead] = None  # Nullable field for destination details

    class Config:
        json_schema_extra = {
            "example": {
                "message": {"detail": "Destination retrieved successfully"},
                "meta": {},
                "data": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "name": "Main Office",
                    "description": "Main office in New York",
                    "is_active": True,
                },
            }
        }
