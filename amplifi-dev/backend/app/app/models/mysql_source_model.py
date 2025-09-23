from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.models.base_uuid_model import BaseUUIDModel

if TYPE_CHECKING:
    from app.models.source_model import Source


class MySQLSource(BaseUUIDModel, table=True):
    __tablename__ = "mysql_sources"

    source_id: UUID = Field(foreign_key="sources.id", primary_key=True)
    host: str = Field(..., nullable=False)
    port: int = Field(..., nullable=False)
    database_name: str = Field(..., nullable=False)
    username: str = Field(..., nullable=False)  # Vault reference
    password: str = Field(..., nullable=False)  # Vault reference

    # SSL configuration fields
    ssl_mode: Optional[str] = Field(default=None, nullable=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    source: Optional["Source"] = Relationship(back_populates="source_mysql")
