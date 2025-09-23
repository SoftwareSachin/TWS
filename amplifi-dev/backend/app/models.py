#!/usr/bin/env python3
"""
Database models for Amplifi platform
"""
from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
import os
from datetime import datetime

class User(SQLModel, table=True):
    """User model for testing CRUD operations"""
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str
    is_active: bool = Field(default=True)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

class UserCreate(SQLModel):
    """User creation model"""
    email: str
    name: str
    is_active: bool = True

class UserResponse(SQLModel):
    """User response model"""
    id: int
    email: str
    name: str
    is_active: bool
    created_at: datetime

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

engine = create_engine(DATABASE_URL)

def create_tables():
    """Create database tables"""
    SQLModel.metadata.create_all(engine)

def get_session():
    """Get database session"""
    with Session(engine) as session:
        yield session