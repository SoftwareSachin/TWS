#!/usr/bin/env python3
"""
Simple FastAPI server for testing in Replit environment
"""
from fastapi import FastAPI
import os
import asyncpg
import asyncio

# Simple FastAPI app for testing - NO MIDDLEWARE OR AUTHENTICATION
app = FastAPI(title="Amplifi API - Development", version="1.0.0")

async def test_database():
    """Test database connectivity"""
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"status": "error", "message": "DATABASE_URL not found"}
        
        conn = await asyncpg.connect(database_url)
        result = await conn.fetchval("SELECT version()")
        await conn.close()
        return {"status": "connected", "version": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Amplifi API is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "amplifi-backend"}

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint"""
    db_status = await test_database()
    return {
        "status": "running",
        "version": "1.0.0",
        "environment": "development",
        "database": db_status
    }

@app.get("/api/v1/db-test")
async def database_test():
    """Test database connectivity"""
    return await test_database()

# Simple CRUD endpoints for testing functionality
from fastapi import HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

# Simple in-memory data store for testing (in production, use real database)
users_db: Dict[int, Dict] = {}
next_user_id = 1

class UserCreate(BaseModel):
    """User creation model"""
    email: str
    name: str
    is_active: bool = True

class UserResponse(BaseModel):
    """User response model"""
    id: int
    email: str
    name: str
    is_active: bool
    created_at: str

# Test PostgreSQL table creation with asyncpg
@app.on_event("startup")
async def startup_event():
    """Initialize database connection and create test table"""
    try:
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            conn = await asyncpg.connect(database_url)
            # Create a simple test table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS test_users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await conn.close()
            print("Database table created successfully")
    except Exception as e:
        print(f"Database setup error: {e}")
        # Continue with in-memory storage

# CRUD endpoints for testing
@app.post("/api/v1/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    """Create a new user"""
    global next_user_id
    user_data = {
        "id": next_user_id,
        "email": user.email,
        "name": user.name,
        "is_active": user.is_active,
        "created_at": datetime.utcnow().isoformat()
    }
    users_db[next_user_id] = user_data
    next_user_id += 1
    return UserResponse(**user_data)

@app.get("/api/v1/users", response_model=List[UserResponse])
async def get_users():
    """Get all users"""
    return [UserResponse(**user) for user in users_db.values()]

@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get a specific user"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**users_db[user_id])

@app.put("/api/v1/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user_update: UserCreate):
    """Update a user"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    users_db[user_id].update({
        "email": user_update.email,
        "name": user_update.name,
        "is_active": user_update.is_active
    })
    return UserResponse(**users_db[user_id])

@app.delete("/api/v1/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    del users_db[user_id]
    return {"message": "User deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)