#!/usr/bin/env python3
"""
Simple FastAPI server for testing in Replit environment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncpg
import asyncio

# Simple FastAPI app for testing - NO MIDDLEWARE OR AUTHENTICATION
app = FastAPI(title="Amplifi API - Development", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

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
from datetime import datetime, timezone

# Simple in-memory data store for testing (in production, use real database)
users_db: Dict[int, Dict] = {}
next_user_id = 1

# Mock data stores for organization-related endpoints
destinations_db: Dict[str, List[Dict]] = {}
workspaces_db: Dict[str, List[Dict]] = {}
workflows_db: Dict[str, List[Dict]] = {}
datasets_db: Dict[str, List[Dict]] = {}
chat_apps_db: Dict[str, List[Dict]] = {}

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
        "created_at": datetime.now(timezone.utc).isoformat()
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

# ==================== API V2 ENDPOINTS ====================
# Organization-based endpoints that the frontend expects

@app.get("/v2/organization/{organization_id}/destination")
async def get_destinations(organization_id: str):
    """Get all destinations for an organization"""
    if organization_id not in destinations_db:
        destinations_db[organization_id] = [
            {
                "id": "dest-1",
                "name": "Sample Destination",
                "type": "database",
                "status": "connected",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": {"host": "localhost", "port": 5432}
            }
        ]
    return {"data": destinations_db[organization_id], "message": "Destinations retrieved successfully"}

@app.post("/v2/organization/{organization_id}/destination")
async def create_destination(organization_id: str, destination_data: dict):
    """Create a new destination for an organization"""
    if organization_id not in destinations_db:
        destinations_db[organization_id] = []
    
    new_destination = {
        "id": f"dest-{len(destinations_db[organization_id]) + 1}",
        "name": destination_data.get("name", "New Destination"),
        "type": destination_data.get("type", "database"),
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": destination_data.get("config", {})
    }
    destinations_db[organization_id].append(new_destination)
    return {"data": new_destination, "message": "Destination created successfully"}

@app.get("/v2/organization/{organization_id}/destination/{destination_id}/connection_status")
async def get_destination_status(organization_id: str, destination_id: str):
    """Get connection status for a specific destination"""
    return {
        "status": "connected",
        "last_check": datetime.utcnow().isoformat(),
        "message": "Connection is healthy"
    }

@app.delete("/v2/organization/{organization_id}/destination/{destination_id}")
async def delete_destination(organization_id: str, destination_id: str):
    """Delete a specific destination"""
    if organization_id in destinations_db:
        destinations_db[organization_id] = [
            dest for dest in destinations_db[organization_id]
            if dest["id"] != destination_id
        ]
    return {"message": "Destination deleted successfully"}

@app.get("/v2/organization/{organization_id}/workspace")
async def get_workspaces(organization_id: str):
    """Get all workspaces for an organization"""
    if organization_id not in workspaces_db:
        workspaces_db[organization_id] = [
            {
                "id": "workspace-1",
                "name": "Main Workspace",
                "description": "Primary workspace for data analysis",
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ]
    return {"data": workspaces_db[organization_id], "message": "Workspaces retrieved successfully"}

@app.get("/v2/organization/{organization_id}/workflow")
async def get_workflows(organization_id: str):
    """Get all workflows for an organization"""
    if organization_id not in workflows_db:
        workflows_db[organization_id] = [
            {
                "id": "workflow-1",
                "name": "Data Processing Pipeline",
                "description": "Automated data processing and analysis",
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ]
    return {"data": workflows_db[organization_id], "message": "Workflows retrieved successfully"}

@app.get("/v2/organization/{organization_id}/dataset")
async def get_datasets(organization_id: str):
    """Get all datasets for an organization"""
    if organization_id not in datasets_db:
        datasets_db[organization_id] = [
            {
                "id": "dataset-1",
                "name": "Sample Dataset",
                "description": "Sample data for testing",
                "status": "ready",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "size": "1.2MB",
                "records": 1500
            }
        ]
    return {"data": datasets_db[organization_id], "message": "Datasets retrieved successfully"}

# Additional endpoints that might be needed
@app.get("/v2/organization/{organization_id}")
async def get_organization(organization_id: str):
    """Get organization details"""
    return {
        "id": organization_id,
        "name": "Test Organization",
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

@app.get("/v2/my_chat_app")
async def get_my_chat_apps(
    chat_app_type: Optional[str] = None,
    order: Optional[str] = "ascendent", 
    page: int = 1,
    size: int = 8
):
    """Get user's chat apps"""
    # Mock chat app data for development
    mock_chat_apps = [
        {
            "id": "chat-app-1",
            "name": "Customer Support Assistant",
            "description": "AI assistant for customer support queries",
            "chat_app_type": "unstructured_chat_app",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": "workspace-1",
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        },
        {
            "id": "chat-app-2", 
            "name": "Data Analysis Helper",
            "description": "AI assistant for data analysis tasks",
            "chat_app_type": "unstructured_chat_app",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": "workspace-1",
            "config": {
                "model": "gpt-4",
                "temperature": 0.5
            }
        }
    ]
    
    # Filter by chat_app_type if provided
    if chat_app_type:
        mock_chat_apps = [app for app in mock_chat_apps if app["chat_app_type"] == chat_app_type]
    
    # Simple pagination 
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_apps = mock_chat_apps[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_apps,
            "total": len(mock_chat_apps),
            "page": page,
            "size": size,
            "pages": (len(mock_chat_apps) + size - 1) // size
        },
        "message": "Chat apps retrieved successfully"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)