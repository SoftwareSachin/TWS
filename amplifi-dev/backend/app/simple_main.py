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
from fastapi import HTTPException, UploadFile
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime, timezone
import tempfile
import shutil

# Simple in-memory data store for testing (in production, use real database)
simple_users_db: Dict[int, Dict] = {}  # Simple user store for basic CRUD
next_user_id = 1

# Mock data stores for organization-related endpoints
destinations_db: Dict[str, List[Dict]] = {}
workspaces_db: Dict[str, List[Dict]] = {}
workflows_db: Dict[str, List[Dict]] = {}
datasets_db: Dict[str, List[Dict]] = {}
chat_apps_db: Dict[str, List[Dict]] = {}
files_db: Dict[str, List[Dict]] = {}  # Store uploaded files by workspace_id
sources_db: Dict[str, List[Dict]] = {}  # Store sources by workspace_id

# Additional mock data stores for new endpoints
workspace_datasets_db: Dict[str, List[Dict]] = {}  # Store datasets by workspace_id
workspace_tools_db: Dict[str, List[Dict]] = {}  # Store tools by workspace_id
workspace_agents_db: Dict[str, List[Dict]] = {}  # Store agents by workspace_id
workspace_chat_apps_db: Dict[str, List[Dict]] = {}  # Store chat apps by workspace_id
global_tools_db: List[Dict] = []  # Global tools and MCP tools
users_db: List[Dict] = []  # Users database
search_results_db: Dict[str, List[Dict]] = {}  # Search results by workspace_id

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
    simple_users_db[next_user_id] = user_data
    next_user_id += 1
    return UserResponse(**user_data)

@app.get("/api/v1/users", response_model=List[UserResponse])
async def get_users():
    """Get all users"""
    return [UserResponse(**user) for user in simple_users_db.values()]

@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get a specific user"""
    if user_id not in simple_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**simple_users_db[user_id])

@app.put("/api/v1/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user_update: UserCreate):
    """Update a user"""
    if user_id not in simple_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    simple_users_db[user_id].update({
        "email": user_update.email,
        "name": user_update.name,
        "is_active": user_update.is_active
    })
    return UserResponse(**simple_users_db[user_id])

@app.delete("/api/v1/users/{user_id}")
async def delete_simple_user(user_id: int):
    """Delete a user"""
    if user_id not in simple_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    del simple_users_db[user_id]
    return {"message": "User deleted successfully"}

# ==================== API V1 ENDPOINTS ====================
# Organization-based endpoints that the frontend expects (v1 paths)

@app.get("/api/v1/organization/{organization_id}/workspace")
async def get_workspaces_v1(organization_id: str):
    """Get all workspaces for an organization (v1 endpoint)"""
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

@app.post("/api/v1/organization/{organization_id}/workspace") 
async def create_workspace_v1(organization_id: str, workspace_data: dict):
    """Create a new workspace for an organization (v1 endpoint)"""
    if organization_id not in workspaces_db:
        workspaces_db[organization_id] = []
    
    new_workspace = {
        "id": f"workspace-{len(workspaces_db[organization_id]) + 1}",
        "name": workspace_data.get("name", "New Workspace"),
        "description": workspace_data.get("description", ""),
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    workspaces_db[organization_id].append(new_workspace)
    return {"data": new_workspace, "message": "Workspace created successfully"}

@app.get("/api/v1/organization/{organization_id}/dataset")
async def get_datasets_v1(organization_id: str):
    """Get all datasets for an organization (v1 endpoint)"""
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

@app.get("/api/v1/organization/{organization_id}/destination")
async def get_destinations_v1(organization_id: str):
    """Get all destinations for an organization (v1 endpoint)"""
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

@app.get("/api/v1/organization/{organization_id}/workflow")
async def get_workflows_v1(organization_id: str):
    """Get all workflows for an organization (v1 endpoint)"""
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

@app.post("/v2/organization/{organization_id}/workspace")
async def create_workspace(organization_id: str, workspace_data: dict):
    """Create a new workspace for an organization"""
    if organization_id not in workspaces_db:
        workspaces_db[organization_id] = []
    
    new_workspace = {
        "id": f"workspace-{len(workspaces_db[organization_id]) + 1}",
        "name": workspace_data.get("name", "New Workspace"),
        "description": workspace_data.get("description", ""),
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    workspaces_db[organization_id].append(new_workspace)
    return {"data": new_workspace, "message": "Workspace created successfully"}

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

@app.post("/api/v2/workspace/{workspace_id}/file_upload")
async def upload_file(workspace_id: str, files: List[UploadFile]):
    """File upload endpoint for workspace"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    uploaded_files = []
    
    # Initialize workspace files if not exists
    if workspace_id not in files_db:
        files_db[workspace_id] = []
    
    for file in files:
        try:
            # Create a temporary file to save uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                # Copy uploaded file content to temp file
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
            
            # File record for response and storage
            file_record = {
                "id": f"file-{abs(hash(file.filename + str(datetime.now()))) % 10000}",
                "filename": file.filename,
                "mimetype": file.content_type or "application/octet-stream",
                "size": file.size or 0,
                "status": "Uploaded",
                "workspace_id": workspace_id,
                "upload_time": datetime.now(timezone.utc).isoformat(),
                "file_path": temp_file_path,
                "type": "uploaded",
                "ingestion_status": "pending"
            }
            
            # Store in files database
            files_db[workspace_id].append(file_record)
            uploaded_files.append(file_record)
            
        except Exception as e:
            # Handle individual file upload errors
            uploaded_files.append({
                "filename": file.filename,
                "mimetype": file.content_type or "application/octet-stream", 
                "size": file.size or 0,
                "status": "Failed",
                "error": str(e)
            })
    
    return {
        "data": uploaded_files,
        "message": "Files processed."
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

# ==================== MISSING FILE ENDPOINTS ====================
# These endpoints are called by the frontend for file management

@app.get("/api/v1/workspace/{workspace_id}/file")
async def get_workspace_files(
    workspace_id: str,
    order: Optional[str] = "ascendent",
    only_uploaded: Optional[bool] = True,
    page: int = 1,
    size: int = 25
):
    """Get files for a specific workspace"""
    if workspace_id not in files_db:
        files_db[workspace_id] = []
    
    workspace_files = files_db[workspace_id]
    
    # Filter by uploaded files if requested
    if only_uploaded:
        workspace_files = [f for f in workspace_files if f.get("type") == "uploaded"]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_files = workspace_files[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_files,
            "total": len(workspace_files),
            "page": page,
            "size": size,
            "pages": (len(workspace_files) + size - 1) // size if size > 0 else 1
        },
        "message": "Files retrieved successfully"
    }

@app.get("/api/v1/workspace/{workspace_id}/source")
async def get_workspace_sources(
    workspace_id: str,
    order: Optional[str] = "ascendent",
    page: int = 1,
    size: int = 25
):
    """Get sources for a specific workspace"""
    if workspace_id not in sources_db:
        sources_db[workspace_id] = [
            {
                "id": "source-1",
                "name": "Sample Data Source",
                "type": "file",
                "status": "connected",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id
            }
        ]
    
    workspace_sources = sources_db[workspace_id]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_sources = workspace_sources[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_sources,
            "total": len(workspace_sources),
            "page": page,
            "size": size,
            "pages": (len(workspace_sources) + size - 1) // size if size > 0 else 1
        },
        "message": "Sources retrieved successfully"
    }

@app.get("/api/v1/organization/{organization_id}/workspace/{workspace_id}")
async def get_specific_workspace(organization_id: str, workspace_id: str):
    """Get a specific workspace by ID"""
    # Find workspace in organization workspaces
    if organization_id in workspaces_db:
        for workspace in workspaces_db[organization_id]:
            if workspace["id"] == workspace_id:
                return {
                    "data": workspace,
                    "message": "Workspace retrieved successfully"
                }
    
    # If not found, create a default workspace
    default_workspace = {
        "id": workspace_id,
        "name": f"Workspace {workspace_id}",
        "description": "Auto-created workspace",
        "status": "active",
        "organization_id": organization_id,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    if organization_id not in workspaces_db:
        workspaces_db[organization_id] = []
    workspaces_db[organization_id].append(default_workspace)
    
    return {
        "data": default_workspace,
        "message": "Workspace retrieved successfully"
    }

# ==================== WORKSPACE DATASETS ENDPOINTS ====================

@app.get("/api/v1/workspace/{workspace_id}/dataset")
async def get_workspace_datasets(
    workspace_id: str,
    order: Optional[str] = "ascendent",
    page: int = 1,
    size: int = 50
):
    """Get datasets for a specific workspace"""
    if workspace_id not in workspace_datasets_db:
        workspace_datasets_db[workspace_id] = [
            {
                "id": "dataset-1",
                "name": "Sample Dataset",
                "description": "Sample data for testing workspace functionality",
                "status": "ready",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id,
                "size": "2.4MB",
                "records": 3500,
                "type": "text"
            }
        ]
    
    workspace_datasets = workspace_datasets_db[workspace_id]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_datasets = workspace_datasets[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_datasets,
            "total": len(workspace_datasets),
            "page": page,
            "size": size,
            "pages": (len(workspace_datasets) + size - 1) // size if size > 0 else 1
        },
        "message": "Datasets retrieved successfully"
    }

@app.post("/api/v1/workspace/{workspace_id}/dataset")
async def create_workspace_dataset(workspace_id: str, dataset_data: dict):
    """Create a new dataset for a workspace"""
    if workspace_id not in workspace_datasets_db:
        workspace_datasets_db[workspace_id] = []
    
    new_dataset = {
        "id": f"dataset-{len(workspace_datasets_db[workspace_id]) + 1}",
        "name": dataset_data.get("name", "New Dataset"),
        "description": dataset_data.get("description", ""),
        "status": "processing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "size": "0MB",
        "records": 0,
        "type": dataset_data.get("type", "text")
    }
    workspace_datasets_db[workspace_id].append(new_dataset)
    return {"data": new_dataset, "message": "Dataset created successfully"}

@app.get("/api/v1/workspace/{workspace_id}/dataset/{dataset_id}")
async def get_workspace_dataset_by_id(workspace_id: str, dataset_id: str):
    """Get a specific dataset by ID"""
    if workspace_id in workspace_datasets_db:
        for dataset in workspace_datasets_db[workspace_id]:
            if dataset["id"] == dataset_id:
                return {"data": dataset, "message": "Dataset retrieved successfully"}
    raise HTTPException(status_code=404, detail="Dataset not found")

@app.put("/api/v1/workspace/{workspace_id}/dataset/{dataset_id}")
async def update_workspace_dataset(workspace_id: str, dataset_id: str, dataset_data: dict):
    """Update a specific dataset"""
    if workspace_id in workspace_datasets_db:
        for dataset in workspace_datasets_db[workspace_id]:
            if dataset["id"] == dataset_id:
                dataset.update({
                    "name": dataset_data.get("name", dataset["name"]),
                    "description": dataset_data.get("description", dataset["description"]),
                    "type": dataset_data.get("type", dataset["type"])
                })
                return {"data": dataset, "message": "Dataset updated successfully"}
    raise HTTPException(status_code=404, detail="Dataset not found")

@app.delete("/api/v1/workspace/{workspace_id}/dataset/{dataset_id}")
async def delete_workspace_dataset(workspace_id: str, dataset_id: str):
    """Delete a specific dataset"""
    if workspace_id in workspace_datasets_db:
        workspace_datasets_db[workspace_id] = [
            ds for ds in workspace_datasets_db[workspace_id] if ds["id"] != dataset_id
        ]
        return {"message": "Dataset deleted successfully"}
    raise HTTPException(status_code=404, detail="Dataset not found")

@app.get("/api/v1/workspace/{workspace_id}/dataset/{dataset_id}/chunks")
async def get_dataset_chunks(workspace_id: str, dataset_id: str):
    """Get chunks for a specific dataset"""
    return {
        "data": [
            {
                "id": "chunk-1",
                "content": "Sample chunk content for testing...",
                "metadata": {"source": "sample.txt", "page": 1},
                "dataset_id": dataset_id
            }
        ],
        "message": "Dataset chunks retrieved successfully"
    }

@app.get("/api/v1/workspace/{workspace_id}/dataset/{dataset_id}/trainings")
async def get_dataset_trainings(workspace_id: str, dataset_id: str):
    """Get training details for a specific dataset"""
    return {
        "data": [
            {
                "id": "training-1",
                "status": "completed",
                "accuracy": 0.95,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_id": dataset_id
            }
        ],
        "message": "Dataset trainings retrieved successfully"
    }

# ==================== SEARCH ENDPOINTS ====================

@app.post("/api/v1/workspace/{workspace_id}/search")
async def workspace_search(workspace_id: str, search_data: dict):
    """Search within a workspace"""
    query = search_data.get("query", "")
    search_type = search_data.get("type", "semantic")
    
    # Mock search results
    mock_results = [
        {
            "id": "result-1",
            "title": f"Search result for '{query}'",
            "content": f"This is a sample search result for your query: {query}",
            "source": "sample-document.pdf",
            "score": 0.85,
            "workspace_id": workspace_id,
            "type": search_type
        },
        {
            "id": "result-2", 
            "title": f"Another result for '{query}'",
            "content": f"Another relevant result matching your search: {query}",
            "source": "another-document.txt",
            "score": 0.78,
            "workspace_id": workspace_id,
            "type": search_type
        }
    ]
    
    return {
        "data": {
            "results": mock_results,
            "total": len(mock_results),
            "query": query,
            "search_type": search_type
        },
        "message": "Search completed successfully"
    }

# ==================== TOOLS AND MCP ENDPOINTS ====================

@app.get("/api/v1/workspace/{workspace_id}/tool")
async def get_workspace_tools(workspace_id: str):
    """Get tools for a specific workspace"""
    if workspace_id not in workspace_tools_db:
        workspace_tools_db[workspace_id] = [
            {
                "id": "tool-1",
                "name": "Sample Tool",
                "description": "A sample tool for testing",
                "type": "function",
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id
            }
        ]
    
    return {
        "data": workspace_tools_db[workspace_id],
        "message": "Tools retrieved successfully"
    }

@app.post("/api/v1/workspace/{workspace_id}/tool")
async def create_workspace_tool(workspace_id: str, tool_data: dict):
    """Create a new tool for a workspace"""
    if workspace_id not in workspace_tools_db:
        workspace_tools_db[workspace_id] = []
    
    new_tool = {
        "id": f"tool-{len(workspace_tools_db[workspace_id]) + 1}",
        "name": tool_data.get("name", "New Tool"),
        "description": tool_data.get("description", ""),
        "type": tool_data.get("type", "function"),
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "config": tool_data.get("config", {})
    }
    workspace_tools_db[workspace_id].append(new_tool)
    return {"data": new_tool, "message": "Tool created successfully"}

@app.get("/api/v1/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}")
async def get_workspace_tool_by_id(workspace_id: str, workspace_tool_id: str):
    """Get a specific workspace tool by ID"""
    if workspace_id in workspace_tools_db:
        for tool in workspace_tools_db[workspace_id]:
            if tool["id"] == workspace_tool_id:
                return {"data": tool, "message": "Tool retrieved successfully"}
    raise HTTPException(status_code=404, detail="Tool not found")

@app.put("/api/v1/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}")
async def update_workspace_tool(workspace_id: str, workspace_tool_id: str, tool_data: dict):
    """Update a specific workspace tool"""
    if workspace_id in workspace_tools_db:
        for tool in workspace_tools_db[workspace_id]:
            if tool["id"] == workspace_tool_id:
                tool.update({
                    "name": tool_data.get("name", tool["name"]),
                    "description": tool_data.get("description", tool["description"]),
                    "type": tool_data.get("type", tool["type"]),
                    "config": tool_data.get("config", tool.get("config", {}))
                })
                return {"data": tool, "message": "Tool updated successfully"}
    raise HTTPException(status_code=404, detail="Tool not found")

@app.delete("/api/v1/workspace/{workspace_id}/workspace_tool/{tool_id}")
async def delete_workspace_tool(workspace_id: str, tool_id: str):
    """Delete a specific workspace tool"""
    if workspace_id in workspace_tools_db:
        workspace_tools_db[workspace_id] = [
            tool for tool in workspace_tools_db[workspace_id] if tool["id"] != tool_id
        ]
        return {"message": "Tool deleted successfully"}
    raise HTTPException(status_code=404, detail="Tool not found")

# Global tools and MCP endpoints
@app.get("/api/v1/tool")
async def get_global_tools(tool_kind: Optional[str] = None, mcp_type: Optional[str] = None):
    """Get global tools and MCP tools"""
    if not global_tools_db:
        global_tools_db.extend([
            {
                "id": "mcp-tool-1",
                "name": "Web Search MCP",
                "description": "MCP tool for web searching",
                "tool_kind": "mcp",
                "mcp_type": "external",
                "status": "available",
                "config_schema": {"api_key": "string", "endpoint": "string"}
            },
            {
                "id": "function-tool-1", 
                "name": "Data Processor",
                "description": "Function tool for data processing",
                "tool_kind": "function",
                "mcp_type": None,
                "status": "available",
                "config_schema": {"input_format": "string"}
            }
        ])
    
    filtered_tools = global_tools_db
    if tool_kind:
        filtered_tools = [t for t in filtered_tools if t.get("tool_kind") == tool_kind]
    if mcp_type:
        filtered_tools = [t for t in filtered_tools if t.get("mcp_type") == mcp_type]
    
    return {"data": filtered_tools, "message": "Tools retrieved successfully"}

@app.post("/api/v1/tool")
async def create_global_tool(tool_data: dict):
    """Create a new global tool or MCP"""
    new_tool = {
        "id": f"tool-{len(global_tools_db) + 1}",
        "name": tool_data.get("name", "New Tool"),
        "description": tool_data.get("description", ""),
        "tool_kind": tool_data.get("tool_kind", "function"),
        "mcp_type": tool_data.get("mcp_type"),
        "status": "available",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_schema": tool_data.get("config_schema", {})
    }
    global_tools_db.append(new_tool)
    return {"data": new_tool, "message": "Tool created successfully"}

@app.post("/api/v1/tool/validate-mcp-config")
async def validate_mcp_config(config_data: dict):
    """Validate MCP configuration"""
    # Mock validation
    return {
        "valid": True,
        "message": "MCP configuration is valid",
        "errors": []
    }

@app.delete("/api/v1/tool/{tool_id}")
async def delete_global_tool(tool_id: str):
    """Delete a global tool"""
    global global_tools_db
    global_tools_db = [tool for tool in global_tools_db if tool["id"] != tool_id]
    return {"message": "Tool deleted successfully"}

@app.put("/api/v1/tool/{tool_id}")
async def update_global_tool(tool_id: str, tool_data: dict):
    """Update a global tool"""
    for tool in global_tools_db:
        if tool["id"] == tool_id:
            tool.update({
                "name": tool_data.get("name", tool["name"]),
                "description": tool_data.get("description", tool["description"]),
                "tool_kind": tool_data.get("tool_kind", tool["tool_kind"]),
                "mcp_type": tool_data.get("mcp_type", tool.get("mcp_type")),
                "config_schema": tool_data.get("config_schema", tool.get("config_schema", {}))
            })
            return {"data": tool, "message": "Tool updated successfully"}
    raise HTTPException(status_code=404, detail="Tool not found")

# ==================== AGENTS ENDPOINTS ====================

@app.get("/api/v1/workspace/{workspace_id}/agent")
async def get_workspace_agents(workspace_id: str):
    """Get agents for a specific workspace"""
    if workspace_id not in workspace_agents_db:
        workspace_agents_db[workspace_id] = [
            {
                "id": "agent-1",
                "name": "Sample Agent",
                "description": "A sample AI agent for testing",
                "type": "chat",
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id,
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        ]
    
    return {
        "data": workspace_agents_db[workspace_id],
        "message": "Agents retrieved successfully"
    }

@app.post("/api/v1/workspace/{workspace_id}/agent")
async def create_workspace_agent(workspace_id: str, agent_data: dict):
    """Create a new agent for a workspace"""
    if workspace_id not in workspace_agents_db:
        workspace_agents_db[workspace_id] = []
    
    new_agent = {
        "id": f"agent-{len(workspace_agents_db[workspace_id]) + 1}",
        "name": agent_data.get("name", "New Agent"),
        "description": agent_data.get("description", ""),
        "type": agent_data.get("type", "chat"),
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "model": agent_data.get("model", "gpt-3.5-turbo"),
        "temperature": agent_data.get("temperature", 0.7),
        "system_prompt": agent_data.get("system_prompt", "")
    }
    workspace_agents_db[workspace_id].append(new_agent)
    return {"data": new_agent, "message": "Agent created successfully"}

@app.put("/api/v1/workspace/{workspace_id}/agent/{agent_id}")
async def update_workspace_agent(workspace_id: str, agent_id: str, agent_data: dict):
    """Update a specific workspace agent"""
    if workspace_id in workspace_agents_db:
        for agent in workspace_agents_db[workspace_id]:
            if agent["id"] == agent_id:
                agent.update({
                    "name": agent_data.get("name", agent["name"]),
                    "description": agent_data.get("description", agent["description"]),
                    "type": agent_data.get("type", agent["type"]),
                    "model": agent_data.get("model", agent["model"]),
                    "temperature": agent_data.get("temperature", agent["temperature"]),
                    "system_prompt": agent_data.get("system_prompt", agent.get("system_prompt", ""))
                })
                return {"data": agent, "message": "Agent updated successfully"}
    raise HTTPException(status_code=404, detail="Agent not found")

@app.delete("/api/v1/workspace/{workspace_id}/agent/{agent_id}")
async def delete_workspace_agent(workspace_id: str, agent_id: str):
    """Delete a specific workspace agent"""
    if workspace_id in workspace_agents_db:
        workspace_agents_db[workspace_id] = [
            agent for agent in workspace_agents_db[workspace_id] if agent["id"] != agent_id
        ]
        return {"message": "Agent deleted successfully"}
    raise HTTPException(status_code=404, detail="Agent not found")

# ==================== CHAT APP ENDPOINTS ====================

@app.get("/api/v1/workspace/{workspace_id}/chat_apps")
async def get_workspace_chat_apps(workspace_id: str):
    """Get chat apps for a specific workspace"""
    if workspace_id not in workspace_chat_apps_db:
        workspace_chat_apps_db[workspace_id] = [
            {
                "id": "chat-app-1",
                "name": "Workspace Chat Assistant",
                "description": "AI chat assistant for this workspace",
                "type": "unstructured_chat_app",
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id,
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        ]
    
    return {
        "data": workspace_chat_apps_db[workspace_id],
        "message": "Chat apps retrieved successfully"
    }

@app.post("/api/v1/workspace/{workspace_id}/chat_app")
async def create_workspace_chat_app(workspace_id: str, chat_app_data: dict):
    """Create a new chat app for a workspace"""
    if workspace_id not in workspace_chat_apps_db:
        workspace_chat_apps_db[workspace_id] = []
    
    new_chat_app = {
        "id": f"chat-app-{len(workspace_chat_apps_db[workspace_id]) + 1}",
        "name": chat_app_data.get("name", "New Chat App"),
        "description": chat_app_data.get("description", ""),
        "type": chat_app_data.get("type", "unstructured_chat_app"),
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "model": chat_app_data.get("model", "gpt-3.5-turbo"),
        "temperature": chat_app_data.get("temperature", 0.7)
    }
    workspace_chat_apps_db[workspace_id].append(new_chat_app)
    return {"data": new_chat_app, "message": "Chat app created successfully"}

@app.get("/api/v1/workspace/{workspace_id}/chat_app/{chat_app_id}")
async def get_workspace_chat_app_by_id(workspace_id: str, chat_app_id: str):
    """Get a specific chat app by ID"""
    if workspace_id in workspace_chat_apps_db:
        for chat_app in workspace_chat_apps_db[workspace_id]:
            if chat_app["id"] == chat_app_id:
                return {"data": chat_app, "message": "Chat app retrieved successfully"}
    raise HTTPException(status_code=404, detail="Chat app not found")

@app.put("/api/v1/workspace/{workspace_id}/chat_app/{chat_app_id}")
async def update_workspace_chat_app(workspace_id: str, chat_app_id: str, chat_app_data: dict):
    """Update a specific workspace chat app"""
    if workspace_id in workspace_chat_apps_db:
        for chat_app in workspace_chat_apps_db[workspace_id]:
            if chat_app["id"] == chat_app_id:
                chat_app.update({
                    "name": chat_app_data.get("name", chat_app["name"]),
                    "description": chat_app_data.get("description", chat_app["description"]),
                    "type": chat_app_data.get("type", chat_app["type"]),
                    "model": chat_app_data.get("model", chat_app["model"]),
                    "temperature": chat_app_data.get("temperature", chat_app["temperature"])
                })
                return {"data": chat_app, "message": "Chat app updated successfully"}
    raise HTTPException(status_code=404, detail="Chat app not found")

@app.delete("/api/v1/workspace/{workspace_id}/chat_app/{chat_app_id}")
async def delete_workspace_chat_app(workspace_id: str, chat_app_id: str):
    """Delete a specific workspace chat app"""
    if workspace_id in workspace_chat_apps_db:
        workspace_chat_apps_db[workspace_id] = [
            app for app in workspace_chat_apps_db[workspace_id] if app["id"] != chat_app_id
        ]
        return {"message": "Chat app deleted successfully"}
    raise HTTPException(status_code=404, detail="Chat app not found")

# ==================== USERS ENDPOINTS ====================

@app.get("/api/v1/user/list")
async def get_users_list(organization_id: Optional[str] = None, page: int = 1, size: int = 25):
    """Get users list"""
    if not users_db:
        users_db.extend([
            {
                "id": "user-1",
                "email": "admin@amplifi.com",
                "name": "Admin User",
                "role": "admin",
                "status": "active",
                "organization_id": "test-org-123",
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "user-2",
                "email": "developer@amplifi.com", 
                "name": "Developer User",
                "role": "developer",
                "status": "active",
                "organization_id": "test-org-123",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ])
    
    filtered_users = users_db
    if organization_id:
        filtered_users = [u for u in users_db if u.get("organization_id") == organization_id]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_users = filtered_users[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_users,
            "total": len(filtered_users),
            "page": page,
            "size": size,
            "pages": (len(filtered_users) + size - 1) // size if size > 0 else 1
        },
        "message": "Users retrieved successfully"
    }

@app.post("/api/v1/user/invite-user")
async def invite_user(user_data: dict):
    """Invite a new user"""
    new_user = {
        "id": f"user-{len(users_db) + 1}",
        "email": user_data.get("email", ""),
        "name": user_data.get("name", ""),
        "role": user_data.get("role", "user"),
        "status": "invited",
        "organization_id": user_data.get("organization_id", "test-org-123"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invited_at": datetime.now(timezone.utc).isoformat()
    }
    users_db.append(new_user)
    return {"data": new_user, "message": "User invitation sent successfully"}

@app.delete("/api/v1/user/{user_id}")
async def delete_user(user_id: str):
    """Delete a user"""
    global users_db
    users_db = [user for user in users_db if user["id"] != user_id]
    return {"message": "User deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)