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
from fastapi import HTTPException, UploadFile, File
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime, timezone
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import uuid
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score
from collections import defaultdict
import networkx as nx
from typing import Tuple, Any

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

# CSV Power BI Report data stores
csv_files_db: Dict[str, Dict] = {}  # Store uploaded CSV files and their metadata
power_bi_reports_db: Dict[str, Dict] = {}  # Store generated Power BI reports

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

# CSV Power BI Models
class CSVUploadResponse(BaseModel):
    """CSV upload response model"""
    file_id: str
    filename: str
    columns: List[str]
    row_count: int
    file_size: int
    preview_data: List[Dict]
    column_types: Dict[str, str]

class ColumnSelectionRequest(BaseModel):
    """Column selection request model"""
    file_id: str
    selected_columns: List[str]
    chart_type: Optional[str] = "auto"
    x_axis: Optional[str] = None
    y_axis: Optional[List[str]] = None

class PowerBIReportResponse(BaseModel):
    """Power BI report response model"""
    report_id: str
    file_id: str
    report_name: str
    chart_data: Dict
    plotly_json: str
    created_at: str
    columns_used: List[str]

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

# ==================== CSV POWER BI ENDPOINTS ====================
# Endpoints for CSV upload, processing, and Power BI report generation

@app.post("/api/v1/csv/upload", response_model=CSVUploadResponse)
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload and analyze CSV file for Power BI report generation"""
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV with pandas
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Analyze data
        columns = df.columns.tolist()
        row_count = len(df)
        file_size = len(content)
        
        # Get column types
        column_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype.startswith('int') or dtype.startswith('float'):
                column_types[col] = 'numeric'
            elif dtype == 'object':
                # Check if it's a date
                try:
                    pd.to_datetime(df[col].head(), errors='raise')
                    column_types[col] = 'date'
                except:
                    column_types[col] = 'text'
            elif 'datetime' in dtype:
                column_types[col] = 'date'
            else:
                column_types[col] = 'other'
        
        # Get preview data (first 5 rows)
        preview_data = df.head(5).fillna("").to_dict('records')
        
        # Store file info
        csv_files_db[file_id] = {
            "file_id": file_id,
            "filename": file.filename,
            "dataframe": df,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "columns": columns,
            "row_count": row_count,
            "file_size": file_size,
            "column_types": column_types
        }
        
        return CSVUploadResponse(
            file_id=file_id,
            filename=file.filename,
            columns=columns,
            row_count=row_count,
            file_size=file_size,
            preview_data=preview_data,
            column_types=column_types
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.get("/api/v1/csv/{file_id}/columns")
async def get_csv_columns(file_id: str):
    """Get columns and metadata for a specific CSV file"""
    if file_id not in csv_files_db:
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    file_data = csv_files_db[file_id]
    df = file_data["dataframe"]
    
    # Enhanced column analysis
    column_analysis = {}
    for col in df.columns:
        col_data = df[col]
        analysis = {
            "name": col,
            "type": file_data["column_types"][col],
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique()),
            "sample_values": [str(val) for val in col_data.dropna().head(5).tolist()]
        }
        
        if file_data["column_types"][col] == 'numeric':
            try:
                min_val = col_data.min()
                max_val = col_data.max()
                mean_val = col_data.mean()
                
                analysis.update({
                    "min": float(min_val) if not pd.isna(min_val) else None,
                    "max": float(max_val) if not pd.isna(max_val) else None,
                    "mean": float(mean_val) if not pd.isna(mean_val) else None
                })
            except (ValueError, TypeError):
                # Skip numeric stats if conversion fails
                pass
        
        column_analysis[col] = analysis
    
    return {
        "file_id": file_id,
        "filename": file_data["filename"],
        "columns": column_analysis,
        "total_rows": file_data["row_count"]
    }

@app.post("/api/v1/csv/generate-report", response_model=PowerBIReportResponse)
async def generate_power_bi_report(request: ColumnSelectionRequest):
    """Generate Power BI style report from selected CSV columns"""
    if request.file_id not in csv_files_db:
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    try:
        file_data = csv_files_db[request.file_id]
        df = file_data["dataframe"]
        
        # Filter dataframe to selected columns
        selected_df = df[request.selected_columns].copy()
        
        # Auto-detect chart type if not specified
        chart_type = request.chart_type or "auto"
        
        if chart_type == "auto":
            numeric_cols = [col for col in request.selected_columns 
                          if file_data["column_types"][col] == 'numeric']
            text_cols = [col for col in request.selected_columns 
                        if file_data["column_types"][col] in ['text', 'date']]
            
            if len(numeric_cols) >= 2:
                chart_type = "scatter"
            elif len(numeric_cols) == 1 and len(text_cols) >= 1:
                chart_type = "bar"
            elif len(text_cols) == 1:
                chart_type = "pie"
            else:
                chart_type = "table"
        
        # Generate Plotly chart based on type
        fig = None
        
        if chart_type == "bar":
            x_col = request.x_axis or (request.selected_columns[0] if request.selected_columns else None)
            y_col = request.y_axis[0] if request.y_axis else (request.selected_columns[1] if len(request.selected_columns) > 1 else None)
            
            if x_col and y_col and x_col in selected_df.columns and y_col in selected_df.columns:
                # Group by x_col and sum y_col for bar chart
                grouped_data = selected_df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.bar(grouped_data, x=x_col, y=y_col, 
                           title=f"{y_col} by {x_col}")
        
        elif chart_type == "scatter":
            numeric_cols = [col for col in request.selected_columns 
                          if file_data["column_types"][col] == 'numeric']
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                fig = px.scatter(selected_df, x=x_col, y=y_col,
                               title=f"{y_col} vs {x_col}")
        
        elif chart_type == "pie":
            text_col = next((col for col in request.selected_columns 
                           if file_data["column_types"][col] == 'text'), None)
            if text_col:
                value_counts = selected_df[text_col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribution of {text_col}")
        
        elif chart_type == "line":
            x_col = request.x_axis or (request.selected_columns[0] if request.selected_columns else None)
            y_cols = request.y_axis or [col for col in request.selected_columns[1:] 
                                      if file_data["column_types"][col] == 'numeric']
            
            if x_col and y_cols:
                fig = go.Figure()
                for y_col in y_cols:
                    if y_col in selected_df.columns:
                        fig.add_trace(go.Scatter(x=selected_df[x_col], y=selected_df[y_col],
                                               mode='lines+markers', name=y_col))
                fig.update_layout(title=f"Trend Analysis: {', '.join(y_cols)} over {x_col}",
                                xaxis_title=x_col, yaxis_title="Values")
        
        # Default to table if no chart could be generated
        if fig is None:
            # Create a summary table
            summary_data = []
            for col in request.selected_columns:
                if file_data["column_types"][col] == 'numeric':
                    summary_data.append({
                        'Column': col,
                        'Type': 'Numeric',
                        'Count': len(selected_df[col].dropna()),
                        'Mean': round(selected_df[col].mean(), 2) if not pd.isna(selected_df[col].mean()) else 'N/A',
                        'Min': selected_df[col].min() if not pd.isna(selected_df[col].min()) else 'N/A',
                        'Max': selected_df[col].max() if not pd.isna(selected_df[col].max()) else 'N/A'
                    })
                else:
                    summary_data.append({
                        'Column': col,
                        'Type': file_data["column_types"][col].title(),
                        'Count': len(selected_df[col].dropna()),
                        'Unique Values': selected_df[col].nunique(),
                        'Most Common': selected_df[col].mode().iloc[0] if len(selected_df[col].mode()) > 0 else 'N/A'
                    })
            
            # Create table visualization
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(summary_data[0].keys()) if summary_data else ['Column', 'Info'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[[row[col] for row in summary_data] for col in summary_data[0].keys()] if summary_data else [['No data'], ['N/A']],
                          fill_color='lavender',
                          align='left'))
            ])
            fig.update_layout(title="Data Summary Table")
        
        # Generate report
        report_id = str(uuid.uuid4())
        plotly_json = fig.to_json() if fig else "{}"
        
        report_data = {
            "report_id": report_id,
            "file_id": request.file_id,
            "report_name": f"Report for {file_data['filename']}",
            "chart_data": json.loads(plotly_json),
            "plotly_json": plotly_json,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "columns_used": request.selected_columns,
            "chart_type": chart_type
        }
        
        # Store report
        power_bi_reports_db[report_id] = report_data
        
        return PowerBIReportResponse(**report_data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating report: {str(e)}")

@app.get("/api/v1/csv/reports")
async def get_all_reports():
    """Get all generated Power BI reports"""
    return {
        "data": list(power_bi_reports_db.values()),
        "message": "Reports retrieved successfully"
    }

@app.get("/api/v1/csv/reports/{report_id}")
async def get_report(report_id: str):
    """Get specific Power BI report by ID"""
    if report_id not in power_bi_reports_db:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return power_bi_reports_db[report_id]

@app.delete("/api/v1/csv/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a specific Power BI report"""
    if report_id not in power_bi_reports_db:
        raise HTTPException(status_code=404, detail="Report not found")
    
    del power_bi_reports_db[report_id]
    return {"message": "Report deleted successfully"}

@app.delete("/api/v1/csv/{file_id}")
async def delete_csv_file(file_id: str):
    """Delete CSV file and associated reports"""
    if file_id not in csv_files_db:
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    # Delete associated reports
    reports_to_delete = [report_id for report_id, report in power_bi_reports_db.items() 
                        if report["file_id"] == file_id]
    for report_id in reports_to_delete:
        del power_bi_reports_db[report_id]
    
    # Delete CSV file
    del csv_files_db[file_id]
    
    return {"message": "CSV file and associated reports deleted successfully"}

# ==================== ENHANCED MULTI-CSV POWER BI ENDPOINTS ====================
# Professional PowerBI features with multi-CSV support and relationship detection

class MultiCSVUploadResponse(BaseModel):
    """Multi-CSV upload response model"""
    file_id: str
    filename: str
    columns: List[str]
    row_count: int
    file_size: int
    preview_data: List[Dict]
    column_types: Dict[str, str]
    potential_keys: List[str]  # Columns that might be keys/IDs

class RelationshipDetectionRequest(BaseModel):
    """Request model for relationship detection"""
    file_ids: List[str]

class RelationshipData(BaseModel):
    """Individual relationship data"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str
    confidence: float
    match_count: int

class RelationshipDetectionResponse(BaseModel):
    """Response model for relationship detection"""
    relationships: List[RelationshipData]
    graph_data: Dict
    total_tables: int
    total_relationships: int

class QueryBasedReportRequest(BaseModel):
    """Request model for query-based report generation"""
    file_ids: List[str]
    query: str
    report_type: Optional[str] = "auto"

@app.post("/api/v1/csv/upload-multiple", response_model=MultiCSVUploadResponse)
async def upload_csv_file_multiple(file: UploadFile = File(...)):
    """Enhanced CSV upload with relationship detection capabilities"""
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV with pandas
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Analyze data
        columns = df.columns.tolist()
        row_count = len(df)
        file_size = len(content)
        
        # Enhanced column type detection
        column_types = {}
        potential_keys = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Enhanced type detection
            if dtype.startswith('int') or dtype.startswith('float'):
                column_types[col] = 'numeric'
                # Check if it could be a key (unique values, ID patterns)
                if (col.lower().endswith('id') or col.lower().endswith('key') or 
                    'id' in col.lower() or df[col].nunique() == len(df)):
                    potential_keys.append(col)
            elif dtype == 'object':
                # Check if it's a date
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    column_types[col] = 'date'
                except:
                    column_types[col] = 'text'
                    # Check if text could be a foreign key
                    if (col.lower().endswith('id') or col.lower().endswith('key') or 
                        'id' in col.lower()):
                        potential_keys.append(col)
            elif 'datetime' in dtype:
                column_types[col] = 'date'
            else:
                column_types[col] = 'other'
        
        # Get preview data (first 5 rows)
        preview_data = df.head(5).fillna("").to_dict('records')
        
        # Store enhanced file info
        csv_files_db[file_id] = {
            "file_id": file_id,
            "filename": file.filename,
            "dataframe": df,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "columns": columns,
            "row_count": row_count,
            "file_size": file_size,
            "column_types": column_types,
            "potential_keys": potential_keys,
            "data_sample": df.head(100).to_dict('records')  # Store more sample data
        }
        
        return MultiCSVUploadResponse(
            file_id=file_id,
            filename=file.filename,
            columns=columns,
            row_count=row_count,
            file_size=file_size,
            preview_data=preview_data,
            column_types=column_types,
            potential_keys=potential_keys
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.post("/api/v1/csv/detect-relationships", response_model=RelationshipDetectionResponse)
async def detect_relationships(request: RelationshipDetectionRequest):
    """Detect relationships between multiple CSV files"""
    if len(request.file_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 files required for relationship detection")
    
    # Validate all files exist
    for file_id in request.file_ids:
        if file_id not in csv_files_db:
            raise HTTPException(status_code=404, detail=f"CSV file {file_id} not found")
    
    try:
        relationships = []
        files_data = {fid: csv_files_db[fid] for fid in request.file_ids}
        
        # Compare each pair of files
        for i, file_id_1 in enumerate(request.file_ids):
            for file_id_2 in request.file_ids[i+1:]:
                file_1 = files_data[file_id_1]
                file_2 = files_data[file_id_2]
                df_1 = file_1["dataframe"]
                df_2 = file_2["dataframe"]
                
                # Check for potential relationships
                for col_1 in file_1["columns"]:
                    for col_2 in file_2["columns"]:
                        # Calculate relationship confidence
                        confidence = calculate_column_similarity(
                            df_1[col_1], df_2[col_2], col_1, col_2
                        )
                        
                        if confidence > 0.5:  # Minimum confidence threshold
                            # Determine relationship type
                            rel_type = determine_relationship_type(
                                df_1[col_1], df_2[col_2], col_1, col_2
                            )
                            
                            # Count matching values
                            matches = len(set(df_1[col_1].dropna()) & set(df_2[col_2].dropna()))
                            
                            relationships.append(RelationshipData(
                                from_table=file_id_1,
                                from_column=col_1,
                                to_table=file_id_2,
                                to_column=col_2,
                                relationship_type=rel_type,
                                confidence=confidence,
                                match_count=matches
                            ))
        
        # Sort relationships by confidence
        relationships.sort(key=lambda x: x.confidence, reverse=True)
        
        # Generate graph data for visualization
        graph_data = {
            "nodes": [
                {
                    "id": fid,
                    "label": files_data[fid]["filename"].replace('.csv', ''),
                    "columns": files_data[fid]["columns"][:10],  # First 10 columns
                    "total_columns": len(files_data[fid]["columns"]),
                    "row_count": files_data[fid]["row_count"]
                }
                for fid in request.file_ids
            ],
            "edges": [
                {
                    "from": rel.from_table,
                    "to": rel.to_table,
                    "label": f"{rel.from_column} → {rel.to_column}",
                    "confidence": rel.confidence,
                    "type": rel.relationship_type
                }
                for rel in relationships
            ]
        }
        
        return RelationshipDetectionResponse(
            relationships=relationships,
            graph_data=graph_data,
            total_tables=len(request.file_ids),
            total_relationships=len(relationships)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting relationships: {str(e)}")

def calculate_column_similarity(col1: pd.Series, col2: pd.Series, name1: str, name2: str) -> float:
    """Calculate similarity score between two columns"""
    score = 0.0
    
    # Name similarity (high weight for ID/key patterns)
    if name1.lower() == name2.lower():
        score += 0.4
    elif any(keyword in name1.lower() and keyword in name2.lower() 
             for keyword in ['id', 'key', 'code', 'number']):
        score += 0.3
    elif name1.lower().replace('_', '').replace('-', '') == name2.lower().replace('_', '').replace('-', ''):
        score += 0.25
    
    # Data type similarity
    if str(col1.dtype) == str(col2.dtype):
        score += 0.2
    
    # Value overlap
    unique_1 = set(col1.dropna().astype(str))
    unique_2 = set(col2.dropna().astype(str))
    
    if unique_1 and unique_2:
        intersection = len(unique_1 & unique_2)
        union = len(unique_1 | unique_2)
        if union > 0:
            jaccard = intersection / union
            score += 0.4 * jaccard
    
    return min(score, 1.0)

def determine_relationship_type(col1: pd.Series, col2: pd.Series, name1: str, name2: str) -> str:
    """Determine the type of relationship between two columns"""
    unique_1 = col1.nunique()
    unique_2 = col2.nunique()
    total_1 = len(col1.dropna())
    total_2 = len(col2.dropna())
    
    # One-to-One
    if unique_1 == total_1 and unique_2 == total_2:
        return "one-to-one"
    
    # One-to-Many / Many-to-One
    if unique_1 == total_1 or unique_2 == total_2:
        return "one-to-many"
    
    # Many-to-Many (default)
    return "many-to-many"

@app.post("/api/v1/csv/generate-query-report")
async def generate_query_based_report(request: QueryBasedReportRequest):
    """Generate reports based on natural language queries"""
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="At least one file required")
    
    # Validate all files exist
    for file_id in request.file_ids:
        if file_id not in csv_files_db:
            raise HTTPException(status_code=404, detail=f"CSV file {file_id} not found")
    
    try:
        # Parse query to understand what user wants
        query_lower = request.query.lower()
        
        # Simple query parsing (in production, use NLP/LLM)
        result_data = []
        chart_config = {}
        
        if "top" in query_lower and any(num in query_lower for num in ["5", "10", "20"]):
            # Extract number
            import re
            numbers = re.findall(r'\d+', query_lower)
            top_n = int(numbers[0]) if numbers else 5
            
            # Find the main file (largest or first)
            main_file_id = max(request.file_ids, key=lambda x: csv_files_db[x]["row_count"])
            main_df = csv_files_db[main_file_id]["dataframe"]
            
            # Try to identify what to rank by
            if "user" in query_lower:
                # Look for user-related columns
                user_cols = [col for col in main_df.columns if 'user' in col.lower() or 'name' in col.lower()]
                if user_cols:
                    result_data = main_df[user_cols[0]].value_counts().head(top_n).to_dict()
                    chart_config = {
                        "type": "bar",
                        "title": f"Top {top_n} {user_cols[0]}",
                        "data": [{"x": k, "y": v} for k, v in result_data.items()]
                    }
        
        elif "summary" in query_lower or "overview" in query_lower:
            # Generate data summary across all files
            summary = {}
            for file_id in request.file_ids:
                file_data = csv_files_db[file_id]
                df = file_data["dataframe"]
                summary[file_data["filename"]] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": len([col for col in df.columns if file_data["column_types"].get(col) == 'numeric']),
                    "text_columns": len([col for col in df.columns if file_data["column_types"].get(col) == 'text'])
                }
            result_data = summary
            chart_config = {
                "type": "summary",
                "title": "Dataset Overview",
                "data": summary
            }
        
        else:
            # Default: return basic statistics from first file
            main_file_id = request.file_ids[0]
            main_df = csv_files_db[main_file_id]["dataframe"]
            numeric_cols = [col for col in main_df.columns 
                           if csv_files_db[main_file_id]["column_types"].get(col) == 'numeric']
            
            if numeric_cols:
                col = numeric_cols[0]
                result_data = {
                    "mean": float(main_df[col].mean()),
                    "median": float(main_df[col].median()),
                    "min": float(main_df[col].min()),
                    "max": float(main_df[col].max())
                }
                chart_config = {
                    "type": "histogram",
                    "title": f"{col} Distribution",
                    "data": main_df[col].hist().to_dict()
                }
        
        # Generate report ID and store
        report_id = str(uuid.uuid4())
        report_data = {
            "report_id": report_id,
            "query": request.query,
            "file_ids": request.file_ids,
            "result_data": result_data,
            "chart_config": chart_config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "query-based"
        }
        
        power_bi_reports_db[report_id] = report_data
        
        return {
            "report_id": report_id,
            "query": request.query,
            "result_data": result_data,
            "chart_config": chart_config,
            "message": "Query-based report generated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating query report: {str(e)}")

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

@app.get("/api/v2/organization/{organization_id}/destination")
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

@app.post("/api/v2/organization/{organization_id}/destination")
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

@app.get("/api/v2/organization/{organization_id}/destination/{destination_id}/connection_status")
async def get_destination_status(organization_id: str, destination_id: str):
    """Get connection status for a specific destination"""
    return {
        "status": "connected",
        "last_check": datetime.utcnow().isoformat(),
        "message": "Connection is healthy"
    }

@app.delete("/api/v2/organization/{organization_id}/destination/{destination_id}")
async def delete_destination(organization_id: str, destination_id: str):
    """Delete a specific destination"""
    if organization_id in destinations_db:
        destinations_db[organization_id] = [
            dest for dest in destinations_db[organization_id]
            if dest["id"] != destination_id
        ]
    return {"message": "Destination deleted successfully"}

@app.get("/api/v2/organization/{organization_id}/workspace")
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

@app.post("/api/v2/organization/{organization_id}/workspace")
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

@app.get("/api/v2/organization/{organization_id}/workflow")
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

@app.get("/api/v2/organization/{organization_id}/dataset")
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

@app.get("/api/v2/dataset/{dataset_id}/ingestion_status")
async def get_dataset_ingestion_status(dataset_id: str):
    """Get ingestion status for a specific dataset"""
    return {
        "dataset_id": dataset_id,
        "status": "completed",
        "progress": 100,
        "ingested_files": 15,
        "total_files": 15,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "ingestion_time": "2024-09-24T07:30:00Z",
        "errors": []
    }

# Additional endpoints that might be needed
@app.get("/api/v2/organization/{organization_id}")
async def get_organization(organization_id: str):
    """Get organization details"""
    return {
        "id": organization_id,
        "name": "Test Organization",
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/v2/workspace/{workspace_id}/file_upload")
async def upload_file(workspace_id: str, files: List[UploadFile] = File(...)):
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
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
                file_size = len(content)
            
            # File record for response and storage
            file_record = {
                "id": f"file-{abs(hash(file.filename + str(datetime.now()))) % 10000}",
                "filename": file.filename,
                "mimetype": file.content_type or "application/octet-stream",
                "size": file_size,
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
                "filename": file.filename if hasattr(file, 'filename') else "unknown",
                "mimetype": file.content_type if hasattr(file, 'content_type') else "application/octet-stream", 
                "size": 0,
                "status": "Failed",
                "error": str(e)
            })
    
    return {
        "data": uploaded_files,
        "message": "Files processed."
    }

@app.get("/api/v2/my_chat_app")
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

@app.get("/api/v1/workspace/{workspace_id}/source/{source_id}/files_metadata")
async def get_source_files_metadata(
    workspace_id: str,
    source_id: str,
    order_by: Optional[str] = "id",
    order: Optional[str] = "ascendent",
    page: int = 1,
    size: int = 25
):
    """Get files metadata for a specific source"""
    if source_id == "undefined":
        raise HTTPException(status_code=404, detail="Source ID not provided")
    
    # Mock files metadata for the source
    mock_files = [
        {
            "id": f"file-meta-{i}",
            "filename": f"document_{i}.pdf",
            "size": f"{i*100}KB",
            "type": "application/pdf",
            "source_id": source_id,
            "workspace_id": workspace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "processed"
        } for i in range(1, 6)
    ]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_files = mock_files[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_files,
            "total": len(mock_files),
            "page": page,
            "size": size,
            "pages": (len(mock_files) + size - 1) // size if size > 0 else 1
        },
        "message": "Source files metadata retrieved successfully"
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

# ==================== API V2 WORKSPACE ENDPOINTS ====================
# Additional v2 endpoints that the frontend expects

@app.get("/api/v2/workspace/{workspace_id}/chat_apps")
async def get_workspace_chat_apps_v2(
    workspace_id: str,
    order: Optional[str] = "ascendent",
    page: int = 1,
    size: int = 50
):
    """Get chat apps for a specific workspace (v2 endpoint)"""
    if workspace_id not in workspace_chat_apps_db:
        workspace_chat_apps_db[workspace_id] = [
            {
                "id": "chat-app-1",
                "name": "Workspace Chat Assistant",
                "description": "AI chat assistant for this workspace",
                "chat_app_type": "unstructured_chat_app",
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id,
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "config": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            },
            {
                "id": "chat-app-2",
                "name": "Code Review Assistant",
                "description": "AI assistant specialized in code review and debugging",
                "chat_app_type": "unstructured_chat_app",
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id": workspace_id,
                "model": "gpt-4",
                "temperature": 0.3,
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "max_tokens": 4096
                }
            }
        ]
    
    workspace_chat_apps = workspace_chat_apps_db[workspace_id]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_apps = workspace_chat_apps[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_apps,
            "total": len(workspace_chat_apps),
            "page": page,
            "size": size,
            "pages": (len(workspace_chat_apps) + size - 1) // size if size > 0 else 1
        },
        "message": "Chat apps retrieved successfully"
    }

@app.get("/api/v1/organization/{organization_id}/workspace/{workspace_id}/get_users")
async def get_organization_workspace_users(
    organization_id: str,
    workspace_id: str,
    page: int = 1,
    size: int = 10
):
    """Get users for a specific organization workspace"""
    # Return mock users for the workspace
    workspace_users = [
        {
            "id": "user-1",
            "email": "john.doe@example.com",
            "name": "John Doe",
            "role": "admin",
            "status": "active",
            "workspace_id": workspace_id,
            "organization_id": organization_id,
            "joined_at": datetime.now(timezone.utc).isoformat(),
            "last_login": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": "user-2",
            "email": "jane.smith@example.com",
            "name": "Jane Smith",
            "role": "developer",
            "status": "active",
            "workspace_id": workspace_id,
            "organization_id": organization_id,
            "joined_at": datetime.now(timezone.utc).isoformat(),
            "last_login": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_users = workspace_users[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_users,
            "total": len(workspace_users),
            "page": page,
            "size": size,
            "pages": (len(workspace_users) + size - 1) // size if size > 0 else 1
        },
        "message": "Organization workspace users retrieved successfully"
    }

@app.get("/api/v2/workspace/{workspace_id}/users")
async def get_workspace_users_v2(
    workspace_id: str,
    order: Optional[str] = "ascendent",
    page: int = 1,
    size: int = 50
):
    """Get users for a specific workspace (v2 endpoint)"""
    # Filter users for this workspace or create mock data
    workspace_users = [
        {
            "id": "user-1",
            "email": "john.doe@example.com",
            "name": "John Doe",
            "role": "admin",
            "status": "active",
            "workspace_id": workspace_id,
            "joined_at": datetime.now(timezone.utc).isoformat(),
            "last_login": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": "user-2",
            "email": "jane.smith@example.com",
            "name": "Jane Smith",
            "role": "developer",
            "status": "active",
            "workspace_id": workspace_id,
            "joined_at": datetime.now(timezone.utc).isoformat(),
            "last_login": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": "user-3",
            "email": "mike.wilson@example.com",
            "name": "Mike Wilson",
            "role": "analyst",
            "status": "invited",
            "workspace_id": workspace_id,
            "joined_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None
        }
    ]
    
    # Simple pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_users = workspace_users[start_idx:end_idx]
    
    return {
        "data": {
            "items": paginated_users,
            "total": len(workspace_users),
            "page": page,
            "size": size,
            "pages": (len(workspace_users) + size - 1) // size if size > 0 else 1
        },
        "message": "Workspace users retrieved successfully"
    }

# ==================== ADVANCED POWERBI HELPER FUNCTIONS ====================

def calculate_column_similarity(col1: pd.Series, col2: pd.Series, name1: str, name2: str) -> float:
    """Calculate similarity between two columns for relationship detection"""
    try:
        # Name similarity (40% weight)
        name_sim = calculate_name_similarity(name1, name2)
        
        # Value overlap similarity (40% weight)
        value_sim = calculate_value_overlap(col1, col2)
        
        # Type compatibility (20% weight)
        type_sim = calculate_type_compatibility(col1, col2)
        
        # Weighted combination
        confidence = (name_sim * 0.4) + (value_sim * 0.4) + (type_sim * 0.2)
        return min(confidence, 1.0)
    
    except Exception:
        return 0.0

def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between column names"""
    name1_clean = re.sub(r'[^a-zA-Z0-9]', '', name1.lower())
    name2_clean = re.sub(r'[^a-zA-Z0-9]', '', name2.lower())
    
    # Exact match
    if name1_clean == name2_clean:
        return 1.0
    
    # Check for ID/key patterns
    id_patterns = ['id', 'key', 'code', 'ref']
    name1_is_id = any(pattern in name1_clean for pattern in id_patterns)
    name2_is_id = any(pattern in name2_clean for pattern in id_patterns)
    
    if name1_is_id and name2_is_id:
        # Check if base names are similar
        name1_base = re.sub(r'(id|key|code|ref)$', '', name1_clean)
        name2_base = re.sub(r'(id|key|code|ref)$', '', name2_clean)
        
        if name1_base and name2_base:
            if name1_base == name2_base:
                return 0.9
            elif name1_base in name2_base or name2_base in name1_base:
                return 0.7
    
    # Substring similarity
    if name1_clean in name2_clean or name2_clean in name1_clean:
        return 0.6
    
    return 0.0

def calculate_value_overlap(col1: pd.Series, col2: pd.Series) -> float:
    """Calculate value overlap between two columns"""
    try:
        # Remove nulls and get unique values
        vals1 = set(col1.dropna().astype(str).unique())
        vals2 = set(col2.dropna().astype(str).unique())
        
        if not vals1 or not vals2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(vals1 & vals2)
        union = len(vals1 | vals2)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # Boost score if there's significant overlap
        if intersection > min(len(vals1), len(vals2)) * 0.3:
            jaccard *= 1.2
        
        return min(jaccard, 1.0)
    
    except Exception:
        return 0.0

def calculate_type_compatibility(col1: pd.Series, col2: pd.Series) -> float:
    """Calculate type compatibility between columns"""
    try:
        dtype1 = str(col1.dtype)
        dtype2 = str(col2.dtype)
        
        # Same type
        if dtype1 == dtype2:
            return 1.0
        
        # Compatible numeric types
        numeric_types = ['int', 'float']
        if any(t in dtype1 for t in numeric_types) and any(t in dtype2 for t in numeric_types):
            return 0.8
        
        # Both are object types (could be strings/mixed)
        if dtype1 == 'object' and dtype2 == 'object':
            return 0.7
        
        # One numeric, one object (could be string IDs)
        if (any(t in dtype1 for t in numeric_types) and dtype2 == 'object') or \
           (any(t in dtype2 for t in numeric_types) and dtype1 == 'object'):
            return 0.5
        
        return 0.3
    
    except Exception:
        return 0.3

def determine_relationship_type(col1: pd.Series, col2: pd.Series, name1: str, name2: str) -> str:
    """Determine the type of relationship between two columns"""
    try:
        # Get unique counts
        unique1 = col1.nunique()
        unique2 = col2.nunique()
        total1 = len(col1.dropna())
        total2 = len(col2.dropna())
        
        # Calculate uniqueness ratios
        ratio1 = unique1 / total1 if total1 > 0 else 0
        ratio2 = unique2 / total2 if total2 > 0 else 0
        
        # One-to-One: both columns have unique values
        if ratio1 > 0.95 and ratio2 > 0.95:
            return "one-to-one"
        
        # One-to-Many: one column is mostly unique, other is not
        elif ratio1 > 0.95 and ratio2 < 0.8:
            return "one-to-many"
        elif ratio2 > 0.95 and ratio1 < 0.8:
            return "many-to-one"
        
        # Many-to-Many: both columns have duplicates
        else:
            return "many-to-many"
    
    except Exception:
        return "unknown"

def generate_professional_plotly_theme():
    """Generate professional Plotly theme consistent with the application"""
    return {
        'layout': {
            'paper_bgcolor': '#ffffff',
            'plot_bgcolor': '#fafafa',
            'font': {
                'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'size': 12,
                'color': '#374151'
            },
            'title': {
                'font': {
                    'size': 16,
                    'color': '#111827',
                    'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'gridcolor': '#e5e7eb',
                'linecolor': '#d1d5db',
                'tickcolor': '#9ca3af',
                'title_font': {'color': '#374151'},
                'tickfont': {'color': '#6b7280'}
            },
            'yaxis': {
                'gridcolor': '#e5e7eb',
                'linecolor': '#d1d5db',
                'tickcolor': '#9ca3af',
                'title_font': {'color': '#374151'},
                'tickfont': {'color': '#6b7280'}
            },
            'legend': {
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': '#e5e7eb',
                'borderwidth': 1
            },
            'margin': {'t': 60, 'r': 30, 'b': 50, 'l': 60}
        },
        'colorway': [
            '#374AF1',  # Primary blue
            '#64AC01',  # Green
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Light blue
            '#96CEB4',  # Light green
            '#FCEA2B',  # Yellow
            '#FF9FF3',  # Pink
            '#54A0FF',  # Sky blue
            '#FF9F43'   # Orange
        ]
    }

def create_professional_chart(chart_type: str, data: pd.DataFrame, config: Dict) -> go.Figure:
    """Create professional chart with consistent styling"""
    theme = generate_professional_plotly_theme()
    
    fig = None
    
    if chart_type == 'bar':
        fig = px.bar(
            data, 
            x=config.get('x'), 
            y=config.get('y'),
            color=config.get('color'),
            title=config.get('title', ''),
            color_discrete_sequence=theme['colorway']
        )
    
    elif chart_type == 'line':
        fig = px.line(
            data, 
            x=config.get('x'), 
            y=config.get('y'),
            color=config.get('color'),
            title=config.get('title', ''),
            color_discrete_sequence=theme['colorway']
        )
    
    elif chart_type == 'scatter':
        fig = px.scatter(
            data, 
            x=config.get('x'), 
            y=config.get('y'),
            color=config.get('color'),
            size=config.get('size'),
            title=config.get('title', ''),
            color_discrete_sequence=theme['colorway']
        )
    
    elif chart_type == 'pie':
        fig = px.pie(
            data, 
            values=config.get('values'), 
            names=config.get('names'),
            title=config.get('title', ''),
            color_discrete_sequence=theme['colorway']
        )
    
    elif chart_type == 'histogram':
        fig = px.histogram(
            data, 
            x=config.get('x'),
            color=config.get('color'),
            title=config.get('title', ''),
            color_discrete_sequence=theme['colorway']
        )
    
    elif chart_type == 'box':
        fig = px.box(
            data, 
            x=config.get('x'), 
            y=config.get('y'),
            color=config.get('color'),
            title=config.get('title', ''),
            color_discrete_sequence=theme['colorway']
        )
    
    elif chart_type == 'heatmap':
        if 'z' in config:
            fig = px.imshow(
                config['z'],
                title=config.get('title', ''),
                color_continuous_scale='Blues'
            )
    
    if fig:
        # Apply professional theme
        fig.update_layout(theme['layout'])
        
        # Add hover styling
        fig.update_traces(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter"
            )
        )
    
    return fig

def generate_entity_relationship_graph(files_data: Dict, relationships: List[Dict]) -> Dict:
    """Generate entity relationship graph data for visualization"""
    try:
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (tables)
        for file_id, file_data in files_data.items():
            G.add_node(file_id, 
                      label=file_data['filename'],
                      size=len(file_data['columns']),
                      type='table')
        
        # Add edges (relationships)
        for rel in relationships:
            G.add_edge(
                rel['from_table'], 
                rel['to_table'],
                weight=rel['confidence'],
                type=rel['relationship_type'],
                from_column=rel['from_column'],
                to_column=rel['to_column']
            )
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Convert to visualization format
        nodes = []
        edges = []
        
        for node_id, data in G.nodes(data=True):
            file_data = files_data[node_id]
            nodes.append({
                'id': node_id,
                'label': data['label'],
                'x': pos[node_id][0] * 300,
                'y': pos[node_id][1] * 300,
                'size': max(20, min(60, data['size'] * 2)),
                'color': '#374AF1',
                'columns': file_data['columns'],
                'row_count': file_data['row_count']
            })
        
        for edge in G.edges(data=True):
            edges.append({
                'from': edge[0],
                'to': edge[1],
                'weight': edge[2]['weight'],
                'type': edge[2]['type'],
                'from_column': edge[2]['from_column'],
                'to_column': edge[2]['to_column'],
                'color': '#64AC01' if edge[2]['weight'] > 0.8 else '#FF6B6B'
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_tables': len(nodes),
                'total_relationships': len(edges),
                'avg_confidence': sum(e['weight'] for e in edges) / len(edges) if edges else 0
            }
        }
    
    except Exception as e:
        return {
            'nodes': [],
            'edges': [],
            'stats': {'total_tables': 0, 'total_relationships': 0, 'avg_confidence': 0},
            'error': str(e)
        }

def process_natural_language_query(query: str, files_data: Dict) -> Dict:
    """Process natural language query to generate report specification"""
    try:
        query_lower = query.lower()
        
        # Extract key components
        result = {
            'type': 'unknown',
            'tables': [],
            'columns': [],
            'filters': [],
            'aggregations': [],
            'limit': None,
            'order': 'desc'
        }
        
        # Detect query type
        if any(word in query_lower for word in ['top', 'highest', 'largest', 'maximum']):
            result['type'] = 'top_n'
            result['order'] = 'desc'
        elif any(word in query_lower for word in ['bottom', 'lowest', 'smallest', 'minimum']):
            result['type'] = 'top_n'
            result['order'] = 'asc'
        elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            result['type'] = 'comparison'
        elif any(word in query_lower for word in ['trend', 'over time', 'timeline']):
            result['type'] = 'trend'
        elif any(word in query_lower for word in ['distribution', 'breakdown', 'by']):
            result['type'] = 'distribution'
        else:
            result['type'] = 'general'
        
        # Extract numbers (for top N queries)
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            result['limit'] = int(numbers[0])
        
        # Try to match table/column names
        for file_id, file_data in files_data.items():
            filename_base = file_data['filename'].lower().replace('.csv', '')
            if filename_base in query_lower:
                result['tables'].append(file_id)
            
            for column in file_data['columns']:
                if column.lower() in query_lower:
                    result['columns'].append({
                        'table': file_id,
                        'column': column,
                        'type': file_data['column_types'].get(column, 'unknown')
                    })
        
        # Generate suggested chart type
        if result['type'] == 'top_n':
            result['suggested_chart'] = 'bar'
        elif result['type'] == 'trend':
            result['suggested_chart'] = 'line'
        elif result['type'] == 'distribution':
            result['suggested_chart'] = 'pie'
        elif result['type'] == 'comparison':
            result['suggested_chart'] = 'bar'
        else:
            result['suggested_chart'] = 'table'
        
        return result
    
    except Exception as e:
        return {
            'type': 'error',
            'error': str(e),
            'suggested_chart': 'table'
        }

# ==================== SESSION-BASED CSV MANAGEMENT ====================

csv_sessions_db: Dict[str, Dict] = {}  # Store CSV sessions

class CSVSessionCreate(BaseModel):
    """CSV session creation model"""
    name: str
    description: Optional[str] = ""
    workspace_id: Optional[str] = "default"

class CSVSessionResponse(BaseModel):
    """CSV session response model"""
    session_id: str
    name: str
    description: str
    workspace_id: str
    created_at: str
    file_count: int
    total_rows: int

@app.post("/api/v1/csv/session", response_model=CSVSessionResponse)
async def create_csv_session(session: CSVSessionCreate):
    """Create a new CSV analysis session"""
    session_id = str(uuid.uuid4())
    
    session_data = {
        "session_id": session_id,
        "name": session.name,
        "description": session.description,
        "workspace_id": session.workspace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "relationships": [],
        "dashboards": []
    }
    
    csv_sessions_db[session_id] = session_data
    
    return CSVSessionResponse(
        session_id=session_id,
        name=session.name,
        description=session.description,
        workspace_id=session.workspace_id,
        created_at=session_data["created_at"],
        file_count=0,
        total_rows=0
    )

@app.post("/api/v1/csv/session/{session_id}/upload", response_model=MultiCSVUploadResponse)
async def upload_to_session(session_id: str, file: UploadFile = File(...)):
    """Upload CSV file to existing session"""
    if session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV with pandas
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Enhanced analysis
        columns = df.columns.tolist()
        row_count = len(df)
        file_size = len(content)
        
        column_types = {}
        potential_keys = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Enhanced type detection
            if dtype.startswith('int') or dtype.startswith('float'):
                column_types[col] = 'numeric'
                if (col.lower().endswith('id') or col.lower().endswith('key') or 
                    'id' in col.lower() or df[col].nunique() == len(df)):
                    potential_keys.append(col)
            elif dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    column_types[col] = 'date'
                except:
                    column_types[col] = 'text'
                    if (col.lower().endswith('id') or col.lower().endswith('key') or 
                        'id' in col.lower()):
                        potential_keys.append(col)
            elif 'datetime' in dtype:
                column_types[col] = 'date'
            else:
                column_types[col] = 'other'
        
        preview_data = df.head(5).fillna("").to_dict('records')
        
        # Store file data
        file_data = {
            "file_id": file_id,
            "filename": file.filename,
            "dataframe": df,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "columns": columns,
            "row_count": row_count,
            "file_size": file_size,
            "column_types": column_types,
            "potential_keys": potential_keys,
            "data_sample": df.head(100).to_dict('records')
        }
        
        csv_files_db[file_id] = file_data
        csv_sessions_db[session_id]["files"].append(file_id)
        
        return MultiCSVUploadResponse(
            file_id=file_id,
            filename=file.filename,
            columns=columns,
            row_count=row_count,
            file_size=file_size,
            preview_data=preview_data,
            column_types=column_types,
            potential_keys=potential_keys
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.get("/api/v1/csv/session/{session_id}")
async def get_session(session_id: str):
    """Get session details with all files and statistics"""
    if session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = csv_sessions_db[session_id]
    files_data = []
    total_rows = 0
    
    for file_id in session["files"]:
        if file_id in csv_files_db:
            file_data = csv_files_db[file_id]
            files_data.append({
                "file_id": file_id,
                "filename": file_data["filename"],
                "columns": file_data["columns"],
                "row_count": file_data["row_count"],
                "column_types": file_data["column_types"],
                "potential_keys": file_data["potential_keys"]
            })
            total_rows += file_data["row_count"]
    
    return {
        **session,
        "files_data": files_data,
        "file_count": len(files_data),
        "total_rows": total_rows,
        "relationship_count": len(session["relationships"])
    }

@app.post("/api/v1/csv/session/{session_id}/detect-relationships")
async def detect_session_relationships(session_id: str):
    """Detect relationships between all files in a session"""
    if session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = csv_sessions_db[session_id]
    file_ids = session["files"]
    
    if len(file_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 files required for relationship detection")
    
    # Validate all files exist
    for file_id in file_ids:
        if file_id not in csv_files_db:
            raise HTTPException(status_code=404, detail=f"CSV file {file_id} not found")
    
    try:
        relationships = []
        files_data = {fid: csv_files_db[fid] for fid in file_ids}
        
        # Compare each pair of files
        for i, file_id_1 in enumerate(file_ids):
            for file_id_2 in file_ids[i+1:]:
                file_1 = files_data[file_id_1]
                file_2 = files_data[file_id_2]
                df_1 = file_1["dataframe"]
                df_2 = file_2["dataframe"]
                
                # Check for potential relationships
                for col_1 in file_1["columns"]:
                    for col_2 in file_2["columns"]:
                        # Calculate relationship confidence
                        confidence = calculate_column_similarity(
                            df_1[col_1], df_2[col_2], col_1, col_2
                        )
                        
                        if confidence > 0.5:  # Minimum confidence threshold
                            # Determine relationship type
                            rel_type = determine_relationship_type(
                                df_1[col_1], df_2[col_2], col_1, col_2
                            )
                            
                            # Count matching values
                            matches = len(set(df_1[col_1].dropna()) & set(df_2[col_2].dropna()))
                            
                            relationships.append({
                                "from_table": file_id_1,
                                "from_column": col_1,
                                "to_table": file_id_2,
                                "to_column": col_2,
                                "relationship_type": rel_type,
                                "confidence": confidence,
                                "match_count": matches
                            })
        
        # Sort by confidence
        relationships.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Generate entity relationship graph
        graph_data = generate_entity_relationship_graph(files_data, relationships)
        
        # Store relationships in session
        csv_sessions_db[session_id]["relationships"] = relationships
        
        return {
            "relationships": relationships,
            "graph_data": graph_data,
            "total_tables": len(file_ids),
            "total_relationships": len(relationships)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error detecting relationships: {str(e)}")

@app.get("/api/v1/csv/session/{session_id}/entity-graph")
async def get_entity_relationship_graph(session_id: str):
    """Get entity relationship graph for session"""
    if session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = csv_sessions_db[session_id]
    files_data = {fid: csv_files_db[fid] for fid in session["files"] if fid in csv_files_db}
    relationships = session.get("relationships", [])
    
    graph_data = generate_entity_relationship_graph(files_data, relationships)
    
    return {
        "graph_data": graph_data,
        "session_id": session_id,
        "files_count": len(files_data),
        "relationships_count": len(relationships)
    }

# ==================== ADVANCED DASHBOARD GENERATION ====================

class DashboardConfig(BaseModel):
    """Dashboard configuration model"""
    session_id: str
    name: str
    description: Optional[str] = ""
    charts: List[Dict]
    layout: Optional[Dict] = None
    filters: Optional[List[Dict]] = None

class QueryReportRequest(BaseModel):
    """Query-based report request model"""
    session_id: str
    query: str
    chart_preferences: Optional[Dict] = None

@app.post("/api/v1/csv/dashboard/generate")
async def generate_professional_dashboard(config: DashboardConfig):
    """Generate professional PowerBI-style dashboard"""
    if config.session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = csv_sessions_db[config.session_id]
    files_data = {fid: csv_files_db[fid] for fid in session["files"] if fid in csv_files_db}
    
    try:
        dashboard_id = str(uuid.uuid4())
        dashboard_charts = []
        
        for chart_config in config.charts:
            chart_type = chart_config.get("type", "bar")
            table_id = chart_config.get("table_id")
            
            if table_id not in files_data:
                continue
            
            file_data = files_data[table_id]
            df = file_data["dataframe"]
            
            # Process chart configuration
            processed_config = {
                "title": chart_config.get("title", ""),
                "x": chart_config.get("x_column"),
                "y": chart_config.get("y_column"),
                "color": chart_config.get("color_column"),
                "size": chart_config.get("size_column")
            }
            
            # Apply filters if specified
            filtered_df = df.copy()
            if "filters" in chart_config:
                for filter_config in chart_config["filters"]:
                    column = filter_config.get("column")
                    operator = filter_config.get("operator", "equals")
                    value = filter_config.get("value")
                    
                    if column in filtered_df.columns:
                        if operator == "equals":
                            filtered_df = filtered_df[filtered_df[column] == value]
                        elif operator == "not_equals":
                            filtered_df = filtered_df[filtered_df[column] != value]
                        elif operator == "greater_than":
                            filtered_df = filtered_df[filtered_df[column] > value]
                        elif operator == "less_than":
                            filtered_df = filtered_df[filtered_df[column] < value]
                        elif operator == "contains":
                            filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
            
            # Apply aggregations if specified
            if "aggregation" in chart_config:
                agg_config = chart_config["aggregation"]
                group_by = agg_config.get("group_by")
                metric = agg_config.get("metric")
                operation = agg_config.get("operation", "sum")
                
                if group_by and metric and group_by in filtered_df.columns and metric in filtered_df.columns:
                    if operation == "sum":
                        filtered_df = filtered_df.groupby(group_by)[metric].sum().reset_index()
                    elif operation == "avg":
                        filtered_df = filtered_df.groupby(group_by)[metric].mean().reset_index()
                    elif operation == "count":
                        filtered_df = filtered_df.groupby(group_by)[metric].count().reset_index()
                    elif operation == "max":
                        filtered_df = filtered_df.groupby(group_by)[metric].max().reset_index()
                    elif operation == "min":
                        filtered_df = filtered_df.groupby(group_by)[metric].min().reset_index()
            
            # Generate professional chart
            fig = create_professional_chart(chart_type, filtered_df, processed_config)
            
            if fig:
                chart_data = {
                    "id": str(uuid.uuid4()),
                    "type": chart_type,
                    "title": processed_config["title"],
                    "plotly_json": fig.to_json(),
                    "config": chart_config,
                    "data_summary": {
                        "rows": len(filtered_df),
                        "columns": list(filtered_df.columns)
                    }
                }
                dashboard_charts.append(chart_data)
        
        # Create dashboard
        dashboard = {
            "dashboard_id": dashboard_id,
            "session_id": config.session_id,
            "name": config.name,
            "description": config.description,
            "charts": dashboard_charts,
            "layout": config.layout,
            "filters": config.filters,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_charts": len(dashboard_charts)
        }
        
        # Store dashboard
        csv_sessions_db[config.session_id]["dashboards"].append(dashboard_id)
        power_bi_reports_db[dashboard_id] = dashboard
        
        return dashboard
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating dashboard: {str(e)}")

@app.post("/api/v1/csv/query-report")
async def generate_query_based_report(request: QueryReportRequest):
    """Generate report based on natural language query"""
    if request.session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = csv_sessions_db[request.session_id]
    files_data = {fid: csv_files_db[fid] for fid in session["files"] if fid in csv_files_db}
    
    try:
        # Process natural language query
        query_spec = process_natural_language_query(request.query, files_data)
        
        # Generate appropriate visualization
        if query_spec["type"] == "error":
            raise HTTPException(status_code=400, detail=f"Error processing query: {query_spec.get('error', 'Unknown error')}")
        
        charts = []
        report_data = []
        
        # If specific tables/columns were identified
        if query_spec["tables"] and query_spec["columns"]:
            for table_info in query_spec["columns"]:
                table_id = table_info["table"]
                column = table_info["column"]
                
                if table_id in files_data:
                    df = files_data[table_id]["dataframe"]
                    
                    # Generate chart based on query type
                    chart_config = {
                        "title": f"{request.query} - {files_data[table_id]['filename']}",
                        "type": query_spec["suggested_chart"]
                    }
                    
                    if query_spec["type"] == "top_n" and query_spec["limit"]:
                        # Create top N chart
                        if table_info["type"] == "numeric":
                            sorted_df = df.nlargest(query_spec["limit"], column) if query_spec["order"] == "desc" else df.nsmallest(query_spec["limit"], column)
                            chart_config.update({
                                "x": column,
                                "y": "index"
                            })
                        else:
                            # For text columns, show value counts
                            value_counts = df[column].value_counts().head(query_spec["limit"])
                            sorted_df = value_counts.reset_index()
                            sorted_df.columns = [column, "count"]
                            chart_config.update({
                                "x": column,
                                "y": "count"
                            })
                        
                        fig = create_professional_chart(query_spec["suggested_chart"], sorted_df, chart_config)
                        
                    elif query_spec["type"] == "distribution":
                        # Create distribution chart
                        if table_info["type"] == "numeric":
                            chart_config.update({"x": column})
                            fig = create_professional_chart("histogram", df, chart_config)
                        else:
                            value_counts = df[column].value_counts()
                            chart_config.update({
                                "values": value_counts.values,
                                "names": value_counts.index
                            })
                            fig = create_professional_chart("pie", df, chart_config)
                    
                    else:
                        # Default visualization
                        chart_config.update({"x": column})
                        fig = create_professional_chart(query_spec["suggested_chart"], df, chart_config)
                    
                    if fig:
                        charts.append({
                            "id": str(uuid.uuid4()),
                            "title": chart_config["title"],
                            "type": query_spec["suggested_chart"],
                            "plotly_json": fig.to_json(),
                            "table": table_id,
                            "column": column
                        })
        
        # Create report
        report_id = str(uuid.uuid4())
        report = {
            "report_id": report_id,
            "session_id": request.session_id,
            "query": request.query,
            "query_spec": query_spec,
            "charts": charts,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_charts": len(charts)
        }
        
        power_bi_reports_db[report_id] = report
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating query report: {str(e)}")

@app.get("/api/v1/csv/dashboard/{dashboard_id}")
async def get_dashboard(dashboard_id: str):
    """Get specific dashboard by ID"""
    if dashboard_id not in power_bi_reports_db:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return power_bi_reports_db[dashboard_id]

@app.get("/api/v1/csv/session/{session_id}/dashboards")
async def get_session_dashboards(session_id: str):
    """Get all dashboards for a session"""
    if session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = csv_sessions_db[session_id]
    dashboard_ids = session.get("dashboards", [])
    
    dashboards = []
    for dashboard_id in dashboard_ids:
        if dashboard_id in power_bi_reports_db:
            dashboard = power_bi_reports_db[dashboard_id]
            # Return summary without full chart data
            dashboards.append({
                "dashboard_id": dashboard["dashboard_id"],
                "name": dashboard["name"],
                "description": dashboard["description"],
                "created_at": dashboard["created_at"],
                "total_charts": dashboard["total_charts"]
            })
    
    return {
        "session_id": session_id,
        "dashboards": dashboards,
        "total": len(dashboards)
    }

# ==================== ADVANCED COLUMN SELECTION ====================

class ColumnSelectionConfig(BaseModel):
    """Advanced column selection configuration"""
    session_id: str
    tables: List[Dict]  # Each dict contains table_id and selected columns
    join_config: Optional[List[Dict]] = None  # Join specifications
    filters: Optional[List[Dict]] = None
    aggregations: Optional[List[Dict]] = None

@app.post("/api/v1/csv/column-selection/preview")
async def preview_column_selection(config: ColumnSelectionConfig):
    """Preview data based on column selection and joins"""
    if config.session_id not in csv_sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = csv_sessions_db[config.session_id]
    files_data = {fid: csv_files_db[fid] for fid in session["files"] if fid in csv_files_db}
    
    try:
        result_data = None
        
        if len(config.tables) == 1:
            # Single table selection
            table_config = config.tables[0]
            table_id = table_config["table_id"]
            selected_columns = table_config["columns"]
            
            if table_id in files_data:
                df = files_data[table_id]["dataframe"]
                result_data = df[selected_columns].copy()
        
        elif len(config.tables) > 1 and config.join_config:
            # Multi-table join
            base_table = config.tables[0]
            base_df = files_data[base_table["table_id"]]["dataframe"][base_table["columns"]].copy()
            
            for i, table_config in enumerate(config.tables[1:], 1):
                if i <= len(config.join_config):
                    join_spec = config.join_config[i-1]
                    join_df = files_data[table_config["table_id"]]["dataframe"][table_config["columns"]].copy()
                    
                    # Perform join
                    base_df = base_df.merge(
                        join_df,
                        left_on=join_spec["left_column"],
                        right_on=join_spec["right_column"],
                        how=join_spec.get("join_type", "inner")
                    )
            
            result_data = base_df
        
        if result_data is not None:
            # Apply filters
            if config.filters:
                for filter_config in config.filters:
                    column = filter_config.get("column")
                    operator = filter_config.get("operator", "equals")
                    value = filter_config.get("value")
                    
                    if column in result_data.columns:
                        if operator == "equals":
                            result_data = result_data[result_data[column] == value]
                        elif operator == "contains":
                            result_data = result_data[result_data[column].astype(str).str.contains(str(value), na=False)]
            
            # Apply aggregations
            if config.aggregations:
                for agg_config in config.aggregations:
                    group_by = agg_config.get("group_by")
                    metric = agg_config.get("metric")
                    operation = agg_config.get("operation", "sum")
                    
                    if group_by and metric and group_by in result_data.columns and metric in result_data.columns:
                        if operation == "sum":
                            result_data = result_data.groupby(group_by)[metric].sum().reset_index()
                        elif operation == "avg":
                            result_data = result_data.groupby(group_by)[metric].mean().reset_index()
                        elif operation == "count":
                            result_data = result_data.groupby(group_by)[metric].count().reset_index()
            
            # Return preview
            preview_rows = result_data.head(20)
            
            return {
                "preview_data": preview_rows.to_dict('records'),
                "total_rows": len(result_data),
                "columns": list(result_data.columns),
                "column_types": {col: str(result_data[col].dtype) for col in result_data.columns},
                "summary_stats": {
                    col: {
                        "count": int(result_data[col].count()),
                        "null_count": int(result_data[col].isnull().sum()),
                        "unique_count": int(result_data[col].nunique())
                    } for col in result_data.columns
                }
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unable to process column selection")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error previewing selection: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)