#!/usr/bin/env python3
"""
Simple FastAPI server for testing in Replit environment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncpg
import asyncio

# Simple FastAPI app for testing
app = FastAPI(title="Amplifi API - Development", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)