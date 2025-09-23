#!/usr/bin/env python3
"""
Simple FastAPI server for testing in Replit environment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

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
    return {
        "status": "running",
        "version": "1.0.0",
        "environment": "development",
        "database": "not connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)