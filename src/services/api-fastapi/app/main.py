"""
FastAPI main application module for space telemetry operations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime

app = FastAPI(
    title="Space Telemetry Operations API",
    description="Mission-critical space telemetry processing and monitoring system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "service": "Space Telemetry Operations API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancer."""
    return {
        "status": "healthy",
        "service": "space-telemetry-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/telemetry/status")
async def telemetry_status():
    """Get current telemetry system status."""
    return {
        "ingestion_rate": "50000 msgs/sec",
        "active_spacecraft": 3,
        "system_health": "nominal",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
