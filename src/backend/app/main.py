"""
Space Telemetry Operations - FastAPI Backend

Main application entry point for the space telemetry operations system.
Provides REST API endpoints for telemetry data access, mission management,
and system monitoring with enhanced dashboard capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import asyncio
import logging
import time
from datetime import datetime, timedelta
import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Import dashboard enhancement services
from ...services.dashboard_enhancement.api import router as dashboard_router
from ...services.dashboard_enhancement.integration import (
    DashboardIntegrationService,
    dashboard_integration_service
)
from ...services.anomaly_detection.anomaly_detection import AnomalyDetectionService
from ...services.performance_optimization.performance_service import PerformanceOptimizationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
telemetry_queries = Counter('telemetry_queries_total', 'Total telemetry queries', ['type'])
active_connections = Counter('websocket_connections_active', 'Active WebSocket connections')

# Security
security = HTTPBearer()

# Pydantic models
class TelemetryData(BaseModel):
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    satellite_id: str = Field(..., min_length=1, max_length=50)
    mission_id: str = Field(..., min_length=1, max_length=50)
    telemetry_type: str = Field(..., min_length=1, max_length=50)
    data: Dict[str, Any] = Field(..., description="Telemetry payload")
    quality_score: Optional[int] = Field(None, ge=0, le=100)

class TelemetryQuery(BaseModel):
    satellite_ids: Optional[List[str]] = None
    mission_ids: Optional[List[str]] = None
    telemetry_types: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)

class HealthStatus(BaseModel):
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str = "1.0.0"
    services: Dict[str, str]
    metrics: Dict[str, Any]

class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Middleware for metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        request_duration.observe(duration)
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        return response

# Global variables for connections and services
redis_pool = None
postgres_pool = None
anomaly_service = None
performance_service = None
app_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global redis_pool, postgres_pool, anomaly_service, performance_service

    logger.info("Starting Space Telemetry API with Enhanced Dashboard...")

    # Initialize Redis connection
    try:
        redis_pool = redis.ConnectionPool.from_url(
            "redis://redis:6379/0",
            max_connections=20,
            retry_on_timeout=True
        )
        # Test connection
        redis_client = redis.Redis(connection_pool=redis_pool)
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_pool = None

    # Initialize PostgreSQL connection pool
    try:
        postgres_pool = await asyncpg.create_pool(
            "postgresql://telemetry_user:telemetry_pass123@postgres:5432/telemetry",
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool established")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        postgres_pool = None

    # Initialize enhanced services
    try:
        # Initialize anomaly detection service
        anomaly_service = AnomalyDetectionService()
        await anomaly_service.initialize()
        logger.info("Anomaly detection service initialized")

        # Initialize performance optimization service
        performance_service = PerformanceOptimizationService()
        await performance_service.initialize()
        logger.info("Performance optimization service initialized")

        # Initialize dashboard integration service
        await dashboard_integration_service.initialize(
            anomaly_service=anomaly_service,
            performance_service=performance_service,
            telemetry_processor=None  # Would be connected to actual processor
        )
        await dashboard_integration_service.start()
        logger.info("Dashboard integration service started")

    except Exception as e:
        logger.error(f"Failed to initialize enhanced services: {e}")

    yield

    # Cleanup
    logger.info("Shutting down Space Telemetry API...")

    # Stop enhanced services
    if 'dashboard_integration_service' in globals():
        try:
            await dashboard_integration_service.stop()
            logger.info("Dashboard integration service stopped")
        except Exception as e:
            logger.error(f"Error stopping dashboard service: {e}")

    if anomaly_service:
        try:
            await anomaly_service.stop()
            logger.info("Anomaly detection service stopped")
        except Exception as e:
            logger.error(f"Error stopping anomaly service: {e}")

    if performance_service:
        try:
            await performance_service.stop()
            logger.info("Performance optimization service stopped")
        except Exception as e:
            logger.error(f"Error stopping performance service: {e}")

    # Cleanup database connections
    if redis_pool:
        await redis_pool.disconnect()
    if postgres_pool:
        await postgres_pool.close()

# Create FastAPI application
app = FastAPI(
    title="Space Telemetry Operations API",
    description="Enterprise-grade REST API for space telemetry data management with Enhanced Mission Control Dashboard",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(MetricsMiddleware)

# Include enhanced dashboard API routes
app.include_router(dashboard_router, tags=["Enhanced Mission Control Dashboard"])

# Mount static files for dashboard frontend (optional - can be served separately)
try:
    app.mount("/dashboard", StaticFiles(directory="../../services/dashboard-enhancement/static", html=True), name="dashboard")
    logger.info("Dashboard static files mounted at /dashboard")
except Exception as e:
    logger.warning(f"Could not mount dashboard static files: {e}")

logger.info("Enhanced Mission Control Dashboard API integration complete")

# Dependency to get database connections
async def get_redis():
    """Get Redis connection"""
    if not redis_pool:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return redis.Redis(connection_pool=redis_pool)

async def get_postgres():
    """Get PostgreSQL connection"""
    if not postgres_pool:
        raise HTTPException(status_code=503, detail="PostgreSQL unavailable")
    return postgres_pool

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token (simplified for demo)"""
    # In production, implement proper JWT verification
    if not credentials or credentials.credentials != "demo-token":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Health check endpoint
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check endpoint"""

    # Check service health
    services = {}

    # Check Redis
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        services["redis"] = "healthy"
    except:
        services["redis"] = "unhealthy"

    # Check PostgreSQL
    try:
        pool = await get_postgres()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        services["postgresql"] = "healthy"
    except:
        services["postgresql"] = "unhealthy"

    # Calculate uptime
    uptime = time.time() - app_start_time

    # Overall status
    status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"

    return HealthStatus(
        status=status,
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime,
        services=services,
        metrics={
            "total_requests": sum(request_count._value._value.values()),
            "average_response_time": request_duration._sum._value / max(request_duration._count._value, 1),
            "active_connections": 0  # Would be populated by WebSocket handler
        }
    )

# Telemetry endpoints
@app.get("/api/telemetry", response_model=ApiResponse)
async def get_telemetry(
    query: TelemetryQuery = Depends(),
    postgres = Depends(get_postgres)
):
    """Retrieve telemetry data with filtering and pagination"""

    telemetry_queries.labels(type="fetch").inc()

    try:
        # Build dynamic query
        where_conditions = ["1=1"]
        params = []
        param_count = 0

        if query.satellite_ids:
            param_count += 1
            where_conditions.append(f"satellite_id = ANY(${param_count})")
            params.append(query.satellite_ids)

        if query.mission_ids:
            param_count += 1
            where_conditions.append(f"mission_id = ANY(${param_count})")
            params.append(query.mission_ids)

        if query.telemetry_types:
            param_count += 1
            where_conditions.append(f"telemetry_type = ANY(${param_count})")
            params.append(query.telemetry_types)

        if query.start_time:
            param_count += 1
            where_conditions.append(f"timestamp >= ${param_count}")
            params.append(query.start_time)

        if query.end_time:
            param_count += 1
            where_conditions.append(f"timestamp <= ${param_count}")
            params.append(query.end_time)

        # Add limit and offset
        param_count += 1
        limit_clause = f"LIMIT ${param_count}"
        params.append(query.limit)

        param_count += 1
        offset_clause = f"OFFSET ${param_count}"
        params.append(query.offset)

        # Execute query
        sql = f"""
        SELECT id, timestamp, satellite_id, mission_id, telemetry_type,
               category, data_quality_score, temperature, pressure, voltage,
               current, power, altitude, velocity, status, created_at
        FROM telemetry_processed
        WHERE {' AND '.join(where_conditions)}
        ORDER BY timestamp DESC
        {limit_clause} {offset_clause}
        """

        async with postgres.acquire() as conn:
            rows = await conn.fetch(sql, *params)

            # Convert to list of dicts
            telemetry_data = [dict(row) for row in rows]

            # Get total count for pagination
            count_sql = f"""
            SELECT COUNT(*) FROM telemetry_processed
            WHERE {' AND '.join(where_conditions[:-2])}  # Exclude LIMIT/OFFSET conditions
            """
            total_count = await conn.fetchval(count_sql, *params[:-2])

        return ApiResponse(
            success=True,
            data={
                "telemetry": telemetry_data,
                "pagination": {
                    "total": total_count,
                    "limit": query.limit,
                    "offset": query.offset,
                    "has_more": (query.offset + query.limit) < total_count
                }
            },
            message=f"Retrieved {len(telemetry_data)} telemetry records"
        )

    except Exception as e:
        logger.error(f"Error retrieving telemetry: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve telemetry data")

@app.get("/api/telemetry/latest", response_model=ApiResponse)
async def get_latest_telemetry(
    satellite_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    redis_client = Depends(get_redis)
):
    """Get latest telemetry data from Redis hot path"""

    telemetry_queries.labels(type="latest").inc()

    try:
        if satellite_id:
            # Get latest for specific satellite
            key = f"telemetry:{satellite_id}:latest"
            data = await redis_client.get(key)
            if data:
                import json
                telemetry = json.loads(data)
                return ApiResponse(success=True, data=telemetry)
            else:
                return ApiResponse(success=False, message="No recent data found")

        else:
            # Get latest from all active satellites
            keys = await redis_client.keys("telemetry:*:latest")
            if not keys:
                return ApiResponse(success=True, data=[], message="No active telemetry")

            # Get all latest values
            pipe = redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            results = await pipe.execute()

            # Parse JSON data
            import json
            telemetry_data = []
            for result in results:
                if result:
                    telemetry_data.append(json.loads(result))

            return ApiResponse(
                success=True,
                data=telemetry_data,
                message=f"Retrieved latest telemetry from {len(telemetry_data)} satellites"
            )

    except Exception as e:
        logger.error(f"Error retrieving latest telemetry: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve latest telemetry")

@app.get("/api/satellites", response_model=ApiResponse)
async def get_satellites(postgres = Depends(get_postgres)):
    """Get list of active satellites"""

    try:
        async with postgres.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT satellite_id,
                       COUNT(*) as message_count,
                       MAX(timestamp) as last_seen,
                       MIN(timestamp) as first_seen
                FROM telemetry_processed
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY satellite_id
                ORDER BY last_seen DESC
            """)

            satellites = [dict(row) for row in rows]

        return ApiResponse(
            success=True,
            data=satellites,
            message=f"Retrieved {len(satellites)} active satellites"
        )

    except Exception as e:
        logger.error(f"Error retrieving satellites: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve satellites")

@app.get("/api/missions", response_model=ApiResponse)
async def get_missions(postgres = Depends(get_postgres)):
    """Get list of active missions"""

    try:
        async with postgres.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT mission_id,
                       COUNT(DISTINCT satellite_id) as satellite_count,
                       COUNT(*) as message_count,
                       MAX(timestamp) as last_activity,
                       MIN(timestamp) as mission_start
                FROM telemetry_processed
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY mission_id
                ORDER BY last_activity DESC
            """)

            missions = [dict(row) for row in rows]

        return ApiResponse(
            success=True,
            data=missions,
            message=f"Retrieved {len(missions)} active missions"
        )

    except Exception as e:
        logger.error(f"Error retrieving missions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve missions")

# Analytics endpoints
@app.get("/api/analytics/summary", response_model=ApiResponse)
async def get_analytics_summary(
    hours: int = 24,
    postgres = Depends(get_postgres)
):
    """Get telemetry analytics summary"""

    try:
        async with postgres.acquire() as conn:
            # Get summary statistics
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT satellite_id) as active_satellites,
                    COUNT(DISTINCT mission_id) as active_missions,
                    AVG(data_quality_score) as avg_quality_score,
                    MIN(timestamp) as earliest_message,
                    MAX(timestamp) as latest_message
                FROM telemetry_processed
                WHERE timestamp > NOW() - INTERVAL '%s hours'
            """, hours)

            # Get message count by hour
            hourly_counts = await conn.fetch("""
                SELECT
                    date_trunc('hour', timestamp) as hour,
                    COUNT(*) as message_count
                FROM telemetry_processed
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                GROUP BY hour
                ORDER BY hour
            """, hours)

            return ApiResponse(
                success=True,
                data={
                    "summary": dict(stats),
                    "hourly_counts": [dict(row) for row in hourly_counts],
                    "time_period_hours": hours
                }
            )

    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Root endpoint
@app.get("/", response_model=ApiResponse)
async def root():
    """API root endpoint"""
    return ApiResponse(
        success=True,
        data={
            "name": "Space Telemetry Operations API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "telemetry": "/api/telemetry",
                "latest": "/api/telemetry/latest",
                "satellites": "/api/satellites",
                "missions": "/api/missions",
                "analytics": "/api/analytics/summary",
                "metrics": "/metrics"
            }
        },
        message="Space Telemetry Operations API is operational"
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8083,
        reload=True,
        log_level="info"
    )
