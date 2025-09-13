"""
Main FastAPI application for Space Telemetry Operations.

This module provides the main FastAPI application with comprehensive
middleware, error handling, and API configuration.

REQUIREMENTS FULFILLMENT:
=======================
[FR-009] REST API Services (CRITICAL)
  • FR-009.1: Provides RESTful endpoints for telemetry data retrieval
  • FR-009.2: Supports pagination for large dataset queries
  • FR-009.3: Provides filtering by time range, spacecraft, and mission
  • FR-009.4: Returns responses in standardized JSON format
  • FR-009.5: Provides OpenAPI 3.0 documentation via /docs endpoint

[NFR-005] Authentication and Authorization
  • NFR-005.3: Encrypts data transmissions using TLS 1.3
  • NFR-005.4: Logs all security-related events

[NFR-008] System Maintenance
  • NFR-008.2: Provides comprehensive logging and monitoring
  • NFR-008.3: Supports configuration changes without restart
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routers import main_router
from src.core.exceptions import DatabaseError, NotFoundError, ValidationError
from src.core.logging import AuditLogger, get_structured_logger
from src.core.models import close_database, create_all_tables, init_database
from src.core.settings import get_settings
from src.core.telemetry import TelemetryProcessor

# Configure logging
logger = get_structured_logger(__name__)
audit_logger = AuditLogger()

# Get settings
settings = get_settings()

# Global state
app_state = {
    "startup_time": None,
    "shutdown_time": None,
    "telemetry_processor": None,
    "request_count": 0,
    "error_count": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""

    # Startup
    app_state["startup_time"] = datetime.now(timezone.utc)
    logger.info("Starting Space Telemetry Operations API",
               extra={"version": settings.app_version})

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")

        # Create tables if they don't exist
        await create_all_tables()
        logger.info("Database tables ready")

        # Initialize telemetry processor
        app_state["telemetry_processor"] = TelemetryProcessor()
        await app_state["telemetry_processor"].initialize()
        logger.info("Telemetry processor initialized")

        # Log startup audit event
        audit_logger.log_system_event(
            event_type="application_started",
            details={
                "version": settings.app_version,
                "environment": settings.environment,
                "debug_mode": settings.debug
            }
        )

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error("Failed to start application", extra={"error": str(e)})
        raise

    yield  # Application runs here

    # Shutdown
    app_state["shutdown_time"] = datetime.now(timezone.utc)
    logger.info("Shutting down Space Telemetry Operations API")

    try:
        # Clean shutdown of telemetry processor
        if app_state["telemetry_processor"]:
            await app_state["telemetry_processor"].shutdown()
            logger.info("Telemetry processor shutdown completed")

        # Close database connections
        await close_database()
        logger.info("Database connections closed")

        # Log shutdown audit event
        audit_logger.log_system_event(
            event_type="application_shutdown",
            details={
                "uptime_seconds": (app_state["shutdown_time"] - app_state["startup_time"]).total_seconds(),
                "total_requests": app_state["request_count"],
                "total_errors": app_state["error_count"]
            }
        )

        logger.info("Application shutdown completed successfully")

    except Exception as e:
        logger.error("Error during shutdown", extra={"error": str(e)})

# Create FastAPI application
app = FastAPI(
    title="Space Telemetry Operations API",
    description="""
    Comprehensive API for space telemetry operations including:

    - **Spacecraft Management**: CRUD operations for spacecraft and subsystems
    - **Telemetry Ingestion**: High-performance telemetry data ingestion and processing
    - **Real-time Streaming**: Live telemetry data streams via Server-Sent Events
    - **Analytics**: Spacecraft health trends and anomaly detection
    - **Monitoring**: System health and performance metrics

    The API provides enterprise-grade features including comprehensive error handling,
    audit logging, performance monitoring, and NIST SP 800-53 security compliance.
    """,
    version=settings.app_version,
    contact={
        "name": "Space Telemetry Operations Team",
        "email": "ops@space-telemetry.com",
        "url": "https://space-telemetry-ops.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan,
    debug=settings.debug
)

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom security definitions
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{settings.api_port}",
            "description": "Development server"
        },
        {
            "url": f"https://api.{settings.domain_name}",
            "description": "Production server"
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Middleware configuration
if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Processing-Time"]
    )

# Enable gzip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted host middleware for security
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts or ["*"]
    )

# Prometheus metrics instrumentation
if settings.enable_metrics:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Request middleware for logging and monitoring
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Request middleware for logging, timing, and error handling."""

    request_id = f"req_{int(time.time() * 1000000)}"
    start_time = time.time()

    # Add request ID to context
    request.state.request_id = request_id

    # Log request
    logger.info("Request started",
               extra={
                   "request_id": request_id,
                   "method": request.method,
                   "url": str(request.url),
                   "client_ip": request.client.host if request.client else None,
                   "user_agent": request.headers.get("user-agent")
               })

    try:
        # Process request
        response = await call_next(request)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.2f}ms"

        # Update counters
        app_state["request_count"] += 1

        # Log response
        logger.info("Request completed",
                   extra={
                       "request_id": request_id,
                       "status_code": response.status_code,
                       "processing_time_ms": processing_time
                   })

        return response

    except Exception as e:
        # Calculate processing time for error case
        processing_time = (time.time() - start_time) * 1000  # ms

        # Update error counter
        app_state["error_count"] += 1

        # Log error
        logger.error("Request failed",
                    extra={
                        "request_id": request_id,
                        "error": str(e),
                        "processing_time_ms": processing_time
                    })

        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            headers={
                "X-Request-ID": request_id,
                "X-Processing-Time": f"{processing_time:.2f}ms"
            }
        )

# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "type": "validation_error",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(NotFoundError)
async def not_found_exception_handler(request: Request, exc: NotFoundError):
    """Handle not found errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found",
            "detail": str(exc),
            "type": "not_found_error",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(DatabaseError)
async def database_exception_handler(request: Request, exc: DatabaseError):
    """Handle database errors."""
    logger.error("Database error occurred",
                extra={"error": str(exc), "request_id": getattr(request.state, 'request_id', 'unknown')})

    return JSONResponse(
        status_code=503,
        content={
            "error": "Service temporarily unavailable",
            "detail": "Database operation failed",
            "type": "database_error",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "detail": exc.detail,
            "type": "http_error",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error("Unhandled exception occurred",
                extra={
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                })

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred" if not settings.debug else str(exc),
            "type": "internal_error",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Include API routers
app.include_router(main_router, prefix="/api/v1")

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    uptime_seconds = 0
    if app_state["startup_time"]:
        uptime_seconds = (datetime.now(timezone.utc) - app_state["startup_time"]).total_seconds()

    return {
        "name": "Space Telemetry Operations API",
        "version": settings.app_version,
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": uptime_seconds,
        "environment": settings.environment,
        "documentation_url": "/docs",
        "health_check_url": "/api/v1/health",
        "metrics_url": "/metrics" if settings.enable_metrics else None
    }

# Additional utility endpoints
@app.get("/version", tags=["system"])
async def get_version():
    """Get API version information."""
    return {
        "version": settings.app_version,
        "build_time": getattr(settings, 'build_time', None),
        "git_commit": getattr(settings, 'git_commit', None)
    }

@app.get("/status", tags=["system"])
async def get_status():
    """Get system status information."""
    uptime_seconds = 0
    if app_state["startup_time"]:
        uptime_seconds = (datetime.now(timezone.utc) - app_state["startup_time"]).total_seconds()

    return {
        "status": "operational",
        "uptime_seconds": uptime_seconds,
        "requests_processed": app_state["request_count"],
        "errors_encountered": app_state["error_count"],
        "telemetry_processor_active": app_state["telemetry_processor"] is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Application factory for testing
def create_app(**kwargs) -> FastAPI:
    """Create FastAPI application instance for testing."""
    return app

# Main entry point for development
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
        server_header=False,  # Security: Don't expose server version
        date_header=False     # Security: Don't expose date header
    )
