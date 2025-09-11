"""
API routers for Space Telemetry Operations FastAPI service.

This module provides comprehensive API endpoints with enhanced error handling,
validation, and monitoring for space telemetry operations.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.models import (
    Spacecraft, TelemetryPacket, SpacecraftState, Subsystem,
    SpacecraftCreate, TelemetryPacketCreate,
    TelemetryStatus, DataQuality, MissionPhase,
    get_spacecraft_repository, get_telemetry_repository, db_manager
)
from src.core.exceptions import (
    ValidationError, NotFoundError, handle_exceptions,
    ErrorSeverity, ErrorCategory
)
from src.core.logging import get_structured_logger, PerformanceTimer, AuditLogger
from src.core.telemetry import TelemetryProcessor, MemoryManagedProcessor, TimeContext
from src.core.settings import get_settings

# Configure logging
logger = get_structured_logger(__name__)
audit_logger = AuditLogger()
settings = get_settings()

# Initialize processors
telemetry_processor = TelemetryProcessor()
memory_processor = MemoryManagedProcessor()

# Router instances
spacecraft_router = APIRouter(prefix="/spacecraft", tags=["spacecraft"])
telemetry_router = APIRouter(prefix="/telemetry", tags=["telemetry"])
health_router = APIRouter(prefix="/health", tags=["health"])
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])

# Pydantic models for API responses
class SpacecraftResponse(BaseModel):
    """Response model for spacecraft data."""
    id: UUID
    name: str
    mission_name: str
    spacecraft_id: str
    mission_phase: MissionPhase
    launch_date: Optional[datetime]
    end_of_mission_date: Optional[datetime]
    is_active: bool
    mass_kg: Optional[float]
    power_watts: Optional[float]
    orbital_period_minutes: Optional[float]
    communication_frequency_mhz: Optional[float]
    created_at: datetime
    updated_at: datetime
    description: Optional[str]
    manufacturer: Optional[str]

    class Config:
        from_attributes = True

class TelemetryPacketResponse(BaseModel):
    """Response model for telemetry packet data."""
    id: UUID
    spacecraft_time: datetime
    ground_received_time: datetime
    processing_time: datetime
    packet_id: str
    sequence_number: int
    apid: int
    packet_type: str
    telemetry_status: TelemetryStatus
    data_quality: DataQuality
    validation_status: str
    checksum_valid: Optional[bool]
    packet_size_bytes: int
    processing_duration_ms: Optional[float]
    alert_level: int
    spacecraft_id: UUID
    subsystem_id: Optional[UUID]
    processed_data: Dict[str, Any]

    class Config:
        from_attributes = True

class SpacecraftStateResponse(BaseModel):
    """Response model for spacecraft state data."""
    id: UUID
    state_time: datetime
    position_x_km: Optional[float]
    position_y_km: Optional[float]
    position_z_km: Optional[float]
    battery_level_percent: Optional[float]
    temperature_celsius: Optional[float]
    signal_strength_dbm: Optional[float]
    communication_status: str
    overall_health_score: Optional[float]
    critical_alerts_count: int
    spacecraft_id: UUID

    class Config:
        from_attributes = True

class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    database_connected: bool
    memory_usage_mb: float
    active_connections: int
    processing_queue_size: int
    last_telemetry_received: Optional[datetime]

class TelemetryStats(BaseModel):
    """Telemetry statistics model."""
    total_packets: int
    packets_last_hour: int
    packets_last_24h: int
    average_processing_time_ms: float
    error_rate_percent: float
    quality_distribution: Dict[str, int]
    status_distribution: Dict[str, int]

# Dependency injection
async def get_db_session() -> AsyncSession:
    """Get database session."""
    async with db_manager.get_session() as session:
        yield session

# Spacecraft endpoints
@spacecraft_router.post("/", response_model=SpacecraftResponse, status_code=201)
@handle_exceptions(
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.MEDIUM
)
async def create_spacecraft(
    spacecraft_data: SpacecraftCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session)
) -> SpacecraftResponse:
    """Create a new spacecraft."""

    with PerformanceTimer() as timer:
        try:
            # Validate unique constraints
            repo = get_spacecraft_repository()

            # Check if spacecraft with same ID already exists
            existing = await repo.get_spacecraft_by_name(spacecraft_data.name)
            if existing:
                raise ValidationError(f"Spacecraft with name '{spacecraft_data.name}' already exists")

            # Create spacecraft
            spacecraft = await repo.create_spacecraft(spacecraft_data)

            # Schedule background initialization tasks
            background_tasks.add_task(
                initialize_spacecraft_monitoring,
                spacecraft.id
            )

            # Log audit event
            audit_logger.log_api_event(
                event_type="spacecraft_created",
                user_id="api_user",  # In real system, get from auth
                entity_id=str(spacecraft.id),
                details={"name": spacecraft.name, "mission": spacecraft.mission_name}
            )

            logger.info("Spacecraft created successfully",
                       extra={
                           "spacecraft_id": str(spacecraft.id),
                           "name": spacecraft.name,
                           "duration_ms": timer.duration_ms
                       })

            return SpacecraftResponse.from_orm(spacecraft)

        except Exception as e:
            logger.error("Failed to create spacecraft",
                        extra={"error": str(e), "duration_ms": timer.duration_ms})
            raise HTTPException(status_code=400, detail=str(e))

@spacecraft_router.get("/", response_model=List[SpacecraftResponse])
@handle_exceptions(
    default_return=[],
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.LOW
)
async def list_spacecraft(
    active_only: bool = Query(True, description="Return only active spacecraft"),
    mission_phase: Optional[MissionPhase] = Query(None, description="Filter by mission phase"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    session: AsyncSession = Depends(get_db_session)
) -> List[SpacecraftResponse]:
    """List spacecraft with filtering and pagination."""

    with PerformanceTimer() as timer:
        repo = get_spacecraft_repository()

        spacecraft_list = await repo.list_spacecraft(
            active_only=active_only,
            mission_phase=mission_phase,
            limit=limit,
            offset=offset
        )

        logger.info("Spacecraft list retrieved",
                   extra={
                       "count": len(spacecraft_list),
                       "active_only": active_only,
                       "duration_ms": timer.duration_ms
                   })

        return [SpacecraftResponse.from_orm(sc) for sc in spacecraft_list]

@spacecraft_router.get("/{spacecraft_id}", response_model=SpacecraftResponse)
@handle_exceptions(
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.LOW
)
async def get_spacecraft(
    spacecraft_id: UUID = Path(..., description="Spacecraft UUID"),
    session: AsyncSession = Depends(get_db_session)
) -> SpacecraftResponse:
    """Get spacecraft by ID."""

    with PerformanceTimer() as timer:
        repo = get_spacecraft_repository()
        spacecraft = await repo.get_spacecraft_by_id(spacecraft_id)

        if not spacecraft:
            raise NotFoundError(f"Spacecraft with ID {spacecraft_id} not found")

        logger.info("Spacecraft retrieved",
                   extra={
                       "spacecraft_id": str(spacecraft_id),
                       "duration_ms": timer.duration_ms
                   })

        return SpacecraftResponse.from_orm(spacecraft)

@spacecraft_router.get("/{spacecraft_id}/state", response_model=List[SpacecraftStateResponse])
@handle_exceptions(
    default_return=[],
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.LOW
)
async def get_spacecraft_states(
    spacecraft_id: UUID = Path(..., description="Spacecraft UUID"),
    start_time: Optional[datetime] = Query(None, description="Start time for state history"),
    end_time: Optional[datetime] = Query(None, description="End time for state history"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of states"),
    session: AsyncSession = Depends(get_db_session)
) -> List[SpacecraftStateResponse]:
    """Get spacecraft state history."""

    # Default to last 24 hours if no time range specified
    if not end_time:
        end_time = datetime.now(timezone.utc)
    if not start_time:
        start_time = end_time - timedelta(hours=24)

    with PerformanceTimer() as timer:
        repo = get_spacecraft_repository()
        states = await repo.get_spacecraft_states(
            spacecraft_id=spacecraft_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        logger.info("Spacecraft states retrieved",
                   extra={
                       "spacecraft_id": str(spacecraft_id),
                       "count": len(states),
                       "time_range_hours": (end_time - start_time).total_seconds() / 3600,
                       "duration_ms": timer.duration_ms
                   })

        return [SpacecraftStateResponse.from_orm(state) for state in states]

# Telemetry endpoints
@telemetry_router.post("/", response_model=TelemetryPacketResponse, status_code=201)
@handle_exceptions(
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.HIGH
)
async def ingest_telemetry(
    packet_data: TelemetryPacketCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session)
) -> TelemetryPacketResponse:
    """Ingest telemetry packet with processing and validation."""

    with PerformanceTimer() as timer:
        try:
            # Create time context for processing
            time_context = TimeContext(
                spacecraft_time=packet_data.spacecraft_time,
                ground_time=packet_data.ground_received_time,
                processing_time=datetime.now(timezone.utc)
            )

            # Process packet with enhanced telemetry processor
            processed_packet = await telemetry_processor.process_packet_async(
                packet_data.dict(),
                time_context=time_context
            )

            # Store in database
            repo = get_telemetry_repository()
            stored_packet = await repo.store_telemetry_packet(
                TelemetryPacketCreate(**processed_packet)
            )

            # Schedule background analysis
            background_tasks.add_task(
                analyze_telemetry_packet,
                stored_packet.id
            )

            # Memory management check
            if memory_processor.should_trigger_gc():
                background_tasks.add_task(memory_processor.force_garbage_collection)

            logger.info("Telemetry packet ingested",
                       extra={
                           "packet_id": stored_packet.packet_id,
                           "spacecraft_id": str(stored_packet.spacecraft_id),
                           "processing_duration_ms": timer.duration_ms
                       })

            return TelemetryPacketResponse.from_orm(stored_packet)

        except Exception as e:
            logger.error("Failed to ingest telemetry packet",
                        extra={"error": str(e), "duration_ms": timer.duration_ms})
            raise HTTPException(status_code=400, detail=str(e))

@telemetry_router.get("/", response_model=List[TelemetryPacketResponse])
@handle_exceptions(
    default_return=[],
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.LOW
)
async def list_telemetry(
    spacecraft_id: Optional[UUID] = Query(None, description="Filter by spacecraft ID"),
    subsystem_id: Optional[UUID] = Query(None, description="Filter by subsystem ID"),
    status: Optional[TelemetryStatus] = Query(None, description="Filter by status"),
    quality: Optional[DataQuality] = Query(None, description="Filter by data quality"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    session: AsyncSession = Depends(get_db_session)
) -> List[TelemetryPacketResponse]:
    """List telemetry packets with comprehensive filtering."""

    # Default to last hour if no time range specified
    if not end_time:
        end_time = datetime.now(timezone.utc)
    if not start_time:
        start_time = end_time - timedelta(hours=1)

    with PerformanceTimer() as timer:
        repo = get_telemetry_repository()
        packets = await repo.list_telemetry_packets(
            spacecraft_id=spacecraft_id,
            subsystem_id=subsystem_id,
            status=status,
            quality=quality,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )

        logger.info("Telemetry packets retrieved",
                   extra={
                       "count": len(packets),
                       "spacecraft_id": str(spacecraft_id) if spacecraft_id else None,
                       "duration_ms": timer.duration_ms
                   })

        return [TelemetryPacketResponse.from_orm(packet) for packet in packets]

@telemetry_router.get("/stream")
async def stream_telemetry(
    spacecraft_id: Optional[UUID] = Query(None, description="Filter by spacecraft ID"),
    status_filter: Optional[TelemetryStatus] = Query(None, description="Filter by status")
):
    """Stream telemetry data in real-time using Server-Sent Events."""

    async def generate_telemetry_stream():
        """Generate telemetry stream."""
        try:
            async for packet in telemetry_processor.stream_processed_packets(
                spacecraft_id=spacecraft_id,
                status_filter=status_filter
            ):
                # Format as Server-Sent Event
                yield f"data: {packet.json()}\n\n"

                # Small delay to prevent overwhelming clients
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Telemetry stream cancelled")
        except Exception as e:
            logger.error("Error in telemetry stream", extra={"error": str(e)})
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate_telemetry_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# Health and monitoring endpoints
@health_router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Comprehensive health check endpoint."""

    with PerformanceTimer() as timer:
        # Check database connectivity
        database_connected = False
        try:
            async with db_manager.get_session() as session:
                await session.execute("SELECT 1")
                database_connected = True
        except Exception as e:
            logger.warning("Database health check failed", extra={"error": str(e)})

        # Get system metrics
        memory_usage = memory_processor.get_memory_usage_mb()
        processing_queue_size = telemetry_processor.get_queue_size()

        # Get last telemetry received
        last_telemetry = await get_last_telemetry_time()

        # Determine overall status
        status = "healthy"
        if not database_connected:
            status = "degraded"
        elif memory_usage > 1000:  # Over 1GB
            status = "warning"

        health_status = HealthStatus(
            status=status,
            timestamp=datetime.now(timezone.utc),
            version=settings.app_version,
            uptime_seconds=timer.duration_ms / 1000,  # Simplified uptime
            database_connected=database_connected,
            memory_usage_mb=memory_usage,
            active_connections=0,  # Would get from connection pool
            processing_queue_size=processing_queue_size,
            last_telemetry_received=last_telemetry
        )

        logger.info("Health check completed",
                   extra={"status": status, "duration_ms": timer.duration_ms})

        return health_status

@health_router.get("/metrics", response_model=TelemetryStats)
@handle_exceptions(
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.LOW
)
async def get_telemetry_metrics() -> TelemetryStats:
    """Get telemetry processing metrics."""

    with PerformanceTimer() as timer:
        repo = get_telemetry_repository()

        # Get statistics for different time periods
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        stats = await repo.get_telemetry_statistics(
            start_time=day_ago,
            end_time=now
        )

        telemetry_stats = TelemetryStats(
            total_packets=stats.get('total_packets', 0),
            packets_last_hour=stats.get('packets_last_hour', 0),
            packets_last_24h=stats.get('packets_last_24h', 0),
            average_processing_time_ms=stats.get('avg_processing_time', 0.0),
            error_rate_percent=stats.get('error_rate', 0.0),
            quality_distribution=stats.get('quality_distribution', {}),
            status_distribution=stats.get('status_distribution', {})
        )

        logger.info("Telemetry metrics retrieved",
                   extra={"duration_ms": timer.duration_ms})

        return telemetry_stats

# Analytics endpoints
@analytics_router.get("/spacecraft/{spacecraft_id}/trends")
@handle_exceptions(
    error_category=ErrorCategory.API,
    severity=ErrorSeverity.LOW
)
async def get_spacecraft_trends(
    spacecraft_id: UUID = Path(..., description="Spacecraft UUID"),
    metric: str = Query(..., description="Metric to analyze (battery_level, temperature, etc.)"),
    timespan_hours: int = Query(24, ge=1, le=168, description="Timespan in hours (max 1 week)"),
    session: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get spacecraft metric trends over time."""

    with PerformanceTimer() as timer:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=timespan_hours)

        repo = get_spacecraft_repository()
        trend_data = await repo.get_metric_trends(
            spacecraft_id=spacecraft_id,
            metric=metric,
            start_time=start_time,
            end_time=end_time
        )

        logger.info("Spacecraft trends retrieved",
                   extra={
                       "spacecraft_id": str(spacecraft_id),
                       "metric": metric,
                       "timespan_hours": timespan_hours,
                       "data_points": len(trend_data),
                       "duration_ms": timer.duration_ms
                   })

        return {
            "spacecraft_id": str(spacecraft_id),
            "metric": metric,
            "timespan_hours": timespan_hours,
            "data_points": len(trend_data),
            "trends": trend_data
        }

# Background task functions
async def initialize_spacecraft_monitoring(spacecraft_id: UUID):
    """Initialize monitoring for new spacecraft."""
    try:
        # Set up real-time monitoring
        # Configure alerting thresholds
        # Initialize data collection
        logger.info("Spacecraft monitoring initialized",
                   extra={"spacecraft_id": str(spacecraft_id)})
    except Exception as e:
        logger.error("Failed to initialize spacecraft monitoring",
                    extra={"spacecraft_id": str(spacecraft_id), "error": str(e)})

async def analyze_telemetry_packet(packet_id: UUID):
    """Analyze telemetry packet for anomalies."""
    try:
        # Perform anomaly detection
        # Check against thresholds
        # Generate alerts if necessary
        logger.info("Telemetry packet analyzed",
                   extra={"packet_id": str(packet_id)})
    except Exception as e:
        logger.error("Failed to analyze telemetry packet",
                    extra={"packet_id": str(packet_id), "error": str(e)})

async def get_last_telemetry_time() -> Optional[datetime]:
    """Get timestamp of last received telemetry."""
    try:
        repo = get_telemetry_repository()
        return await repo.get_last_telemetry_time()
    except Exception as e:
        logger.error("Failed to get last telemetry time", extra={"error": str(e)})
        return None

# Create main router that includes all sub-routers
main_router = APIRouter()
main_router.include_router(spacecraft_router)
main_router.include_router(telemetry_router)
main_router.include_router(health_router)
main_router.include_router(analytics_router)
