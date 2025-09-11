"""
Database models for Space Telemetry Operations.

This module provides comprehensive database models with enhanced error handling,
connection management, and data validation for space telemetry operations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, Field, validator
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, JSON,
    UniqueConstraint, Index, CheckConstraint, ForeignKey, Table
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from src.core.exceptions import (
    DatabaseError, ValidationError, ConnectionError as CustomConnectionError,
    handle_exceptions, ErrorSeverity, ErrorCategory
)
from src.core.logging import get_structured_logger, PerformanceTimer, AuditLogger
from src.core.settings import get_settings

# Configure logging
logger = get_structured_logger(__name__)
audit_logger = AuditLogger()

# Base model
Base = declarative_base()

# Enums for data classification
class TelemetryStatus(str, Enum):
    """Status enumeration for telemetry data."""
    NOMINAL = "nominal"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"
    UNKNOWN = "unknown"

class DataQuality(str, Enum):
    """Data quality enumeration."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CORRUPTED = "corrupted"
    MISSING = "missing"

class MissionPhase(str, Enum):
    """Mission phase enumeration."""
    PRELAUNCH = "prelaunch"
    LAUNCH = "launch"
    EARLY_ORBIT = "early_orbit"
    NOMINAL_OPERATIONS = "nominal_operations"
    EXTENDED_MISSION = "extended_mission"
    END_OF_LIFE = "end_of_life"

# Association tables for many-to-many relationships
spacecraft_subsystem_association = Table(
    'spacecraft_subsystem_association',
    Base.metadata,
    Column('spacecraft_id', PostgresUUID(as_uuid=True), ForeignKey('spacecraft.id')),
    Column('subsystem_id', PostgresUUID(as_uuid=True), ForeignKey('subsystems.id')),
    UniqueConstraint('spacecraft_id', 'subsystem_id')
)

# Database Models
class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                       onupdate=lambda: datetime.now(timezone.utc))

class Spacecraft(Base, TimestampMixin):
    """Spacecraft model with enhanced validation and metadata."""

    __tablename__ = 'spacecraft'

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False, unique=True)
    mission_name = Column(String(150), nullable=False)
    spacecraft_id = Column(String(50), nullable=False, unique=True)  # Official ID
    mission_phase = Column(String(20), nullable=False, default=MissionPhase.PRELAUNCH.value)
    launch_date = Column(DateTime(timezone=True))
    end_of_mission_date = Column(DateTime(timezone=True))

    # Technical specifications
    mass_kg = Column(Float, CheckConstraint('mass_kg > 0'))
    power_watts = Column(Float, CheckConstraint('power_watts > 0'))
    orbital_period_minutes = Column(Float, CheckConstraint('orbital_period_minutes > 0'))

    # Status and configuration
    is_active = Column(Boolean, default=True, nullable=False)
    communication_frequency_mhz = Column(Float)
    ground_station_contact_schedule = Column(JSONB)

    # Metadata
    metadata_json = Column(JSONB, default={})
    description = Column(Text)
    manufacturer = Column(String(100))

    # Relationships
    subsystems = relationship("Subsystem", secondary=spacecraft_subsystem_association, back_populates="spacecraft")
    telemetry_packets = relationship("TelemetryPacket", back_populates="spacecraft")
    spacecraft_states = relationship("SpacecraftState", back_populates="spacecraft")

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('launch_date IS NULL OR end_of_mission_date IS NULL OR launch_date < end_of_mission_date'),
        Index('idx_spacecraft_name', 'name'),
        Index('idx_spacecraft_id', 'spacecraft_id'),
        Index('idx_spacecraft_active', 'is_active'),
        Index('idx_spacecraft_mission_phase', 'mission_phase'),
    )

    def __repr__(self) -> str:
        return f"<Spacecraft(name='{self.name}', id='{self.spacecraft_id}', phase='{self.mission_phase}')>"

class Subsystem(Base, TimestampMixin):
    """Spacecraft subsystem model."""

    __tablename__ = 'subsystems'

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    subsystem_type = Column(String(50), nullable=False)
    description = Column(Text)

    # Status and health
    is_critical = Column(Boolean, default=False)
    current_status = Column(String(20), default=TelemetryStatus.NOMINAL.value)
    last_health_check = Column(DateTime(timezone=True))

    # Configuration
    telemetry_parameters = Column(JSONB, default=[])
    alert_thresholds = Column(JSONB, default={})
    metadata_json = Column(JSONB, default={})

    # Relationships
    spacecraft = relationship("Spacecraft", secondary=spacecraft_subsystem_association, back_populates="subsystems")
    telemetry_packets = relationship("TelemetryPacket", back_populates="subsystem")

    # Indexes
    __table_args__ = (
        Index('idx_subsystem_name', 'name'),
        Index('idx_subsystem_type', 'subsystem_type'),
        Index('idx_subsystem_status', 'current_status'),
        Index('idx_subsystem_critical', 'is_critical'),
    )

    def __repr__(self) -> str:
        return f"<Subsystem(name='{self.name}', type='{self.subsystem_type}', status='{self.current_status}')>"

class TelemetryPacket(Base, TimestampMixin):
    """Enhanced telemetry packet model with comprehensive validation."""

    __tablename__ = 'telemetry_packets'

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)

    # Timing information
    spacecraft_time = Column(DateTime(timezone=True), nullable=False)
    ground_received_time = Column(DateTime(timezone=True), nullable=False)
    processing_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Packet identification
    packet_id = Column(String(50), nullable=False)
    sequence_number = Column(Integer, nullable=False)
    apid = Column(Integer, nullable=False)  # Application Process ID
    packet_type = Column(String(20), nullable=False)

    # Data content
    raw_data = Column(Text, nullable=False)  # Hex or base64 encoded
    processed_data = Column(JSONB, nullable=False)

    # Quality and validation
    data_quality = Column(String(20), default=DataQuality.HIGH.value)
    validation_status = Column(String(20), default="pending")
    checksum_valid = Column(Boolean)

    # Status and flags
    telemetry_status = Column(String(20), default=TelemetryStatus.NOMINAL.value)
    anomaly_flags = Column(JSONB, default=[])
    alert_level = Column(Integer, default=0)

    # Size and performance metrics
    packet_size_bytes = Column(Integer, nullable=False)
    processing_duration_ms = Column(Float)

    # Foreign keys
    spacecraft_id = Column(PostgresUUID(as_uuid=True), ForeignKey('spacecraft.id'), nullable=False)
    subsystem_id = Column(PostgresUUID(as_uuid=True), ForeignKey('subsystems.id'))

    # Relationships
    spacecraft = relationship("Spacecraft", back_populates="telemetry_packets")
    subsystem = relationship("Subsystem", back_populates="telemetry_packets")

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('spacecraft_id', 'packet_id', 'sequence_number'),
        CheckConstraint('ground_received_time >= spacecraft_time - interval \'1 day\''),
        CheckConstraint('processing_time >= ground_received_time'),
        CheckConstraint('alert_level >= 0 AND alert_level <= 5'),
        CheckConstraint('packet_size_bytes > 0'),
        Index('idx_telemetry_spacecraft_time', 'spacecraft_time'),
        Index('idx_telemetry_ground_time', 'ground_received_time'),
        Index('idx_telemetry_spacecraft_id', 'spacecraft_id'),
        Index('idx_telemetry_subsystem_id', 'subsystem_id'),
        Index('idx_telemetry_status', 'telemetry_status'),
        Index('idx_telemetry_quality', 'data_quality'),
        Index('idx_telemetry_alert_level', 'alert_level'),
        Index('idx_telemetry_apid', 'apid'),
        Index('idx_telemetry_packet_type', 'packet_type'),
        # Composite indexes for common queries
        Index('idx_telemetry_spacecraft_status_time', 'spacecraft_id', 'telemetry_status', 'spacecraft_time'),
        Index('idx_telemetry_subsystem_time', 'subsystem_id', 'spacecraft_time'),
    )

    def __repr__(self) -> str:
        return f"<TelemetryPacket(id='{self.packet_id}', seq={self.sequence_number}, status='{self.telemetry_status}')>"

class SpacecraftState(Base, TimestampMixin):
    """Spacecraft state snapshots for trend analysis."""

    __tablename__ = 'spacecraft_states'

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)

    # Timing
    state_time = Column(DateTime(timezone=True), nullable=False)

    # Position and orientation
    position_x_km = Column(Float)
    position_y_km = Column(Float)
    position_z_km = Column(Float)
    velocity_x_kms = Column(Float)
    velocity_y_kms = Column(Float)
    velocity_z_kms = Column(Float)

    # Attitude
    quaternion_q0 = Column(Float)
    quaternion_q1 = Column(Float)
    quaternion_q2 = Column(Float)
    quaternion_q3 = Column(Float)

    # Power and thermal
    battery_level_percent = Column(Float, CheckConstraint('battery_level_percent >= 0 AND battery_level_percent <= 100'))
    solar_panel_voltage = Column(Float)
    temperature_celsius = Column(Float, CheckConstraint('temperature_celsius >= -273.15'))

    # Communication
    signal_strength_dbm = Column(Float)
    communication_status = Column(String(20), default="nominal")

    # Overall health
    overall_health_score = Column(Float, CheckConstraint('overall_health_score >= 0 AND overall_health_score <= 1'))
    critical_alerts_count = Column(Integer, default=0)

    # Metadata
    state_data = Column(JSONB, default={})

    # Foreign key
    spacecraft_id = Column(PostgresUUID(as_uuid=True), ForeignKey('spacecraft.id'), nullable=False)

    # Relationship
    spacecraft = relationship("Spacecraft", back_populates="spacecraft_states")

    # Indexes
    __table_args__ = (
        Index('idx_state_spacecraft_time', 'spacecraft_id', 'state_time'),
        Index('idx_state_time', 'state_time'),
        Index('idx_state_health_score', 'overall_health_score'),
        Index('idx_state_battery_level', 'battery_level_percent'),
        Index('idx_state_comm_status', 'communication_status'),
    )

    def __repr__(self) -> str:
        return f"<SpacecraftState(spacecraft_id='{self.spacecraft_id}', time='{self.state_time}', health={self.overall_health_score})>"

# Pydantic models for API
class SpacecraftCreate(BaseModel):
    """Pydantic model for spacecraft creation."""
    name: str = Field(..., min_length=1, max_length=100)
    mission_name: str = Field(..., min_length=1, max_length=150)
    spacecraft_id: str = Field(..., min_length=1, max_length=50)
    mission_phase: MissionPhase = MissionPhase.PRELAUNCH
    launch_date: Optional[datetime] = None
    end_of_mission_date: Optional[datetime] = None
    mass_kg: Optional[float] = Field(None, gt=0)
    power_watts: Optional[float] = Field(None, gt=0)
    orbital_period_minutes: Optional[float] = Field(None, gt=0)
    communication_frequency_mhz: Optional[float] = None
    description: Optional[str] = None
    manufacturer: Optional[str] = Field(None, max_length=100)
    metadata_json: Optional[Dict[str, Any]] = {}

    @validator('end_of_mission_date')
    def validate_mission_dates(cls, v, values):
        if v and 'launch_date' in values and values['launch_date'] and v <= values['launch_date']:
            raise ValueError('End of mission date must be after launch date')
        return v

class TelemetryPacketCreate(BaseModel):
    """Pydantic model for telemetry packet creation."""
    spacecraft_time: datetime
    ground_received_time: datetime
    packet_id: str = Field(..., min_length=1, max_length=50)
    sequence_number: int = Field(..., ge=0)
    apid: int = Field(..., ge=0, le=2047)  # 11-bit APID
    packet_type: str = Field(..., min_length=1, max_length=20)
    raw_data: str = Field(..., min_length=1)
    processed_data: Dict[str, Any]
    spacecraft_id: UUID
    subsystem_id: Optional[UUID] = None
    data_quality: DataQuality = DataQuality.HIGH
    telemetry_status: TelemetryStatus = TelemetryStatus.NOMINAL
    packet_size_bytes: int = Field(..., gt=0)
    checksum_valid: Optional[bool] = None

    @validator('ground_received_time')
    def validate_receive_time(cls, v, values):
        if 'spacecraft_time' in values:
            # Allow some tolerance for time differences
            time_diff = (v - values['spacecraft_time']).total_seconds()
            if time_diff < -86400:  # More than 1 day in the past
                raise ValueError('Ground received time cannot be more than 1 day before spacecraft time')
        return v

# Database Connection Manager
class DatabaseManager:
    """Enhanced database connection manager with connection pooling and error handling."""

    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.async_session_maker = None
        self._connection_pool_size = 20
        self._max_overflow = 30
        self._pool_timeout = 30

    @handle_exceptions(
        default_return=None,
        error_category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.HIGH
    )
    async def initialize(self) -> None:
        """Initialize database engine and session maker."""
        with PerformanceTimer() as timer:
            try:
                # Create async engine with connection pooling
                self.engine = create_async_engine(
                    self.settings.async_database_url,
                    echo=self.settings.database_echo,
                    pool_size=self._connection_pool_size,
                    max_overflow=self._max_overflow,
                    pool_timeout=self._pool_timeout,
                    pool_pre_ping=True,  # Validate connections before use
                    pool_recycle=3600,   # Recycle connections every hour
                )

                # Create session maker
                self.async_session_maker = async_sessionmaker(
                    bind=self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )

                # Test connection
                async with self.engine.begin() as conn:
                    await conn.execute("SELECT 1")

                logger.info("Database initialized successfully",
                          extra={"duration_ms": timer.duration_ms})

                # Log audit event
                audit_logger.log_system_event(
                    event_type="database_initialized",
                    details={"pool_size": self._connection_pool_size}
                )

            except Exception as e:
                logger.error("Failed to initialize database",
                           extra={"error": str(e), "duration_ms": timer.duration_ms})
                raise DatabaseError(f"Database initialization failed: {str(e)}")

    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper error handling."""
        if not self.async_session_maker:
            raise DatabaseError("Database not initialized")

        session = self.async_session_maker()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", extra={"error": str(e)})
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            await session.close()

    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise DatabaseError("Database engine not initialized")

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database tables created successfully")

        except Exception as e:
            logger.error("Failed to create database tables", extra={"error": str(e)})
            raise DatabaseError(f"Table creation failed: {str(e)}")

    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

# Repository classes for data access
class SpacecraftRepository:
    """Repository for spacecraft data operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    @handle_exceptions(
        default_return=None,
        error_category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.MEDIUM
    )
    async def create_spacecraft(self, spacecraft_data: SpacecraftCreate) -> Optional[Spacecraft]:
        """Create a new spacecraft."""
        async with self.db_manager.get_session() as session:
            spacecraft = Spacecraft(**spacecraft_data.dict())
            session.add(spacecraft)
            await session.flush()
            await session.refresh(spacecraft)

            audit_logger.log_data_event(
                event_type="spacecraft_created",
                entity_id=str(spacecraft.id),
                details={"name": spacecraft.name, "mission": spacecraft.mission_name}
            )

            return spacecraft

    async def get_spacecraft_by_id(self, spacecraft_id: UUID) -> Optional[Spacecraft]:
        """Get spacecraft by ID."""
        async with self.db_manager.get_session() as session:
            result = await session.get(Spacecraft, spacecraft_id)
            return result

    async def get_spacecraft_by_name(self, name: str) -> Optional[Spacecraft]:
        """Get spacecraft by name."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Spacecraft).where(Spacecraft.name == name)
            )
            return result.scalar_one_or_none()

    async def get_active_spacecraft(self) -> List[Spacecraft]:
        """Get all active spacecraft."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Spacecraft).where(Spacecraft.is_active == True).order_by(Spacecraft.name)
            )
            return result.scalars().all()

    async def list_spacecraft(
        self,
        active_only: bool = True,
        mission_phase: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Spacecraft]:
        """List spacecraft with filtering."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select

            query = select(Spacecraft)

            if active_only:
                query = query.where(Spacecraft.is_active == True)

            if mission_phase:
                query = query.where(Spacecraft.mission_phase == mission_phase)

            query = query.order_by(Spacecraft.name).offset(offset).limit(limit)

            result = await session.execute(query)
            return result.scalars().all()

    async def get_spacecraft_states(
        self,
        spacecraft_id: UUID,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[SpacecraftState]:
        """Get spacecraft state history."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select

            query = select(SpacecraftState).where(
                SpacecraftState.spacecraft_id == spacecraft_id,
                SpacecraftState.state_time >= start_time,
                SpacecraftState.state_time <= end_time
            ).order_by(SpacecraftState.state_time.desc()).limit(limit)

            result = await session.execute(query)
            return result.scalars().all()

    async def get_metric_trends(
        self,
        spacecraft_id: UUID,
        metric: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get metric trends for spacecraft."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select, text

            # Build dynamic query based on metric
            metric_column = getattr(SpacecraftState, metric, None)
            if not metric_column:
                return []

            query = select(
                SpacecraftState.state_time,
                metric_column
            ).where(
                SpacecraftState.spacecraft_id == spacecraft_id,
                SpacecraftState.state_time >= start_time,
                SpacecraftState.state_time <= end_time
            ).order_by(SpacecraftState.state_time)

            result = await session.execute(query)
            rows = result.all()

            return [
                {
                    "timestamp": row[0].isoformat(),
                    "value": row[1]
                }
                for row in rows
                if row[1] is not None
            ]

class TelemetryRepository:
    """Repository for telemetry data operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    @handle_exceptions(
        default_return=None,
        error_category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.MEDIUM
    )
    async def store_telemetry_packet(self, packet_data: TelemetryPacketCreate) -> Optional[TelemetryPacket]:
        """Store telemetry packet with validation."""
        async with self.db_manager.get_session() as session:
            packet = TelemetryPacket(**packet_data.dict())
            packet.processing_time = datetime.now(timezone.utc)

            session.add(packet)
            await session.flush()
            await session.refresh(packet)

            # Log for monitoring
            logger.info("Telemetry packet stored",
                       extra={
                           "packet_id": packet.packet_id,
                           "spacecraft_id": str(packet.spacecraft_id),
                           "status": packet.telemetry_status
                       })

            return packet

    async def list_telemetry_packets(
        self,
        spacecraft_id: Optional[UUID] = None,
        subsystem_id: Optional[UUID] = None,
        status: Optional[str] = None,
        quality: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TelemetryPacket]:
        """List telemetry packets with filtering."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select

            query = select(TelemetryPacket)

            if spacecraft_id:
                query = query.where(TelemetryPacket.spacecraft_id == spacecraft_id)

            if subsystem_id:
                query = query.where(TelemetryPacket.subsystem_id == subsystem_id)

            if status:
                query = query.where(TelemetryPacket.telemetry_status == status)

            if quality:
                query = query.where(TelemetryPacket.data_quality == quality)

            if start_time:
                query = query.where(TelemetryPacket.spacecraft_time >= start_time)

            if end_time:
                query = query.where(TelemetryPacket.spacecraft_time <= end_time)

            query = query.order_by(TelemetryPacket.spacecraft_time.desc()).offset(offset).limit(limit)

            result = await session.execute(query)
            return result.scalars().all()

    async def get_telemetry_statistics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get telemetry processing statistics."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select, func, text

            # Total packets
            total_query = select(func.count(TelemetryPacket.id))
            total_result = await session.execute(total_query)
            total_packets = total_result.scalar() or 0

            # Packets in time range
            range_query = select(func.count(TelemetryPacket.id)).where(
                TelemetryPacket.processing_time >= start_time,
                TelemetryPacket.processing_time <= end_time
            )
            range_result = await session.execute(range_query)
            packets_in_range = range_result.scalar() or 0

            # Last hour
            hour_ago = end_time - timedelta(hours=1)
            hour_query = select(func.count(TelemetryPacket.id)).where(
                TelemetryPacket.processing_time >= hour_ago,
                TelemetryPacket.processing_time <= end_time
            )
            hour_result = await session.execute(hour_query)
            packets_last_hour = hour_result.scalar() or 0

            # Average processing time
            avg_query = select(func.avg(TelemetryPacket.processing_duration_ms)).where(
                TelemetryPacket.processing_time >= start_time,
                TelemetryPacket.processing_time <= end_time
            )
            avg_result = await session.execute(avg_query)
            avg_processing_time = avg_result.scalar() or 0.0

            # Quality distribution
            quality_query = select(
                TelemetryPacket.data_quality,
                func.count(TelemetryPacket.id)
            ).where(
                TelemetryPacket.processing_time >= start_time,
                TelemetryPacket.processing_time <= end_time
            ).group_by(TelemetryPacket.data_quality)

            quality_result = await session.execute(quality_query)
            quality_distribution = {row[0]: row[1] for row in quality_result.all()}

            # Status distribution
            status_query = select(
                TelemetryPacket.telemetry_status,
                func.count(TelemetryPacket.id)
            ).where(
                TelemetryPacket.processing_time >= start_time,
                TelemetryPacket.processing_time <= end_time
            ).group_by(TelemetryPacket.telemetry_status)

            status_result = await session.execute(status_query)
            status_distribution = {row[0]: row[1] for row in status_result.all()}

            # Error rate
            error_count = status_distribution.get('error', 0)
            error_rate = (error_count / packets_in_range * 100) if packets_in_range > 0 else 0.0

            return {
                'total_packets': total_packets,
                'packets_last_hour': packets_last_hour,
                'packets_last_24h': packets_in_range,
                'avg_processing_time': float(avg_processing_time),
                'error_rate': error_rate,
                'quality_distribution': quality_distribution,
                'status_distribution': status_distribution
            }

    async def get_last_telemetry_time(self) -> Optional[datetime]:
        """Get timestamp of last received telemetry."""
        async with self.db_manager.get_session() as session:
            from sqlalchemy import select, func

            query = select(func.max(TelemetryPacket.ground_received_time))
            result = await session.execute(query)
            return result.scalar()

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
async def init_database() -> None:
    """Initialize database connection."""
    await db_manager.initialize()

async def create_all_tables() -> None:
    """Create all database tables."""
    await db_manager.create_tables()

async def close_database() -> None:
    """Close database connections."""
    await db_manager.close()

def get_spacecraft_repository() -> SpacecraftRepository:
    """Get spacecraft repository instance."""
    return SpacecraftRepository(db_manager)

def get_telemetry_repository() -> TelemetryRepository:
    """Get telemetry repository instance."""
    return TelemetryRepository(db_manager)
