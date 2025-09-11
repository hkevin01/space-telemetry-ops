"""
Enhanced telemetry processing module with comprehensive time handling and boundary conditions.

This module provides:
- Precise time measurement and synchronization
- Boundary condition validation
- Data persistence with recovery mechanisms
- Memory-safe processing for large datasets
- Graceful handling of nominal and off-nominal situations
"""

import asyncio
import struct
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

import numpy as np
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..core.logging import logger, PerformanceTimer, audit
from ..core.exceptions import (
    TelemetryProcessingError, ValidationError, DatabaseError,
    handle_errors, graceful_degradation
)


class TimeUnit(Enum):
    """Supported time measurement units for telemetry processing."""
    NANOSECONDS = "ns"
    MICROSECONDS = "μs"
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"
    DAYS = "d"


class TelemetryQuality(Enum):
    """Telemetry data quality indicators."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    POOR = "poor"
    INVALID = "invalid"


class ProcessingStatus(Enum):
    """Status of telemetry processing operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DISCARDED = "discarded"


@dataclass
class TimeContext:
    """Comprehensive time context for telemetry data."""
    spacecraft_time: datetime
    ground_time: datetime
    received_time: datetime
    processing_time: Optional[datetime] = None

    # Time synchronization data
    time_offset_ms: Optional[float] = None
    clock_drift_rate: Optional[float] = None  # seconds per second
    time_quality: Optional[str] = None

    # Mission-specific time references
    mission_elapsed_time: Optional[timedelta] = None
    orbital_period: Optional[timedelta] = None
    phase_of_mission: Optional[str] = None

    def __post_init__(self):
        """Calculate derived time values."""
        if self.processing_time is None:
            self.processing_time = datetime.now(timezone.utc)

        # Calculate time offset if not provided
        if self.time_offset_ms is None:
            delta = (self.ground_time - self.spacecraft_time).total_seconds() * 1000
            self.time_offset_ms = delta

    def to_timestamp(self, time_unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """Convert spacecraft time to timestamp in specified unit."""
        epoch_time = self.spacecraft_time.timestamp()

        if time_unit == TimeUnit.NANOSECONDS:
            return epoch_time * 1_000_000_000
        elif time_unit == TimeUnit.MICROSECONDS:
            return epoch_time * 1_000_000
        elif time_unit == TimeUnit.MILLISECONDS:
            return epoch_time * 1000
        elif time_unit == TimeUnit.SECONDS:
            return epoch_time
        elif time_unit == TimeUnit.MINUTES:
            return epoch_time / 60
        elif time_unit == TimeUnit.HOURS:
            return epoch_time / 3600
        elif time_unit == TimeUnit.DAYS:
            return epoch_time / 86400
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")

    def is_time_synchronized(self, tolerance_ms: float = 1000.0) -> bool:
        """Check if spacecraft time is synchronized within tolerance."""
        return abs(self.time_offset_ms or 0) <= tolerance_ms

    def get_latency_ms(self) -> float:
        """Calculate end-to-end latency in milliseconds."""
        if self.processing_time:
            return (self.processing_time - self.spacecraft_time).total_seconds() * 1000
        return (self.received_time - self.spacecraft_time).total_seconds() * 1000


class TelemetryPacket(BaseModel):
    """Enhanced telemetry packet with comprehensive validation and metadata."""

    # Packet identification
    packet_id: str = Field(..., description="Unique packet identifier")
    sequence_number: int = Field(..., ge=0, le=2**32-1, description="Packet sequence number")
    vehicle_id: str = Field(..., min_length=1, max_length=50, description="Vehicle/spacecraft identifier")

    # Time information
    spacecraft_time: datetime = Field(..., description="Spacecraft timestamp")
    ground_time: datetime = Field(..., description="Ground receipt timestamp")

    # Payload data
    payload: Dict[str, Any] = Field(..., description="Telemetry payload data")
    packet_type: str = Field(..., min_length=1, max_length=50, description="Type of telemetry packet")

    # Quality and validation
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality score")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    checksum: Optional[str] = Field(None, description="Packet checksum for integrity verification")

    # Processing metadata
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    processing_attempts: int = Field(default=0, ge=0, le=10)
    last_error: Optional[str] = Field(None, description="Last processing error")

    # Mission context
    mission_phase: Optional[str] = Field(None, description="Current mission phase")
    orbital_position: Optional[Dict[str, float]] = Field(None, description="Orbital position data")

    @validator("spacecraft_time", "ground_time")
    def validate_timestamps(cls, v):
        """Validate timestamp is reasonable."""
        now = datetime.now(timezone.utc)
        # Allow timestamps from 10 years ago to 1 day in the future
        min_time = now - timedelta(days=3650)
        max_time = now + timedelta(days=1)

        if v < min_time or v > max_time:
            raise ValueError(f"Timestamp {v} is outside acceptable range [{min_time}, {max_time}]")
        return v

    @validator("payload")
    def validate_payload_size(cls, v):
        """Validate payload is not too large."""
        payload_str = json.dumps(v)
        max_size = 1024 * 1024  # 1MB
        if len(payload_str.encode('utf-8')) > max_size:
            raise ValueError(f"Payload size exceeds maximum of {max_size} bytes")
        return v

    def calculate_checksum(self) -> str:
        """Calculate CRC32 checksum for packet integrity."""
        import zlib

        # Create deterministic representation
        data = {
            "packet_id": self.packet_id,
            "sequence_number": self.sequence_number,
            "vehicle_id": self.vehicle_id,
            "spacecraft_time": self.spacecraft_time.isoformat(),
            "payload": json.dumps(self.payload, sort_keys=True)
        }

        data_str = json.dumps(data, sort_keys=True)
        checksum = zlib.crc32(data_str.encode('utf-8')) & 0xffffffff
        return f"{checksum:08x}"

    def verify_integrity(self) -> bool:
        """Verify packet integrity using checksum."""
        if not self.checksum:
            return True  # No checksum to verify

        calculated = self.calculate_checksum()
        return calculated == self.checksum

    def get_time_context(self) -> TimeContext:
        """Get comprehensive time context for this packet."""
        return TimeContext(
            spacecraft_time=self.spacecraft_time,
            ground_time=self.ground_time,
            received_time=datetime.now(timezone.utc)
        )

    def to_processing_dict(self) -> Dict[str, Any]:
        """Convert packet to dictionary for processing."""
        return {
            "packet_id": self.packet_id,
            "sequence_number": self.sequence_number,
            "vehicle_id": self.vehicle_id,
            "spacecraft_time": self.spacecraft_time,
            "ground_time": self.ground_time,
            "payload": self.payload,
            "packet_type": self.packet_type,
            "quality_score": self.quality_score,
            "checksum": self.checksum,
            "processing_status": self.processing_status.value,
            "processing_attempts": self.processing_attempts
        }


class BoundaryValidator:
    """Validates telemetry data against boundary conditions and operational limits."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration."""
        default_rules = {
            "temperature": {"min": -273.15, "max": 1000.0, "unit": "celsius"},
            "pressure": {"min": 0.0, "max": 1000.0, "unit": "bar"},
            "voltage": {"min": -50.0, "max": 50.0, "unit": "volts"},
            "current": {"min": -100.0, "max": 100.0, "unit": "amperes"},
            "altitude": {"min": -1000.0, "max": 1000000.0, "unit": "meters"},
            "velocity": {"min": -50000.0, "max": 50000.0, "unit": "m/s"},
            "acceleration": {"min": -1000.0, "max": 1000.0, "unit": "m/s^2"},
            "angular_velocity": {"min": -1000.0, "max": 1000.0, "unit": "rad/s"},
        }

        return {**default_rules, **self.config.get("validation_rules", {})}

    def validate_packet(self, packet: TelemetryPacket) -> Tuple[bool, List[str]]:
        """
        Validate telemetry packet against boundary conditions.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Validate packet integrity
            if not packet.verify_integrity():
                errors.append("Packet integrity check failed")

            # Validate payload values
            payload_errors = self._validate_payload_boundaries(packet.payload)
            errors.extend(payload_errors)

            # Validate time consistency
            time_errors = self._validate_time_boundaries(packet)
            errors.extend(time_errors)

            # Validate sequence number
            seq_errors = self._validate_sequence_boundaries(packet)
            errors.extend(seq_errors)

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Validation error for packet {packet.packet_id}: {str(e)}")
            return False, [f"Validation exception: {str(e)}"]

    def _validate_payload_boundaries(self, payload: Dict[str, Any]) -> List[str]:
        """Validate payload values against defined boundaries."""
        errors = []

        for key, value in payload.items():
            if not isinstance(value, (int, float)):
                continue

            # Check if we have validation rules for this parameter
            rule = self.validation_rules.get(key)
            if not rule:
                continue

            # Validate against min/max bounds
            if "min" in rule and value < rule["min"]:
                errors.append(
                    f"Parameter '{key}' value {value} below minimum {rule['min']} {rule.get('unit', '')}"
                )

            if "max" in rule and value > rule["max"]:
                errors.append(
                    f"Parameter '{key}' value {value} above maximum {rule['max']} {rule.get('unit', '')}"
                )

            # Check for NaN or infinite values
            if math.isnan(value):
                errors.append(f"Parameter '{key}' is NaN")
            elif math.isinf(value):
                errors.append(f"Parameter '{key}' is infinite")

        return errors

    def _validate_time_boundaries(self, packet: TelemetryPacket) -> List[str]:
        """Validate time-related boundaries."""
        errors = []

        # Check time ordering
        if packet.ground_time < packet.spacecraft_time - timedelta(seconds=300):
            errors.append("Ground time significantly before spacecraft time (>5 minutes)")

        # Check for future timestamps (beyond reasonable clock skew)
        now = datetime.now(timezone.utc)
        if packet.spacecraft_time > now + timedelta(minutes=10):
            errors.append("Spacecraft time is too far in the future")

        # Check for very old timestamps
        if packet.spacecraft_time < now - timedelta(days=30):
            errors.append("Spacecraft time is too old (>30 days)")

        return errors

    def _validate_sequence_boundaries(self, packet: TelemetryPacket) -> List[str]:
        """Validate sequence number boundaries."""
        errors = []

        # Sequence number should be within valid range
        if packet.sequence_number < 0:
            errors.append("Sequence number cannot be negative")
        elif packet.sequence_number > 2**32 - 1:
            errors.append("Sequence number exceeds maximum value")

        return errors


class MemoryManagedProcessor:
    """Memory-aware telemetry processor with automatic garbage collection."""

    def __init__(
        self,
        max_memory_mb: int = 1024,
        batch_size: int = 1000,
        enable_gc: bool = True
    ):
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        self.processed_count = 0
        self.memory_warnings = 0

        # Weak references to track processed packets
        self.packet_registry = weakref.WeakSet()

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def check_memory_pressure(self) -> bool:
        """Check if memory usage is approaching limits."""
        current_memory = self.get_memory_usage_mb()
        return current_memory > (self.max_memory_mb * 0.8)  # 80% threshold

    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        if not self.enable_gc:
            return {"collected": 0, "generation": -1}

        collected_counts = []
        for generation in range(gc.get_count().__len__()):
            collected = gc.collect(generation)
            collected_counts.append(collected)

        total_collected = sum(collected_counts)
        logger.info(f"Garbage collection freed {total_collected} objects")

        return {
            "collected": total_collected,
            "generation": len(collected_counts) - 1,
            "by_generation": collected_counts
        }

    async def process_packet_batch(
        self,
        packets: List[TelemetryPacket],
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Process batch of telemetry packets with memory management."""
        batch_start = time.perf_counter()

        # Check memory before processing
        if self.check_memory_pressure():
            self.memory_warnings += 1
            logger.warning(
                f"Memory pressure detected: {self.get_memory_usage_mb():.1f}MB / {self.max_memory_mb}MB"
            )
            gc_stats = self.force_garbage_collection()
            logger.info(f"Forced GC collected {gc_stats['collected']} objects")

        results = {
            "processed": 0,
            "errors": 0,
            "skipped": 0,
            "memory_warnings": 0,
            "processing_time_ms": 0
        }

        validator = BoundaryValidator()

        try:
            # Process packets in smaller chunks to manage memory
            chunk_size = min(self.batch_size, len(packets))

            for i in range(0, len(packets), chunk_size):
                chunk = packets[i:i + chunk_size]

                with PerformanceTimer("packet_chunk_processing") as timer:
                    chunk_results = await self._process_chunk(chunk, validator, session)

                # Accumulate results
                for key in results:
                    if key in chunk_results:
                        results[key] += chunk_results[key]

                # Clear chunk from memory
                del chunk

                # Periodic garbage collection
                if (i // chunk_size) % 10 == 0 and self.enable_gc:
                    gc.collect(0)  # Only collect generation 0 for performance

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            results["errors"] += len(packets)
            raise TelemetryProcessingError(
                f"Batch processing failed: {str(e)}",
                details={"batch_size": len(packets), "memory_mb": self.get_memory_usage_mb()}
            )

        finally:
            batch_time = (time.perf_counter() - batch_start) * 1000
            results["processing_time_ms"] = batch_time

            # Final memory check
            final_memory = self.get_memory_usage_mb()
            if final_memory > self.max_memory_mb:
                results["memory_warnings"] += 1
                logger.error(f"Memory limit exceeded: {final_memory:.1f}MB > {self.max_memory_mb}MB")

        return results

    async def _process_chunk(
        self,
        chunk: List[TelemetryPacket],
        validator: BoundaryValidator,
        session: AsyncSession
    ) -> Dict[str, int]:
        """Process a single chunk of packets."""
        results = {"processed": 0, "errors": 0, "skipped": 0}

        # Validate all packets first
        valid_packets = []
        for packet in chunk:
            try:
                is_valid, errors = validator.validate_packet(packet)
                if is_valid:
                    valid_packets.append(packet)
                else:
                    packet.validation_errors = errors
                    packet.processing_status = ProcessingStatus.FAILED
                    results["errors"] += 1

                    logger.warning(
                        f"Packet validation failed: {packet.packet_id}",
                        extra={"errors": errors, "packet_id": packet.packet_id}
                    )
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Validation exception for packet {packet.packet_id}: {str(e)}")

        # Process valid packets
        if valid_packets:
            try:
                await self._persist_packets(valid_packets, session)
                results["processed"] = len(valid_packets)

                # Add packets to registry for tracking
                for packet in valid_packets:
                    self.packet_registry.add(packet)

            except Exception as e:
                results["errors"] += len(valid_packets)
                logger.error(f"Persistence error: {str(e)}")
                raise

        return results

    async def _persist_packets(self, packets: List[TelemetryPacket], session: AsyncSession):
        """Persist packets to database with error handling."""
        from ..db.models import Telemetry

        try:
            # Prepare batch insert data
            insert_data = []
            for packet in packets:
                insert_data.append({
                    "id": packet.packet_id,
                    "sequence_number": packet.sequence_number,
                    "vehicle_id": packet.vehicle_id,
                    "spacecraft_time": packet.spacecraft_time,
                    "ground_time": packet.ground_time,
                    "payload": packet.payload,
                    "packet_type": packet.packet_type,
                    "quality_score": packet.quality_score,
                    "checksum": packet.checksum,
                    "processing_status": packet.processing_status.value
                })

            # Use PostgreSQL's ON CONFLICT for upsert behavior
            stmt = pg_insert(Telemetry).values(insert_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_=dict(
                    processing_status=stmt.excluded.processing_status,
                    quality_score=stmt.excluded.quality_score,
                    updated_at=datetime.now(timezone.utc)
                )
            )

            await session.execute(stmt)
            await session.commit()

            logger.info(f"Successfully persisted {len(packets)} packets")

        except Exception as e:
            await session.rollback()
            logger.error(f"Database persistence failed: {str(e)}")
            raise DatabaseError(f"Failed to persist telemetry packets: {str(e)}")


class TelemetryProcessor:
    """Main telemetry processing engine with comprehensive error handling."""

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        batch_size: int = 1000,
        memory_limit_mb: int = 2048,
        enable_quality_analysis: bool = True
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        self.enable_quality_analysis = enable_quality_analysis

        # Processing components
        self.memory_processor = MemoryManagedProcessor(
            max_memory_mb=memory_limit_mb,
            batch_size=batch_size
        )
        self.validator = BoundaryValidator()

        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "total_errors": 0,
            "start_time": datetime.now(timezone.utc),
            "last_processed": None,
            "processing_rate": 0.0  # packets per second
        }

        # Task management
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks = set()

    @handle_errors(reraise=False)
    async def process_telemetry_stream(
        self,
        packet_stream: AsyncIterator[TelemetryPacket],
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Process continuous stream of telemetry packets."""
        logger.info("Starting telemetry stream processing")

        batch = []
        batch_count = 0

        try:
            async for packet in packet_stream:
                batch.append(packet)

                # Process batch when full or after timeout
                if len(batch) >= self.batch_size:
                    task = asyncio.create_task(
                        self._process_batch_with_semaphore(batch.copy(), session, batch_count)
                    )
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)

                    batch.clear()
                    batch_count += 1

                    # Limit concurrent tasks
                    if len(self.active_tasks) >= self.max_concurrent_tasks:
                        done, pending = await asyncio.wait(
                            self.active_tasks,
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        self.active_tasks = pending

            # Process remaining packets in final batch
            if batch:
                await self._process_batch_with_semaphore(batch, session, batch_count)

            # Wait for all remaining tasks
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)

            return self._get_processing_summary()

        except Exception as e:
            logger.error(f"Stream processing error: {str(e)}")
            raise TelemetryProcessingError(f"Stream processing failed: {str(e)}")

    async def _process_batch_with_semaphore(
        self,
        batch: List[TelemetryPacket],
        session: AsyncSession,
        batch_id: int
    ):
        """Process batch with concurrency control."""
        async with self.processing_semaphore:
            try:
                with PerformanceTimer(f"batch_{batch_id}") as timer:
                    results = await self.memory_processor.process_packet_batch(batch, session)

                # Update statistics
                self.stats["total_processed"] += results.get("processed", 0)
                self.stats["total_errors"] += results.get("errors", 0)
                self.stats["last_processed"] = datetime.now(timezone.utc)

                # Calculate processing rate
                elapsed_time = (self.stats["last_processed"] - self.stats["start_time"]).total_seconds()
                if elapsed_time > 0:
                    self.stats["processing_rate"] = self.stats["total_processed"] / elapsed_time

                logger.info(
                    f"Batch {batch_id} processed: {results['processed']} packets, "
                    f"{results['errors']} errors, {timer.duration_ms:.2f}ms"
                )

            except Exception as e:
                self.stats["total_errors"] += len(batch)
                logger.error(f"Batch {batch_id} processing failed: {str(e)}")

    def _get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary."""
        now = datetime.now(timezone.utc)
        total_time = (now - self.stats["start_time"]).total_seconds()

        return {
            "total_processed": self.stats["total_processed"],
            "total_errors": self.stats["total_errors"],
            "success_rate": (
                self.stats["total_processed"] /
                (self.stats["total_processed"] + self.stats["total_errors"])
                if (self.stats["total_processed"] + self.stats["total_errors"]) > 0 else 0
            ),
            "processing_rate_pps": self.stats["processing_rate"],
            "total_time_seconds": total_time,
            "start_time": self.stats["start_time"],
            "end_time": now,
            "memory_warnings": self.memory_processor.memory_warnings
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "memory_usage_mb": self.memory_processor.get_memory_usage_mb(),
            "memory_limit_mb": self.memory_limit_mb,
            "active_tasks": len(self.active_tasks),
            "max_tasks": self.max_concurrent_tasks,
            "processing_stats": self.stats
        }

        # Check memory pressure
        if self.memory_processor.check_memory_pressure():
            health_status["status"] = "degraded"
            health_status["warnings"] = ["High memory usage"]

        # Check processing rate
        if self.stats["processing_rate"] < 1.0 and self.stats["total_processed"] > 100:
            health_status["status"] = "degraded"
            health_status.setdefault("warnings", []).append("Low processing rate")

        return health_status


# Export commonly used items
__all__ = [
    "TimeUnit",
    "TelemetryQuality",
    "ProcessingStatus",
    "TimeContext",
    "TelemetryPacket",
    "BoundaryValidator",
    "MemoryManagedProcessor",
    "TelemetryProcessor"
]
