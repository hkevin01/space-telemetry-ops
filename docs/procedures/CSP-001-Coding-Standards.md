# Coding Standards and Procedures

## Space Telemetry Operations System

| Document Information ||
|---|---|
| **Document ID** | CSP-001 |
| **Version** | 1.0 |
| **Date** | December 18, 2024 |
| **Status** | Approved |
| **Classification** | NASA-STD-8739.8 Compliant |

---

## 1. INTRODUCTION

### 1.1 Purpose

This Coding Standards and Procedures document establishes uniform coding practices, style guidelines, and development procedures for the Space Telemetry Operations System. These standards ensure code quality, maintainability, and consistency across all development activities.

### 1.2 Scope

This document applies to all software development activities including:

- Python backend services and APIs
- JavaScript/TypeScript frontend applications
- SQL database scripts and migrations
- Configuration files and infrastructure code
- Documentation and comments
- Testing and validation code

### 1.3 Document Organization

This document follows NASA-STD-8739.8 software development standards with emphasis on code quality, security, and reliability requirements for mission-critical systems.

### 1.4 References

- PEP 8: Style Guide for Python Code
- Airbnb JavaScript Style Guide
- NASA-STD-8739.8: Software Assurance Standard
- OWASP Secure Coding Practices

---

## 2. GENERAL CODING PRINCIPLES

### 2.1 Code Quality Principles

#### 2.1.1 Fundamental Principles

- **Readability**: Code should be self-documenting and easily understood
- **Consistency**: Uniform style and conventions throughout codebase
- **Security**: Security-first approach in all development activities
- **Performance**: Efficient algorithms and resource utilization
- **Reliability**: Robust error handling and fault tolerance
- **Maintainability**: Modular design with clear separation of concerns

#### 2.1.2 SOLID Principles

All code shall adhere to SOLID design principles:

- **Single Responsibility**: Each class/function has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes must be substitutable for base types
- **Interface Segregation**: Many specific interfaces better than one general
- **Dependency Inversion**: Depend on abstractions, not concretions

### 2.2 Security Coding Standards

#### 2.2.1 Security Requirements

All code must implement security best practices:

- **Input Validation**: Validate all inputs at system boundaries
- **Authentication**: Secure authentication and authorization
- **Encryption**: Encrypt sensitive data at rest and in transit
- **Logging**: Comprehensive security event logging
- **Error Handling**: Secure error handling without information leakage

---

## 3. PYTHON CODING STANDARDS

### 3.1 Style Guidelines

#### 3.1.1 PEP 8 Compliance

All Python code shall strictly adhere to PEP 8 guidelines:

```python
# Correct Python style examples

# Imports
import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Class definitions
class TelemetryProcessor:
    """
    Process spacecraft telemetry data with validation and quality checks.

    This class provides comprehensive telemetry processing capabilities
    including packet validation, quality assessment, and data transformation.
    """

    def __init__(self, max_batch_size: int = 1000):
        """Initialize telemetry processor with configuration."""
        self.max_batch_size = max_batch_size
        self.processed_count = 0
        self.error_count = 0

    def process_telemetry_packet(
        self,
        packet: TelemetryPacket
    ) -> ProcessingResult:
        """
        Process a single telemetry packet.

        Args:
            packet: TelemetryPacket instance to process

        Returns:
            ProcessingResult with success status and details

        Raises:
            ValidationError: If packet validation fails
            ProcessingError: If processing encounters an error
        """
        try:
            # Validate input packet
            if not self._validate_packet(packet):
                raise ValidationError("Packet validation failed")

            # Process the packet
            result = self._perform_processing(packet)
            self.processed_count += 1

            return ProcessingResult(
                success=True,
                packet_id=packet.packet_id,
                processing_time=result.processing_time
            )

        except Exception as e:
            self.error_count += 1
            logger.error(
                "Packet processing failed",
                extra={
                    "packet_id": packet.packet_id,
                    "error": str(e),
                    "spacecraft_id": packet.spacecraft_id
                }
            )
            raise ProcessingError(f"Processing failed: {str(e)}") from e
```

#### 3.1.2 Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| **Variables** | snake_case | `telemetry_data`, `packet_count` |
| **Functions** | snake_case | `process_packet()`, `validate_data()` |
| **Classes** | PascalCase | `TelemetryProcessor`, `AnomalyDetector` |
| **Constants** | UPPER_SNAKE_CASE | `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT` |
| **Modules** | snake_case | `telemetry_processor.py`, `data_validator.py` |
| **Packages** | snake_case | `telemetry`, `anomaly_detection` |

#### 3.1.3 Documentation Standards

```python
# Module docstring example
"""
Telemetry Processing Module

This module provides comprehensive telemetry data processing capabilities
for spacecraft operations, including packet validation, quality assessment,
and real-time anomaly detection.

Classes:
    TelemetryProcessor: Main processing engine
    PacketValidator: Validation utilities
    QualityAssessor: Data quality evaluation

Functions:
    create_processor: Factory function for processor instances
    validate_configuration: Configuration validation utility

Example:
    >>> processor = TelemetryProcessor(max_batch_size=1000)
    >>> result = processor.process_packet(packet)
    >>> print(f"Processing time: {result.processing_time}ms")
"""

# Function docstring example
def calculate_anomaly_score(
    data: Dict[str, float],
    baseline: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate anomaly score for telemetry data.

    Computes a weighted anomaly score by comparing current telemetry
    values against established baseline values. Higher scores indicate
    greater deviation from normal operation.

    Args:
        data: Current telemetry parameter values
        baseline: Baseline parameter values for comparison
        weights: Optional parameter weights (default: equal weighting)

    Returns:
        float: Anomaly score between 0.0 (normal) and 1.0 (highly anomalous)

    Raises:
        ValueError: If data and baseline keys don't match
        TypeError: If values are not numeric

    Example:
        >>> current = {"temperature": 25.5, "pressure": 1013.2}
        >>> baseline = {"temperature": 23.0, "pressure": 1013.0}
        >>> score = calculate_anomaly_score(current, baseline)
        >>> print(f"Anomaly score: {score:.3f}")
    """
```

### 3.2 Code Organization

#### 3.2.1 File Structure

```python
# Standard file structure for Python modules
"""Module docstring."""

# Standard library imports
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Third-party imports
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from sqlalchemy import select

# Local imports
from ..core.models import TelemetryPacket
from ..core.exceptions import ValidationError
from .utils import validate_packet_format

# Module-level constants
DEFAULT_BATCH_SIZE = 1000
MAX_RETRY_ATTEMPTS = 3
PROCESSING_TIMEOUT = 30.0

# Module-level logger
logger = logging.getLogger(__name__)

# Class definitions
class TelemetryProcessor:
    # Implementation here
    pass

# Function definitions
def create_processor(config: Dict[str, Any]) -> TelemetryProcessor:
    # Implementation here
    pass

# Module initialization code (if needed)
if __name__ == "__main__":
    # Command-line interface or test code
    pass
```

### 3.3 Error Handling

#### 3.3.1 Exception Hierarchy

```python
# Custom exception hierarchy
class TelemetryError(Exception):
    """Base exception for telemetry processing errors."""

    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.timestamp = datetime.utcnow()

class ValidationError(TelemetryError):
    """Raised when data validation fails."""
    pass

class ProcessingError(TelemetryError):
    """Raised when processing operations fail."""
    pass

class ConfigurationError(TelemetryError):
    """Raised when configuration is invalid."""
    pass

# Exception handling patterns
def process_with_error_handling(packet: TelemetryPacket) -> ProcessingResult:
    """Process packet with comprehensive error handling."""
    try:
        # Attempt processing
        result = process_packet(packet)
        return result

    except ValidationError as e:
        logger.warning(
            "Packet validation failed",
            extra={
                "packet_id": packet.packet_id,
                "error_code": e.error_code,
                "error_message": e.message
            }
        )
        # Handle validation error
        return ProcessingResult(success=False, error=str(e))

    except ProcessingError as e:
        logger.error(
            "Processing error occurred",
            extra={
                "packet_id": packet.packet_id,
                "error_code": e.error_code,
                "error_message": e.message
            }
        )
        # Handle processing error
        raise  # Re-raise for upstream handling

    except Exception as e:
        logger.critical(
            "Unexpected error during processing",
            extra={
                "packet_id": packet.packet_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        # Handle unexpected errors
        raise ProcessingError(f"Unexpected error: {str(e)}") from e
```

### 3.4 Async Programming Standards

#### 3.4.1 Async/Await Patterns

```python
# Proper async/await usage
import asyncio
from typing import AsyncIterator

class AsyncTelemetryProcessor:
    """Async-capable telemetry processor."""

    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def process_packet_stream(
        self,
        packet_stream: AsyncIterator[TelemetryPacket]
    ) -> AsyncIterator[ProcessingResult]:
        """Process stream of telemetry packets asynchronously."""
        tasks = []

        async for packet in packet_stream:
            # Create task with semaphore for concurrency control
            task = asyncio.create_task(
                self._process_packet_with_semaphore(packet)
            )
            tasks.append(task)

            # Yield completed results as they become available
            if len(tasks) >= self.max_concurrent_tasks:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    result = await completed_task
                    yield result

                tasks = list(pending)

        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
            for result in results:
                yield result

    async def _process_packet_with_semaphore(
        self,
        packet: TelemetryPacket
    ) -> ProcessingResult:
        """Process single packet with semaphore control."""
        async with self.semaphore:
            return await self._process_packet_async(packet)

    async def _process_packet_async(
        self,
        packet: TelemetryPacket
    ) -> ProcessingResult:
        """Async packet processing implementation."""
        try:
            # Simulate async processing
            await asyncio.sleep(0.001)  # Non-blocking operation

            # Validation and processing logic
            if not self._validate_packet(packet):
                raise ValidationError("Invalid packet format")

            # Process the packet
            processed_data = await self._perform_async_processing(packet)

            return ProcessingResult(
                success=True,
                packet_id=packet.packet_id,
                data=processed_data
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                packet_id=packet.packet_id,
                error=str(e)
            )
```

---

## 4. JAVASCRIPT/TYPESCRIPT STANDARDS

### 4.1 TypeScript Configuration

#### 4.1.1 TSConfig Standards

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["DOM", "DOM.Iterable", "ES6"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "noImplicitAny": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true
  },
  "include": [
    "src/**/*"
  ],
  "exclude": [
    "node_modules",
    "build",
    "dist"
  ]
}
```

### 4.2 React Component Standards

#### 4.2.1 Component Structure

```typescript
// React component example following standards
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { AlertTriangle, CheckCircle, Clock } from 'lucide-react';

// Type definitions
interface TelemetryData {
  timestamp: string;
  spacecraft_id: string;
  temperature: number;
  pressure: number;
  status: 'nominal' | 'warning' | 'critical';
}

interface TelemetryDisplayProps {
  /** Spacecraft ID to display telemetry for */
  spacecraftId: string;
  /** Refresh interval in milliseconds */
  refreshInterval?: number;
  /** Callback when telemetry data updates */
  onDataUpdate?: (data: TelemetryData) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * TelemetryDisplay Component
 *
 * Displays real-time telemetry data for a specific spacecraft
 * with automatic refresh and status indicators.
 */
const TelemetryDisplay: React.FC<TelemetryDisplayProps> = ({
  spacecraftId,
  refreshInterval = 1000,
  onDataUpdate,
  className = ''
}) => {
  // State management
  const [telemetryData, setTelemetryData] = useState<TelemetryData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Memoized API URL
  const apiUrl = useMemo(
    () => `/api/telemetry/latest?spacecraft_id=${encodeURIComponent(spacecraftId)}`,
    [spacecraftId]
  );

  // Fetch telemetry data
  const fetchTelemetryData = useCallback(async (): Promise<void> => {
    try {
      setError(null);
      const response = await fetch(apiUrl);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: TelemetryData = await response.json();
      setTelemetryData(data);

      // Notify parent component
      if (onDataUpdate) {
        onDataUpdate(data);
      }

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      console.error('Failed to fetch telemetry data:', err);
    } finally {
      setLoading(false);
    }
  }, [apiUrl, onDataUpdate]);

  // Setup automatic refresh
  useEffect(() => {
    // Initial fetch
    fetchTelemetryData();

    // Setup interval for refresh
    const intervalId = setInterval(fetchTelemetryData, refreshInterval);

    // Cleanup interval on unmount
    return (): void => {
      clearInterval(intervalId);
    };
  }, [fetchTelemetryData, refreshInterval]);

  // Status icon component
  const StatusIcon: React.FC<{ status: TelemetryData['status'] }> = ({ status }) => {
    const iconProps = { size: 20, className: 'inline-block mr-2' };

    switch (status) {
      case 'nominal':
        return <CheckCircle {...iconProps} className={`${iconProps.className} text-green-500`} />;
      case 'warning':
        return <AlertTriangle {...iconProps} className={`${iconProps.className} text-yellow-500`} />;
      case 'critical':
        return <AlertTriangle {...iconProps} className={`${iconProps.className} text-red-500`} />;
      default:
        return <Clock {...iconProps} className={`${iconProps.className} text-gray-500`} />;
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className={`telemetry-display loading ${className}`}>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded mb-2"></div>
          <div className="h-4 bg-gray-200 rounded mb-2"></div>
          <div className="h-4 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`telemetry-display error ${className}`}>
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  // No data state
  if (!telemetryData) {
    return (
      <div className={`telemetry-display no-data ${className}`}>
        <div className="text-gray-500">No telemetry data available</div>
      </div>
    );
  }

  // Main render
  return (
    <div className={`telemetry-display ${className}`}>
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">
          Spacecraft {spacecraftId} Telemetry
        </h3>

        <div className="space-y-3">
          <div className="flex items-center">
            <StatusIcon status={telemetryData.status} />
            <span className="font-medium">Status:</span>
            <span className={`ml-2 capitalize ${
              telemetryData.status === 'nominal' ? 'text-green-600' :
              telemetryData.status === 'warning' ? 'text-yellow-600' :
              'text-red-600'
            }`}>
              {telemetryData.status}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="font-medium">Temperature:</span>
              <span className="ml-2">{telemetryData.temperature.toFixed(1)}°C</span>
            </div>
            <div>
              <span className="font-medium">Pressure:</span>
              <span className="ml-2">{telemetryData.pressure.toFixed(1)} kPa</span>
            </div>
          </div>

          <div className="text-sm text-gray-500">
            Last updated: {new Date(telemetryData.timestamp).toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TelemetryDisplay;
```

### 4.3 State Management Standards

#### 4.3.1 Custom Hooks

```typescript
// Custom hook for telemetry data management
import { useState, useEffect, useCallback } from 'react';

interface UseTelemetryOptions {
  refreshInterval?: number;
  autoRefresh?: boolean;
}

interface UseTelemetryReturn {
  data: TelemetryData[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  setAutoRefresh: (enabled: boolean) => void;
}

/**
 * Custom hook for managing telemetry data
 */
export const useTelemetry = (
  spacecraftId: string,
  options: UseTelemetryOptions = {}
): UseTelemetryReturn => {
  const { refreshInterval = 5000, autoRefresh = true } = options;

  const [data, setData] = useState<TelemetryData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState<boolean>(autoRefresh);

  const fetchData = useCallback(async (): Promise<void> => {
    try {
      setError(null);
      const response = await fetch(`/api/telemetry?spacecraft_id=${spacecraftId}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch telemetry data: ${response.statusText}`);
      }

      const result = await response.json();
      setData(result.data || []);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [spacecraftId]);

  useEffect(() => {
    fetchData();

    if (autoRefreshEnabled) {
      const intervalId = setInterval(fetchData, refreshInterval);
      return (): void => clearInterval(intervalId);
    }
  }, [fetchData, refreshInterval, autoRefreshEnabled]);

  return {
    data,
    loading,
    error,
    refresh: fetchData,
    setAutoRefresh: setAutoRefreshEnabled
  };
};
```

---

## 5. DATABASE STANDARDS

### 5.1 SQL Coding Standards

#### 5.1.1 Query Formatting

```sql
-- SQL formatting standards
-- Use uppercase for SQL keywords, lowercase for identifiers

-- Table creation example
CREATE TABLE telemetry_packets (
    packet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    spacecraft_id VARCHAR(50) NOT NULL,
    mission_id VARCHAR(50) NOT NULL,
    telemetry_type VARCHAR(20) NOT NULL,
    quality VARCHAR(20) NOT NULL CHECK (
        quality IN ('EXCELLENT', 'GOOD', 'ACCEPTABLE', 'DEGRADED', 'POOR', 'INVALID')
    ),
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index creation with meaningful names
CREATE INDEX CONCURRENTLY idx_telemetry_packets_timestamp
    ON telemetry_packets USING BTREE (timestamp DESC);

CREATE INDEX CONCURRENTLY idx_telemetry_packets_spacecraft_timestamp
    ON telemetry_packets USING BTREE (spacecraft_id, timestamp DESC);

-- Complex query example
SELECT
    tp.spacecraft_id,
    tp.mission_id,
    COUNT(*) AS packet_count,
    AVG(CAST(tp.data->>'temperature' AS NUMERIC)) AS avg_temperature,
    MAX(tp.timestamp) AS latest_timestamp,
    COUNT(*) FILTER (WHERE tp.quality = 'EXCELLENT') AS excellent_packets,
    COUNT(*) FILTER (WHERE tp.quality IN ('DEGRADED', 'POOR', 'INVALID')) AS problematic_packets
FROM telemetry_packets tp
INNER JOIN spacecraft s ON tp.spacecraft_id = s.spacecraft_id
WHERE tp.timestamp >= NOW() - INTERVAL '24 hours'
    AND tp.telemetry_type = 'sensor'
    AND s.status = 'active'
GROUP BY tp.spacecraft_id, tp.mission_id
HAVING COUNT(*) > 100
ORDER BY avg_temperature DESC, packet_count DESC
LIMIT 50;
```

#### 5.1.2 Migration Scripts

```sql
-- Migration script template
-- Migration: V001__Create_telemetry_tables.sql
-- Description: Initial telemetry database schema
-- Author: Development Team
-- Date: 2024-12-18

BEGIN;

-- Create telemetry_packets table
CREATE TABLE IF NOT EXISTS telemetry_packets (
    packet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    spacecraft_id VARCHAR(50) NOT NULL,
    mission_id VARCHAR(50) NOT NULL,
    telemetry_type VARCHAR(20) NOT NULL,
    quality VARCHAR(20) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_telemetry_quality CHECK (
        quality IN ('EXCELLENT', 'GOOD', 'ACCEPTABLE', 'DEGRADED', 'POOR', 'INVALID')
    ),
    CONSTRAINT chk_telemetry_type CHECK (
        telemetry_type IN ('sensor', 'status', 'command', 'housekeeping')
    )
);

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_telemetry_packets_timestamp
    ON telemetry_packets USING BTREE (timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_telemetry_packets_spacecraft
    ON telemetry_packets USING BTREE (spacecraft_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_telemetry_packets_mission
    ON telemetry_packets USING BTREE (mission_id);

-- Create GIN index for JSONB data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_telemetry_packets_data
    ON telemetry_packets USING GIN (data);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_telemetry_packets_updated_at
    BEFORE UPDATE ON telemetry_packets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE telemetry_packets IS 'Stores telemetry packets from spacecraft';
COMMENT ON COLUMN telemetry_packets.packet_id IS 'Unique identifier for each telemetry packet';
COMMENT ON COLUMN telemetry_packets.timestamp IS 'Timestamp when telemetry was generated';
COMMENT ON COLUMN telemetry_packets.data IS 'JSONB payload containing telemetry parameters';

COMMIT;
```

---

## 6. TESTING STANDARDS

### 6.1 Unit Testing Standards

#### 6.1.1 Python Test Structure

```python
# Unit testing standards for Python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.services.telemetry.processor import TelemetryProcessor
from src.core.models import TelemetryPacket, ProcessingResult
from src.core.exceptions import ValidationError, ProcessingError

class TestTelemetryProcessor:
    """Test suite for TelemetryProcessor class."""

    @pytest.fixture
    def processor(self) -> TelemetryProcessor:
        """Create TelemetryProcessor instance for testing."""
        return TelemetryProcessor(max_batch_size=100)

    @pytest.fixture
    def valid_packet(self) -> TelemetryPacket:
        """Create valid telemetry packet for testing."""
        return TelemetryPacket(
            packet_id="test-001",
            timestamp=datetime.utcnow(),
            spacecraft_id="SAT-001",
            mission_id="MISSION-001",
            telemetry_type="sensor",
            data={"temperature": 23.5, "pressure": 1013.25}
        )

    @pytest.fixture
    def invalid_packet(self) -> TelemetryPacket:
        """Create invalid telemetry packet for testing."""
        return TelemetryPacket(
            packet_id="test-002",
            timestamp=datetime.utcnow(),
            spacecraft_id="",  # Invalid: empty spacecraft_id
            mission_id="MISSION-001",
            telemetry_type="sensor",
            data={}  # Invalid: empty data
        )

    def test_processor_initialization(self, processor: TelemetryProcessor):
        """Test processor initialization."""
        assert processor.max_batch_size == 100
        assert processor.processed_count == 0
        assert processor.error_count == 0

    def test_process_valid_packet(self, processor: TelemetryProcessor, valid_packet: TelemetryPacket):
        """Test processing of valid telemetry packet."""
        # Act
        result = processor.process_packet(valid_packet)

        # Assert
        assert result.success is True
        assert result.packet_id == valid_packet.packet_id
        assert result.error is None
        assert processor.processed_count == 1
        assert processor.error_count == 0

    def test_process_invalid_packet(self, processor: TelemetryProcessor, invalid_packet: TelemetryPacket):
        """Test processing of invalid telemetry packet."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            processor.process_packet(invalid_packet)

        assert "validation failed" in str(exc_info.value).lower()
        assert processor.processed_count == 0
        assert processor.error_count == 1

    @pytest.mark.asyncio
    async def test_async_packet_processing(self, processor: TelemetryProcessor, valid_packet: TelemetryPacket):
        """Test async packet processing."""
        # Mock async dependencies
        with patch.object(processor, '_perform_async_processing', new_callable=AsyncMock) as mock_processing:
            mock_processing.return_value = {"processed": True}

            # Act
            result = await processor.process_packet_async(valid_packet)

            # Assert
            assert result.success is True
            mock_processing.assert_called_once_with(valid_packet)

    def test_batch_processing(self, processor: TelemetryProcessor):
        """Test batch processing functionality."""
        # Arrange
        packets = [
            TelemetryPacket(
                packet_id=f"test-{i:03d}",
                timestamp=datetime.utcnow(),
                spacecraft_id="SAT-001",
                mission_id="MISSION-001",
                telemetry_type="sensor",
                data={"value": i}
            )
            for i in range(10)
        ]

        # Act
        results = processor.process_batch(packets)

        # Assert
        assert len(results) == 10
        assert all(result.success for result in results)
        assert processor.processed_count == 10

    @pytest.mark.parametrize("packet_count,expected_batches", [
        (50, 1),
        (100, 1),
        (150, 2),
        (250, 3)
    ])
    def test_batch_size_handling(self, processor: TelemetryProcessor, packet_count: int, expected_batches: int):
        """Test batch size handling with different packet counts."""
        # Arrange
        packets = [Mock(spec=TelemetryPacket) for _ in range(packet_count)]

        # Mock the batch processing method to count calls
        with patch.object(processor, '_process_single_batch') as mock_batch:
            mock_batch.return_value = [ProcessingResult(success=True) for _ in range(processor.max_batch_size)]

            # Act
            processor.process_batch(packets)

            # Assert
            assert mock_batch.call_count == expected_batches

# Performance test example
@pytest.mark.performance
class TestTelemetryProcessorPerformance:
    """Performance tests for TelemetryProcessor."""

    def test_processing_throughput(self, processor: TelemetryProcessor):
        """Test processing throughput meets requirements."""
        # Arrange
        packet_count = 1000
        packets = [
            TelemetryPacket(
                packet_id=f"perf-{i:04d}",
                timestamp=datetime.utcnow(),
                spacecraft_id="SAT-001",
                mission_id="MISSION-001",
                telemetry_type="sensor",
                data={"value": i}
            )
            for i in range(packet_count)
        ]

        # Act
        start_time = datetime.utcnow()
        results = processor.process_batch(packets)
        end_time = datetime.utcnow()

        # Assert
        processing_time = (end_time - start_time).total_seconds()
        throughput = packet_count / processing_time

        assert len(results) == packet_count
        assert throughput >= 1000  # Minimum 1000 packets/second
        assert all(result.success for result in results)
```

#### 6.1.2 JavaScript/TypeScript Test Structure

```typescript
// Jest/React Testing Library example
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { act } from 'react-dom/test-utils';
import '@testing-library/jest-dom';

import TelemetryDisplay from '../TelemetryDisplay';

// Mock fetch API
global.fetch = jest.fn();

describe('TelemetryDisplay Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const mockTelemetryData = {
    timestamp: '2024-12-18T10:00:00Z',
    spacecraft_id: 'SAT-001',
    temperature: 23.5,
    pressure: 1013.2,
    status: 'nominal' as const
  };

  test('renders loading state initially', () => {
    render(<TelemetryDisplay spacecraftId="SAT-001" />);

    expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();
  });

  test('displays telemetry data after successful fetch', async () => {
    // Arrange
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockTelemetryData
    });

    // Act
    render(<TelemetryDisplay spacecraftId="SAT-001" />);

    // Assert
    await waitFor(() => {
      expect(screen.getByText('Spacecraft SAT-001 Telemetry')).toBeInTheDocument();
      expect(screen.getByText('23.5°C')).toBeInTheDocument();
      expect(screen.getByText('1013.2 kPa')).toBeInTheDocument();
      expect(screen.getByText('nominal')).toBeInTheDocument();
    });
  });

  test('displays error message when fetch fails', async () => {
    // Arrange
    (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    // Act
    render(<TelemetryDisplay spacecraftId="SAT-001" />);

    // Assert
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
      expect(screen.getByText(/network error/i)).toBeInTheDocument();
    });
  });

  test('calls onDataUpdate when data is received', async () => {
    // Arrange
    const mockOnDataUpdate = jest.fn();
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockTelemetryData
    });

    // Act
    render(
      <TelemetryDisplay
        spacecraftId="SAT-001"
        onDataUpdate={mockOnDataUpdate}
      />
    );

    // Assert
    await waitFor(() => {
      expect(mockOnDataUpdate).toHaveBeenCalledWith(mockTelemetryData);
    });
  });

  test('refreshes data at specified interval', async () => {
    // Arrange
    jest.useFakeTimers();
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockTelemetryData
    });

    // Act
    render(<TelemetryDisplay spacecraftId="SAT-001" refreshInterval={1000} />);

    // Initial fetch
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

    // Advance timer
    act(() => {
      jest.advanceTimersByTime(1000);
    });

    // Assert
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));

    jest.useRealTimers();
  });
});
```

---

## 7. SECURITY CODING PRACTICES

### 7.1 Input Validation

#### 7.1.1 Data Validation Patterns

```python
# Input validation examples
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import re

class TelemetryPacketInput(BaseModel):
    """Validated input model for telemetry packets."""

    spacecraft_id: str = Field(..., min_length=1, max_length=50)
    mission_id: str = Field(..., min_length=1, max_length=50)
    telemetry_type: str = Field(..., regex=r'^(sensor|status|command|housekeeping)$')
    data: dict = Field(..., min_items=1)
    timestamp: Optional[datetime] = None

    @validator('spacecraft_id')
    def validate_spacecraft_id(cls, v):
        """Validate spacecraft ID format."""
        if not re.match(r'^[A-Z0-9-]+$', v):
            raise ValueError('Spacecraft ID must contain only uppercase letters, numbers, and hyphens')
        return v

    @validator('data')
    def validate_telemetry_data(cls, v):
        """Validate telemetry data structure."""
        # Check for required fields based on telemetry type
        if not isinstance(v, dict):
            raise ValueError('Telemetry data must be a dictionary')

        # Validate numeric values
        for key, value in v.items():
            if isinstance(value, (int, float)):
                if not (-1000000 <= value <= 1000000):
                    raise ValueError(f'Telemetry value {key} out of acceptable range')

        return v

# API endpoint with validation
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/telemetry")
async def ingest_telemetry(
    packet: TelemetryPacketInput,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Ingest telemetry packet with full validation."""
    try:
        # Validate JWT token
        user = await validate_jwt_token(credentials.credentials)

        # Check permissions
        if not user.has_permission("write:telemetry"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        # Process validated packet
        result = await telemetry_processor.process_packet(packet.dict())

        return {"success": True, "packet_id": result.packet_id}

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Telemetry ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 7.2 SQL Injection Prevention

```python
# Safe database query patterns
import asyncpg
from typing import List, Optional

class TelemetryRepository:
    """Repository with SQL injection prevention."""

    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool

    async def get_telemetry_by_spacecraft(
        self,
        spacecraft_id: str,
        start_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[dict]:
        """Get telemetry data with parameterized queries."""

        # Validate inputs
        if not spacecraft_id or len(spacecraft_id) > 50:
            raise ValueError("Invalid spacecraft_id")

        if limit > 1000:
            limit = 1000  # Prevent excessive data retrieval

        # Use parameterized query to prevent SQL injection
        query = """
            SELECT packet_id, timestamp, spacecraft_id, data
            FROM telemetry_packets
            WHERE spacecraft_id = $1
            AND ($2::timestamptz IS NULL OR timestamp >= $2)
            ORDER BY timestamp DESC
            LIMIT $3
        """

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(query, spacecraft_id, start_time, limit)
            return [dict(row) for row in rows]

    async def insert_telemetry_packet(self, packet: dict) -> str:
        """Insert telemetry packet safely."""

        # Use parameterized insert
        query = """
            INSERT INTO telemetry_packets (
                spacecraft_id, mission_id, telemetry_type, data, timestamp
            ) VALUES ($1, $2, $3, $4, $5)
            RETURNING packet_id
        """

        async with self.pool.acquire() as connection:
            packet_id = await connection.fetchval(
                query,
                packet['spacecraft_id'],
                packet['mission_id'],
                packet['telemetry_type'],
                json.dumps(packet['data']),
                packet.get('timestamp', datetime.utcnow())
            )
            return str(packet_id)
```

---

## 8. PERFORMANCE OPTIMIZATION

### 8.1 Database Query Optimization

```python
# Performance optimization examples
class OptimizedTelemetryQueries:
    """Optimized database queries for high performance."""

    async def get_aggregated_telemetry(
        self,
        spacecraft_ids: List[str],
        time_window_hours: int = 24,
        sample_interval_minutes: int = 5
    ) -> List[dict]:
        """Get aggregated telemetry data with optimized query."""

        # Use time bucketing for efficient aggregation
        query = """
            SELECT
                spacecraft_id,
                time_bucket($1::interval, timestamp) AS time_bucket,
                COUNT(*) as packet_count,
                AVG((data->>'temperature')::numeric) as avg_temperature,
                AVG((data->>'pressure')::numeric) as avg_pressure,
                COUNT(*) FILTER (WHERE quality = 'EXCELLENT') as excellent_count
            FROM telemetry_packets
            WHERE spacecraft_id = ANY($2)
                AND timestamp >= NOW() - $3::interval
                AND data ? 'temperature'
                AND data ? 'pressure'
            GROUP BY spacecraft_id, time_bucket
            ORDER BY spacecraft_id, time_bucket DESC
        """

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                query,
                f"{sample_interval_minutes} minutes",
                spacecraft_ids,
                f"{time_window_hours} hours"
            )
            return [dict(row) for row in rows]

    async def get_latest_telemetry_optimized(
        self,
        spacecraft_id: str
    ) -> Optional[dict]:
        """Get latest telemetry with index optimization."""

        # Optimized query using covering index
        query = """
            SELECT packet_id, timestamp, data
            FROM telemetry_packets
            WHERE spacecraft_id = $1
            ORDER BY timestamp DESC
            LIMIT 1
        """

        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(query, spacecraft_id)
            return dict(row) if row else None
```

### 8.2 Caching Strategies

```python
# Redis caching implementation
import redis.asyncio as redis
import json
from typing import Optional, Union

class TelemetryCacheManager:
    """Efficient caching for telemetry data."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes

    async def get_cached_telemetry(
        self,
        spacecraft_id: str
    ) -> Optional[dict]:
        """Get cached telemetry data."""

        cache_key = f"telemetry:latest:{spacecraft_id}"

        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache read failed: {str(e)}")

        return None

    async def cache_telemetry(
        self,
        spacecraft_id: str,
        data: dict,
        ttl: int = None
    ) -> None:
        """Cache telemetry data with TTL."""

        cache_key = f"telemetry:latest:{spacecraft_id}"
        ttl = ttl or self.default_ttl

        try:
            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {str(e)}")

    async def invalidate_spacecraft_cache(
        self,
        spacecraft_id: str
    ) -> None:
        """Invalidate all cached data for a spacecraft."""

        pattern = f"telemetry:*:{spacecraft_id}"

        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {str(e)}")
```

---

## 9. CODE REVIEW GUIDELINES

### 9.1 Review Checklist

#### 9.1.1 Functional Review

- [ ] **Code correctness**: Logic is correct and handles edge cases
- [ ] **Requirements compliance**: Implementation meets specified requirements
- [ ] **Error handling**: Appropriate exception handling and recovery
- [ ] **Input validation**: All inputs are properly validated
- [ ] **Output verification**: Outputs match expected formats and ranges

#### 9.1.2 Quality Review

- [ ] **Code style**: Adheres to established coding standards
- [ ] **Documentation**: Adequate comments and docstrings
- [ ] **Testability**: Code is structured for effective testing
- [ ] **Performance**: No obvious performance bottlenecks
- [ ] **Security**: No security vulnerabilities or bad practices

#### 9.1.3 Architecture Review

- [ ] **Design patterns**: Appropriate use of design patterns
- [ ] **Separation of concerns**: Clear separation of responsibilities
- [ ] **Dependencies**: Minimal and appropriate dependencies
- [ ] **Interfaces**: Clean and well-defined interfaces
- [ ] **Maintainability**: Code is easy to understand and modify

### 9.2 Review Process

#### 9.2.1 Pull Request Template

```markdown
# Pull Request Template

## Description
Brief description of changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Related Issues
- Fixes #(issue number)
- Related to #(issue number)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No breaking changes without migration plan

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Additional Notes
[Any additional information for reviewers]
```

---

## 10. DEPLOYMENT AND MAINTENANCE

### 10.1 Deployment Standards

#### 10.1.1 Production Deployment Checklist

- [ ] **Code review**: All changes reviewed and approved
- [ ] **Testing**: Full test suite passes
- [ ] **Security scan**: No high/critical vulnerabilities
- [ ] **Performance testing**: Performance requirements met
- [ ] **Documentation**: Deployment docs updated
- [ ] **Rollback plan**: Rollback procedure verified
- [ ] **Monitoring**: Monitoring and alerting configured
- [ ] **Backup**: Data backup completed before deployment

### 10.2 Maintenance Standards

#### 10.2.1 Code Maintenance Guidelines

- **Regular refactoring**: Eliminate technical debt systematically
- **Dependency updates**: Keep dependencies current and secure
- **Performance monitoring**: Continuous performance optimization
- **Documentation updates**: Keep documentation synchronized with code
- **Security patches**: Apply security updates promptly

---

## 11. APPROVAL

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **Development Lead** | Engineering Manager | [Digital Signature] | 2024-12-18 |
| **Software Architect** | Technical Architect | [Digital Signature] | 2024-12-18 |
| **Quality Assurance** | QA Manager | [Digital Signature] | 2024-12-18 |
| **Security Officer** | Security Lead | [Digital Signature] | 2024-12-18 |

---

**Document Classification**: NASA-STD-8739.8 Compliant
**Security Level**: Internal Use
**Distribution**: Development Team, QA Team, Architecture Team

**End of Document**
