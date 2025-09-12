"""
Anomaly Detection API Integration

FastAPI endpoints for real-time anomaly detection and alert management.
Integrates with the core telemetry processing pipeline to provide
real-time anomaly detection capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from ..anomaly_detection import (
    AnomalyDetectionService,
    AnomalyAlert,
    SeverityLevel,
    AnomalyType,
    anomaly_service
)
from ...core.telemetry import TelemetryPacket
from ...core.models import Spacecraft

# Initialize router
router = APIRouter(prefix="/api/anomaly", tags=["Anomaly Detection"])


class AnomalyResponse(BaseModel):
    """Response model for anomaly detection results"""

    anomaly_id: str
    timestamp: datetime
    spacecraft_id: str
    anomaly_type: str
    severity: str
    confidence: float
    parameter_name: str
    current_value: float
    expected_value: Optional[float] = None
    deviation_magnitude: Optional[float] = None
    description: str
    recommended_action: str
    detection_method: str
    processing_time_ms: float = 0.0


class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""

    telemetry_packets: List[Dict[str, Any]] = Field(
        ..., description="List of telemetry packets to analyze"
    )
    spacecraft_id: Optional[str] = Field(
        None, description="Filter by specific spacecraft ID"
    )
    enable_realtime: bool = Field(
        True, description="Enable real-time anomaly detection"
    )


class AnomalyFilterRequest(BaseModel):
    """Request model for filtering anomaly history"""

    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    severity_levels: Optional[List[str]] = Field(None, description="Severity level filter")
    anomaly_types: Optional[List[str]] = Field(None, description="Anomaly type filter")
    spacecraft_ids: Optional[List[str]] = Field(None, description="Spacecraft ID filter")
    limit: int = Field(100, description="Maximum number of results", le=1000)


class PerformanceMetrics(BaseModel):
    """Response model for performance metrics"""

    detection_count: int
    average_processing_time_ms: float
    total_processing_time_ms: float
    alert_history_size: int
    active_detectors: List[str]
    uptime_hours: float
    throughput_per_second: float


@router.post("/detect", response_model=List[AnomalyResponse])
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    background_tasks: BackgroundTasks
) -> List[AnomalyResponse]:
    """
    Detect anomalies in telemetry data

    This endpoint analyzes telemetry packets for anomalous patterns using
    multiple detection algorithms including statistical analysis, temporal
    pattern recognition, and machine learning models.
    """
    try:
        # Convert request data to TelemetryPacket objects
        telemetry_packets = []
        for packet_data in request.telemetry_packets:
            try:
                packet = TelemetryPacket(
                    packet_id=packet_data.get('packet_id', f"pkt_{datetime.now().timestamp()}"),
                    sequence_number=packet_data.get('sequence_number', 0),
                    vehicle_id=packet_data.get('spacecraft_id', 'unknown'),
                    spacecraft_time=datetime.fromisoformat(
                        packet_data.get('timestamp', datetime.now().isoformat())
                    ),
                    ground_time=datetime.now(),
                    payload=packet_data.get('payload', {}),
                    packet_type=packet_data.get('packet_type', 'telemetry')
                )

                # Filter by spacecraft ID if specified
                if request.spacecraft_id and packet.vehicle_id != request.spacecraft_id:
                    continue

                telemetry_packets.append(packet)

            except Exception as e:
                # Log individual packet conversion errors but continue
                print(f"Warning: Failed to convert packet data: {str(e)}")
                continue

        if not telemetry_packets:
            return []

        # Perform anomaly detection
        anomaly_alerts = await anomaly_service.detect_anomalies(telemetry_packets)

        # Schedule model updates in the background
        if request.enable_realtime and len(telemetry_packets) > 10:
            background_tasks.add_task(
                anomaly_service.update_models,
                telemetry_packets[-50:]  # Use recent data for training
            )

        # Convert to response format
        responses = []
        for alert in anomaly_alerts:
            response = AnomalyResponse(
                anomaly_id=alert.anomaly_id,
                timestamp=alert.timestamp,
                spacecraft_id=alert.spacecraft_id,
                anomaly_type=alert.anomaly_type.value,
                severity=alert.severity.value,
                confidence=alert.confidence,
                parameter_name=alert.parameter_name,
                current_value=alert.current_value,
                expected_value=alert.expected_value,
                deviation_magnitude=alert.deviation_magnitude,
                description=alert.description,
                recommended_action=alert.recommended_action,
                detection_method=alert.detection_method,
                processing_time_ms=alert.processing_time_ms
            )
            responses.append(response)

        return responses

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@router.get("/alerts/recent", response_model=List[AnomalyResponse])
async def get_recent_alerts(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    spacecraft_id: Optional[str] = Query(None, description="Filter by spacecraft ID")
) -> List[AnomalyResponse]:
    """
    Get recent anomaly alerts within specified time window

    Returns alerts from the last N hours, optionally filtered by severity
    level and spacecraft ID.
    """
    try:
        # Convert severity string to enum if provided
        severity_filter = None
        if severity:
            try:
                severity_filter = SeverityLevel(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity level: {severity}. "
                           f"Valid options: {[s.value for s in SeverityLevel]}"
                )

        # Get recent alerts
        alerts = anomaly_service.get_recent_alerts(
            hours=hours,
            severity_filter=severity_filter
        )

        # Filter by spacecraft ID if specified
        if spacecraft_id:
            alerts = [alert for alert in alerts if alert.spacecraft_id == spacecraft_id]

        # Convert to response format
        responses = []
        for alert in alerts:
            response = AnomalyResponse(
                anomaly_id=alert.anomaly_id,
                timestamp=alert.timestamp,
                spacecraft_id=alert.spacecraft_id,
                anomaly_type=alert.anomaly_type.value,
                severity=alert.severity.value,
                confidence=alert.confidence,
                parameter_name=alert.parameter_name,
                current_value=alert.current_value,
                expected_value=alert.expected_value,
                deviation_magnitude=alert.deviation_magnitude,
                description=alert.description,
                recommended_action=alert.recommended_action,
                detection_method=alert.detection_method,
                processing_time_ms=alert.processing_time_ms
            )
            responses.append(response)

        return responses

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve recent alerts: {str(e)}"
        )


@router.post("/alerts/search", response_model=List[AnomalyResponse])
async def search_alerts(request: AnomalyFilterRequest) -> List[AnomalyResponse]:
    """
    Search anomaly alerts with advanced filtering options

    Provides comprehensive search capabilities across the anomaly alert
    history with multiple filter criteria.
    """
    try:
        # Get all alerts from history
        all_alerts = anomaly_service.alert_history

        # Apply filters
        filtered_alerts = all_alerts

        # Time range filter
        if request.start_time:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.timestamp >= request.start_time
            ]

        if request.end_time:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.timestamp <= request.end_time
            ]

        # Severity filter
        if request.severity_levels:
            severity_enums = []
            for sev in request.severity_levels:
                try:
                    severity_enums.append(SeverityLevel(sev.lower()))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid severity level: {sev}"
                    )

            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.severity in severity_enums
            ]

        # Anomaly type filter
        if request.anomaly_types:
            type_enums = []
            for atype in request.anomaly_types:
                try:
                    type_enums.append(AnomalyType(atype.lower()))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid anomaly type: {atype}"
                    )

            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.anomaly_type in type_enums
            ]

        # Spacecraft ID filter
        if request.spacecraft_ids:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.spacecraft_id in request.spacecraft_ids
            ]

        # Sort by timestamp (most recent first)
        filtered_alerts = sorted(
            filtered_alerts,
            key=lambda x: x.timestamp,
            reverse=True
        )

        # Apply limit
        filtered_alerts = filtered_alerts[:request.limit]

        # Convert to response format
        responses = []
        for alert in filtered_alerts:
            response = AnomalyResponse(
                anomaly_id=alert.anomaly_id,
                timestamp=alert.timestamp,
                spacecraft_id=alert.spacecraft_id,
                anomaly_type=alert.anomaly_type.value,
                severity=alert.severity.value,
                confidence=alert.confidence,
                parameter_name=alert.parameter_name,
                current_value=alert.current_value,
                expected_value=alert.expected_value,
                deviation_magnitude=alert.deviation_magnitude,
                description=alert.description,
                recommended_action=alert.recommended_action,
                detection_method=alert.detection_method,
                processing_time_ms=alert.processing_time_ms
            )
            responses.append(response)

        return responses

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Alert search failed: {str(e)}"
        )


@router.get("/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics() -> PerformanceMetrics:
    """
    Get performance metrics for the anomaly detection service

    Returns comprehensive performance data including processing times,
    detection counts, and system health information.
    """
    try:
        metrics = anomaly_service.get_performance_metrics()

        # Calculate additional metrics
        uptime_hours = 24.0  # Placeholder - would track actual uptime
        throughput_per_second = (
            metrics['detection_count'] / (uptime_hours * 3600)
            if uptime_hours > 0 else 0
        )

        return PerformanceMetrics(
            detection_count=metrics['detection_count'],
            average_processing_time_ms=metrics['average_processing_time_ms'],
            total_processing_time_ms=metrics['total_processing_time_ms'],
            alert_history_size=metrics['alert_history_size'],
            active_detectors=metrics['active_detectors'],
            uptime_hours=uptime_hours,
            throughput_per_second=throughput_per_second
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for anomaly detection service

    Returns service health status and basic operational information.
    """
    try:
        metrics = anomaly_service.get_performance_metrics()

        # Determine health status based on performance
        status = "healthy"
        if metrics['average_processing_time_ms'] > 1000:  # > 1 second
            status = "degraded"
        elif len(anomaly_service.detectors) == 0:
            status = "unhealthy"

        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "service": "anomaly_detection",
            "version": "1.0.0",
            "active_detectors": metrics['active_detectors'],
            "detection_count": metrics['detection_count'],
            "average_processing_time_ms": metrics['average_processing_time_ms']
        }

    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "service": "anomaly_detection",
            "error": str(e)
        }


@router.get("/config")
async def get_configuration() -> Dict[str, Any]:
    """
    Get current anomaly detection configuration

    Returns configuration settings for all active detectors.
    """
    try:
        config = {
            "detectors": {},
            "service_settings": {
                "max_history_size": anomaly_service.max_history_size,
                "detection_enabled": True
            }
        }

        # Get detector-specific configurations
        for name, detector in anomaly_service.detectors.items():
            if hasattr(detector, '__dict__'):
                detector_config = {}
                for key, value in detector.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        detector_config[key] = value
                config["detectors"][name] = detector_config

        return config

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )


# WebSocket endpoint for real-time anomaly alerts
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

class AnomalyWebSocketManager:
    """Manager for WebSocket connections for real-time anomaly alerts"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.alert_queue = asyncio.Queue()

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_alert(self, alert: AnomalyAlert):
        """Send anomaly alert to all connected clients"""
        if not self.active_connections:
            return

        alert_data = alert.to_dict()
        message = json.dumps(alert_data)

        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


websocket_manager = AnomalyWebSocketManager()


@router.websocket("/alerts/stream")
async def anomaly_alert_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time anomaly alerts

    Provides a real-time stream of anomaly alerts as they are detected.
    Clients can subscribe to receive immediate notifications of anomalies.
    """
    await websocket_manager.connect(websocket)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            try:
                # Wait for any client messages (keepalive, etc.)
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle client messages (e.g., subscription filters)
                if message == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send keepalive if no messages received
                await websocket.send_text('{"type": "keepalive", "timestamp": "' +
                                        datetime.now().isoformat() + '"}')

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        websocket_manager.disconnect(websocket)
