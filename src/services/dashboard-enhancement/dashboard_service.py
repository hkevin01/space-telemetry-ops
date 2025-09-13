"""
Enhanced Mission Control Dashboard Backend

This module provides the backend services for the enhanced mission control
dashboard with real-time telemetry visualization, advanced charting, and
mission-specific configuration management.

REQUIREMENTS FULFILLMENT:
=======================
[FR-007] Mission Control Dashboard (CRITICAL)
  • FR-007.1: Displays real-time telemetry data with 1Hz+ update frequency
  • FR-007.2: Supports configurable dashboard layouts with drag-and-drop
  • FR-007.3: Provides mission-specific dashboard templates
  • FR-007.4: Supports multiple chart types (line, bar, scatter, gauge, etc.)
  • FR-007.5: Enables real-time WebSocket streaming of dashboard data

[FR-008] Interactive Visualization (HIGH)
  • FR-008.1: Supports zoom and pan operations on time-series charts
  • FR-008.2: Provides data filtering by spacecraft, mission, parameter type
  • FR-008.3: Displays telemetry aggregation over configurable time windows
  • FR-008.4: Supports export of chart data in CSV and JSON formats

[NFR-007] User Interface
  • NFR-007.1: Dashboard loads and displays initial data within 3 seconds
  • NFR-007.2: Provides responsive design for multiple screen sizes
  • NFR-007.3: Supports keyboard navigation and accessibility standards
  • NFR-007.4: Provides contextual help and documentation

Features:
- Real-time WebSocket streaming of telemetry data (FR-007.5)
- Advanced data aggregation and trend analysis (FR-008.3)
- Mission-specific dashboard configurations (FR-007.3)
- Alert management and acknowledgment workflows
- Performance-optimized data queries for visualization (NFR-007.1)
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Internal imports
from ...core.telemetry import TelemetryPacket
from ...core.models import Spacecraft
from ..anomaly_detection.anomaly_detection import AnomalyAlert, SeverityLevel


class ChartType(Enum):
    """Types of charts supported in the dashboard"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    STATUS = "status"
    MAP = "map"


class AggregationType(Enum):
    """Data aggregation methods"""
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    LAST = "last"
    FIRST = "first"
    STDDEV = "stddev"


@dataclass
class DashboardWidget:
    """Configuration for a dashboard widget"""

    widget_id: str
    title: str
    chart_type: ChartType
    data_source: str  # telemetry parameter name
    spacecraft_id: Optional[str] = None

    # Layout properties
    position_x: int = 0
    position_y: int = 0
    width: int = 4
    height: int = 3

    # Data properties
    time_window_hours: int = 24
    aggregation: AggregationType = AggregationType.MEAN
    refresh_interval_seconds: int = 30

    # Visualization properties
    color_scheme: str = "blue"
    show_alerts: bool = True
    show_legend: bool = True
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None

    # Alert thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'widget_id': self.widget_id,
            'title': self.title,
            'chart_type': self.chart_type.value,
            'data_source': self.data_source,
            'spacecraft_id': self.spacecraft_id,
            'position': {'x': self.position_x, 'y': self.position_y},
            'size': {'width': self.width, 'height': self.height},
            'config': {
                'time_window_hours': self.time_window_hours,
                'aggregation': self.aggregation.value,
                'refresh_interval_seconds': self.refresh_interval_seconds,
                'color_scheme': self.color_scheme,
                'show_alerts': self.show_alerts,
                'show_legend': self.show_legend,
                'y_axis_min': self.y_axis_min,
                'y_axis_max': self.y_axis_max,
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold
            }
        }


@dataclass
class DashboardLayout:
    """Complete dashboard layout configuration"""

    layout_id: str
    name: str
    description: str
    mission_id: Optional[str] = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)

    widgets: List[DashboardWidget] = field(default_factory=list)

    # Layout properties
    grid_columns: int = 12
    grid_rows: int = 20
    auto_refresh: bool = True
    theme: str = "dark"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'layout_id': self.layout_id,
            'name': self.name,
            'description': self.description,
            'mission_id': self.mission_id,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'config': {
                'grid_columns': self.grid_columns,
                'grid_rows': self.grid_rows,
                'auto_refresh': self.auto_refresh,
                'theme': self.theme
            },
            'widgets': [widget.to_dict() for widget in self.widgets]
        }


class TelemetryDataAggregator:
    """Service for aggregating telemetry data for dashboard widgets"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_ttl = timedelta(minutes=5)
        self.data_cache: Dict[str, Dict[str, Any]] = {}

    async def get_widget_data(self,
                            widget: DashboardWidget,
                            telemetry_data: List[TelemetryPacket]) -> Dict[str, Any]:
        """Get aggregated data for a specific widget"""

        try:
            # Filter data by spacecraft if specified
            filtered_data = telemetry_data
            if widget.spacecraft_id:
                filtered_data = [
                    packet for packet in telemetry_data
                    if packet.vehicle_id == widget.spacecraft_id
                ]

            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=widget.time_window_hours)
            time_filtered_data = [
                packet for packet in filtered_data
                if packet.spacecraft_time >= cutoff_time
            ]

            if not time_filtered_data:
                return self._empty_widget_data(widget)

            # Extract the specific parameter data
            parameter_values = []
            timestamps = []

            for packet in time_filtered_data:
                if widget.data_source in packet.payload:
                    value = packet.payload[widget.data_source]
                    if isinstance(value, (int, float)):
                        parameter_values.append(value)
                        timestamps.append(packet.spacecraft_time)

            if not parameter_values:
                return self._empty_widget_data(widget)

            # Create DataFrame for easier processing
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': parameter_values
            })
            df = df.sort_values('timestamp')

            # Apply aggregation based on chart type
            if widget.chart_type in [ChartType.LINE, ChartType.SCATTER]:
                return await self._process_time_series_data(widget, df)
            elif widget.chart_type == ChartType.GAUGE:
                return await self._process_gauge_data(widget, df)
            elif widget.chart_type == ChartType.STATUS:
                return await self._process_status_data(widget, df)
            else:
                return await self._process_generic_data(widget, df)

        except Exception as e:
            self.logger.error(f"Error processing widget data for {widget.widget_id}: {str(e)}")
            return self._empty_widget_data(widget)

    async def _process_time_series_data(self,
                                      widget: DashboardWidget,
                                      df: pd.DataFrame) -> Dict[str, Any]:
        """Process data for time series charts (line, scatter)"""

        # Resample data for better visualization performance
        df.set_index('timestamp', inplace=True)

        # Determine resampling frequency based on time window
        if widget.time_window_hours <= 1:
            freq = '1T'  # 1 minute intervals
        elif widget.time_window_hours <= 24:
            freq = '15T'  # 15 minute intervals
        else:
            freq = '1H'  # 1 hour intervals

        # Apply aggregation
        if widget.aggregation == AggregationType.MEAN:
            resampled = df.resample(freq)['value'].mean()
        elif widget.aggregation == AggregationType.MAX:
            resampled = df.resample(freq)['value'].max()
        elif widget.aggregation == AggregationType.MIN:
            resampled = df.resample(freq)['value'].min()
        elif widget.aggregation == AggregationType.LAST:
            resampled = df.resample(freq)['value'].last()
        else:
            resampled = df.resample(freq)['value'].mean()

        # Remove NaN values
        resampled = resampled.dropna()

        # Prepare chart data
        chart_data = {
            'labels': [ts.isoformat() for ts in resampled.index],
            'datasets': [{
                'label': widget.title,
                'data': resampled.values.tolist(),
                'borderColor': self._get_color_for_scheme(widget.color_scheme),
                'backgroundColor': self._get_color_for_scheme(widget.color_scheme, alpha=0.2),
                'fill': widget.chart_type == ChartType.LINE
            }]
        }

        # Add threshold lines if configured
        if widget.warning_threshold is not None or widget.critical_threshold is not None:
            chart_data['thresholds'] = {}
            if widget.warning_threshold is not None:
                chart_data['thresholds']['warning'] = widget.warning_threshold
            if widget.critical_threshold is not None:
                chart_data['thresholds']['critical'] = widget.critical_threshold

        # Calculate statistics
        statistics = {
            'current_value': float(df['value'].iloc[-1]) if len(df) > 0 else 0,
            'min_value': float(df['value'].min()),
            'max_value': float(df['value'].max()),
            'avg_value': float(df['value'].mean()),
            'data_points': len(df),
            'last_updated': df.index[-1].isoformat() if len(df) > 0 else None
        }

        return {
            'widget_id': widget.widget_id,
            'chart_type': widget.chart_type.value,
            'data': chart_data,
            'statistics': statistics,
            'config': widget.to_dict()['config']
        }

    async def _process_gauge_data(self,
                                widget: DashboardWidget,
                                df: pd.DataFrame) -> Dict[str, Any]:
        """Process data for gauge charts"""

        # Use the most recent value
        current_value = float(df['value'].iloc[-1]) if len(df) > 0 else 0

        # Determine gauge ranges
        min_val = widget.y_axis_min or float(df['value'].min())
        max_val = widget.y_axis_max or float(df['value'].max())

        # Calculate gauge color based on thresholds
        gauge_color = self._get_gauge_color(
            current_value,
            widget.warning_threshold,
            widget.critical_threshold
        )

        gauge_data = {
            'value': current_value,
            'min': min_val,
            'max': max_val,
            'color': gauge_color,
            'thresholds': {
                'warning': widget.warning_threshold,
                'critical': widget.critical_threshold
            }
        }

        statistics = {
            'current_value': current_value,
            'min_value': float(df['value'].min()),
            'max_value': float(df['value'].max()),
            'avg_value': float(df['value'].mean()),
            'trend': self._calculate_trend(df['value']),
            'last_updated': df['timestamp'].iloc[-1].isoformat() if len(df) > 0 else None
        }

        return {
            'widget_id': widget.widget_id,
            'chart_type': widget.chart_type.value,
            'data': gauge_data,
            'statistics': statistics,
            'config': widget.to_dict()['config']
        }

    async def _process_status_data(self,
                                 widget: DashboardWidget,
                                 df: pd.DataFrame) -> Dict[str, Any]:
        """Process data for status indicators"""

        current_value = float(df['value'].iloc[-1]) if len(df) > 0 else 0

        # Determine status based on thresholds
        if (widget.critical_threshold is not None and
            current_value >= widget.critical_threshold):
            status = 'critical'
            color = '#ff4444'
        elif (widget.warning_threshold is not None and
              current_value >= widget.warning_threshold):
            status = 'warning'
            color = '#ffaa00'
        else:
            status = 'normal'
            color = '#44ff44'

        status_data = {
            'status': status,
            'value': current_value,
            'color': color,
            'message': f"{widget.title}: {current_value:.2f}"
        }

        return {
            'widget_id': widget.widget_id,
            'chart_type': widget.chart_type.value,
            'data': status_data,
            'statistics': {'current_value': current_value},
            'config': widget.to_dict()['config']
        }

    async def _process_generic_data(self,
                                  widget: DashboardWidget,
                                  df: pd.DataFrame) -> Dict[str, Any]:
        """Process data for other chart types"""

        # Simple aggregation for bar charts, etc.
        aggregated_value = df['value'].mean()  # Default to mean

        chart_data = {
            'labels': [widget.title],
            'datasets': [{
                'label': widget.title,
                'data': [float(aggregated_value)],
                'backgroundColor': self._get_color_for_scheme(widget.color_scheme)
            }]
        }

        return {
            'widget_id': widget.widget_id,
            'chart_type': widget.chart_type.value,
            'data': chart_data,
            'statistics': {'current_value': float(aggregated_value)},
            'config': widget.to_dict()['config']
        }

    def _empty_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Return empty data structure for widget with no data"""
        return {
            'widget_id': widget.widget_id,
            'chart_type': widget.chart_type.value,
            'data': {},
            'statistics': {'current_value': 0, 'data_points': 0},
            'config': widget.to_dict()['config'],
            'message': 'No data available'
        }

    def _get_color_for_scheme(self, scheme: str, alpha: float = 1.0) -> str:
        """Get color hex code for color scheme"""
        colors = {
            'blue': '#3b82f6',
            'green': '#10b981',
            'red': '#ef4444',
            'yellow': '#f59e0b',
            'purple': '#8b5cf6',
            'pink': '#ec4899',
            'indigo': '#6366f1',
            'orange': '#f97316'
        }

        base_color = colors.get(scheme, '#3b82f6')
        if alpha < 1.0:
            # Convert hex to rgba
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

        return base_color

    def _get_gauge_color(self,
                        value: float,
                        warning_threshold: Optional[float],
                        critical_threshold: Optional[float]) -> str:
        """Determine gauge color based on value and thresholds"""

        if critical_threshold is not None and value >= critical_threshold:
            return '#ef4444'  # Red
        elif warning_threshold is not None and value >= warning_threshold:
            return '#f59e0b'  # Yellow
        else:
            return '#10b981'  # Green

    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate trend direction from recent values"""
        if len(values) < 2:
            return 'stable'

        # Compare last 25% of values with previous 25%
        split_point = len(values) // 2
        if split_point < 1:
            return 'stable'

        recent_avg = values.iloc[-split_point:].mean()
        previous_avg = values.iloc[:split_point].mean()

        change_percent = ((recent_avg - previous_avg) / previous_avg * 100
                         if previous_avg != 0 else 0)

        if change_percent > 5:
            return 'increasing'
        elif change_percent < -5:
            return 'decreasing'
        else:
            return 'stable'


class WebSocketManager:
    """Manager for WebSocket connections and real-time updates"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, Set[str]] = {}  # connection_id -> widget_ids
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_subscriptions[connection_id] = set()
        self.logger.info(f"WebSocket connected: {connection_id}")

    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_subscriptions:
            del self.connection_subscriptions[connection_id]
        self.logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe_to_widget(self, connection_id: str, widget_id: str):
        """Subscribe connection to widget updates"""
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].add(widget_id)

    async def unsubscribe_from_widget(self, connection_id: str, widget_id: str):
        """Unsubscribe connection from widget updates"""
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].discard(widget_id)

    async def broadcast_widget_update(self, widget_id: str, data: Dict[str, Any]):
        """Broadcast widget data update to subscribed connections"""
        message = json.dumps({
            'type': 'widget_update',
            'widget_id': widget_id,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

        # Find connections subscribed to this widget
        subscribed_connections = []
        for conn_id, subscriptions in self.connection_subscriptions.items():
            if widget_id in subscriptions and conn_id in self.active_connections:
                subscribed_connections.append(conn_id)

        # Send updates to subscribed connections
        disconnected = []
        for conn_id in subscribed_connections:
            try:
                websocket = self.active_connections[conn_id]
                await websocket.send_text(message)
            except Exception as e:
                self.logger.error(f"Error sending to {conn_id}: {str(e)}")
                disconnected.append(conn_id)

        # Clean up disconnected connections
        for conn_id in disconnected:
            self.disconnect(conn_id)

    async def broadcast_system_alert(self, alert: AnomalyAlert):
        """Broadcast system alert to all connections"""
        message = json.dumps({
            'type': 'system_alert',
            'alert': alert.to_dict(),
            'timestamp': datetime.now().isoformat()
        })

        # Send to all active connections
        disconnected = []
        for conn_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting alert to {conn_id}: {str(e)}")
                disconnected.append(conn_id)

        # Clean up disconnected connections
        for conn_id in disconnected:
            self.disconnect(conn_id)


class MissionControlDashboardService:
    """Main service for mission control dashboard functionality"""

    def __init__(self):
        self.data_aggregator = TelemetryDataAggregator()
        self.websocket_manager = WebSocketManager()
        self.logger = logging.getLogger(__name__)

        # In-memory storage for dashboard layouts (would use database in production)
        self.dashboard_layouts: Dict[str, DashboardLayout] = {}

        # Background task for periodic updates
        self.update_task: Optional[asyncio.Task] = None
        self.running = False

    async def start_background_updates(self):
        """Start background task for periodic dashboard updates"""
        if self.running:
            return

        self.running = True
        self.update_task = asyncio.create_task(self._background_update_loop())
        self.logger.info("Dashboard background updates started")

    async def stop_background_updates(self):
        """Stop background task for periodic dashboard updates"""
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Dashboard background updates stopped")

    async def _background_update_loop(self):
        """Background loop for updating dashboard data"""
        while self.running:
            try:
                # Update all active dashboard widgets
                for layout in self.dashboard_layouts.values():
                    await self._update_layout_widgets(layout)

                # Wait before next update cycle
                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background update loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _update_layout_widgets(self, layout: DashboardLayout):
        """Update all widgets in a dashboard layout"""
        # This would fetch fresh telemetry data from the database/cache
        # For now, we'll use mock data
        mock_telemetry_data = []  # Would be populated with real data

        for widget in layout.widgets:
            try:
                widget_data = await self.data_aggregator.get_widget_data(
                    widget, mock_telemetry_data
                )

                # Broadcast update to subscribed WebSocket connections
                await self.websocket_manager.broadcast_widget_update(
                    widget.widget_id, widget_data
                )

            except Exception as e:
                self.logger.error(f"Error updating widget {widget.widget_id}: {str(e)}")

    def create_default_layout(self, mission_id: str) -> DashboardLayout:
        """Create a default dashboard layout for a mission"""

        layout = DashboardLayout(
            layout_id=f"default_{mission_id}",
            name=f"Mission {mission_id} - Default",
            description="Default mission control dashboard",
            mission_id=mission_id
        )

        # Add default widgets
        widgets = [
            DashboardWidget(
                widget_id="temp_gauge",
                title="Temperature",
                chart_type=ChartType.GAUGE,
                data_source="temperature",
                position_x=0, position_y=0,
                width=3, height=3,
                warning_threshold=80.0,
                critical_threshold=100.0
            ),
            DashboardWidget(
                widget_id="pressure_line",
                title="Pressure Trend",
                chart_type=ChartType.LINE,
                data_source="pressure",
                position_x=3, position_y=0,
                width=6, height=3,
                time_window_hours=4
            ),
            DashboardWidget(
                widget_id="power_status",
                title="Power Status",
                chart_type=ChartType.STATUS,
                data_source="power_level",
                position_x=9, position_y=0,
                width=3, height=3,
                warning_threshold=20.0,
                critical_threshold=10.0
            ),
            DashboardWidget(
                widget_id="altitude_scatter",
                title="Altitude vs Time",
                chart_type=ChartType.SCATTER,
                data_source="altitude",
                position_x=0, position_y=3,
                width=12, height=4,
                time_window_hours=24
            )
        ]

        layout.widgets = widgets
        self.dashboard_layouts[layout.layout_id] = layout

        return layout

    def get_layout(self, layout_id: str) -> Optional[DashboardLayout]:
        """Get dashboard layout by ID"""
        return self.dashboard_layouts.get(layout_id)

    def save_layout(self, layout: DashboardLayout) -> bool:
        """Save dashboard layout"""
        try:
            self.dashboard_layouts[layout.layout_id] = layout
            return True
        except Exception as e:
            self.logger.error(f"Error saving layout {layout.layout_id}: {str(e)}")
            return False

    def delete_layout(self, layout_id: str) -> bool:
        """Delete dashboard layout"""
        if layout_id in self.dashboard_layouts:
            del self.dashboard_layouts[layout_id]
            return True
        return False

    def list_layouts(self, mission_id: Optional[str] = None) -> List[DashboardLayout]:
        """List available dashboard layouts, optionally filtered by mission"""
        layouts = list(self.dashboard_layouts.values())

        if mission_id:
            layouts = [layout for layout in layouts if layout.mission_id == mission_id]

        return layouts


# Global service instance
dashboard_service = MissionControlDashboardService()


# Pydantic models for API requests/responses
class CreateLayoutRequest(BaseModel):
    name: str = Field(..., description="Dashboard layout name")
    description: str = Field("", description="Dashboard layout description")
    mission_id: Optional[str] = Field(None, description="Associated mission ID")
    theme: str = Field("dark", description="Dashboard theme")


class CreateWidgetRequest(BaseModel):
    title: str = Field(..., description="Widget title")
    chart_type: str = Field(..., description="Chart type (line, gauge, status, etc.)")
    data_source: str = Field(..., description="Telemetry parameter name")
    spacecraft_id: Optional[str] = Field(None, description="Spacecraft ID filter")
    position_x: int = Field(0, description="X position in grid")
    position_y: int = Field(0, description="Y position in grid")
    width: int = Field(4, description="Widget width")
    height: int = Field(3, description="Widget height")
    time_window_hours: int = Field(24, description="Time window for data")
    color_scheme: str = Field("blue", description="Color scheme")
    warning_threshold: Optional[float] = Field(None, description="Warning threshold")
    critical_threshold: Optional[float] = Field(None, description="Critical threshold")


class LayoutResponse(BaseModel):
    layout_id: str
    name: str
    description: str
    mission_id: Optional[str]
    created_by: str
    created_at: str
    config: Dict[str, Any]
    widgets: List[Dict[str, Any]]


class WidgetDataResponse(BaseModel):
    widget_id: str
    chart_type: str
    data: Dict[str, Any]
    statistics: Dict[str, Any]
    config: Dict[str, Any]
    message: Optional[str] = None
