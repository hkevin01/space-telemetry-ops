"""
Mission Control Dashboard Enhancement Integration

This module integrates the enhanced mission control dashboard with the main
FastAPI application, providing complete real-time telemetry visualization
capabilities with WebSocket streaming, advanced analytics, and interactive
dashboard management.

REQUIREMENTS FULFILLMENT:
=======================
[FR-007] Mission Control Dashboard (CRITICAL)
  • FR-007.1: Integrates real-time telemetry data streaming
  • FR-007.2: Provides configurable dashboard layout management
  • FR-007.3: Implements mission-specific dashboard templates
  • FR-007.4: Supports multiple chart types and visualizations
  • FR-007.5: Enables WebSocket-based real-time data streaming

[FR-008] Interactive Visualization (HIGH)
  • FR-008.1: Provides interactive chart controls and navigation
  • FR-008.2: Implements multi-dimensional data filtering
  • FR-008.3: Supports configurable time window aggregation
  • FR-008.4: Enables data export functionality

[NFR-008] System Maintenance
  • NFR-008.2: Provides comprehensive logging and monitoring integration
  • NFR-008.3: Supports runtime configuration and service management

Integration Features:
- Seamless integration with existing telemetry pipeline (FR-007.1)
- Real-time data streaming via WebSocket (FR-007.5)
- Advanced anomaly detection integration (FR-008.2)
- Performance optimization service integration (NFR-008.2)
- Mission-specific dashboard templates (FR-007.3)
- Enterprise-ready monitoring and analytics (NFR-008.2)

Usage:
    from src.services.dashboard_enhancement.integration import DashboardIntegrationService

    # Initialize service
    dashboard_service = DashboardIntegrationService()

    # Start background services
    await dashboard_service.start()

    # Create mission-specific dashboard
    layout = await dashboard_service.create_mission_dashboard(mission_id="MISSION_001")
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ...core.models import Spacecraft
from ...core.telemetry import TelemetryPacket, TelemetryProcessor
from ..anomaly_detection.anomaly_detection import AnomalyAlert, AnomalyDetectionService
from ..performance_optimization.performance_service import (
    PerformanceOptimizationService,
)
from .api import router as dashboard_router

# Internal imports
from .dashboard_service import (
    ChartType,
    DashboardLayout,
    DashboardWidget,
    dashboard_service,
)


class DashboardIntegrationService:
    """
    Service for integrating enhanced dashboard capabilities with the main application
    """

    def __init__(self, app: Optional[FastAPI] = None):
        self.app = app
        self.logger = logging.getLogger(__name__)

        # Service references
        self.dashboard_service = dashboard_service
        self.anomaly_service: Optional[AnomalyDetectionService] = None
        self.performance_service: Optional[PerformanceOptimizationService] = None
        self.telemetry_processor: Optional[TelemetryProcessor] = None

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False

        # Data integration
        self.telemetry_cache: Dict[str, List[TelemetryPacket]] = {}
        self.cache_max_age = timedelta(hours=24)

        # Mission dashboard templates
        self.mission_templates = {
            'satellite': self._create_satellite_dashboard_config,
            'rover': self._create_rover_dashboard_config,
            'probe': self._create_probe_dashboard_config,
            'station': self._create_station_dashboard_config
        }

    async def initialize(self,
                        anomaly_service: Optional[AnomalyDetectionService] = None,
                        performance_service: Optional[PerformanceOptimizationService] = None,
                        telemetry_processor: Optional[TelemetryProcessor] = None):
        """Initialize the dashboard integration service with other services"""

        self.anomaly_service = anomaly_service
        self.performance_service = performance_service
        self.telemetry_processor = telemetry_processor

        # Set up event handlers for real-time updates
        if self.anomaly_service:
            await self._setup_anomaly_integration()

        if self.performance_service:
            await self._setup_performance_integration()

        if self.telemetry_processor:
            await self._setup_telemetry_integration()

        self.logger.info("Dashboard integration service initialized")

    async def start(self):
        """Start the dashboard integration service and background tasks"""
        if self.running:
            return

        self.running = True

        # Start dashboard background updates
        await self.dashboard_service.start_background_updates()

        # Start integration background tasks
        self.background_tasks.extend([
            asyncio.create_task(self._telemetry_cache_manager()),
            asyncio.create_task(self._dashboard_data_updater()),
            asyncio.create_task(self._alert_processor())
        ])

        self.logger.info("Dashboard integration service started")

    async def stop(self):
        """Stop the dashboard integration service and background tasks"""
        self.running = False

        # Stop dashboard background updates
        await self.dashboard_service.stop_background_updates()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        self.logger.info("Dashboard integration service stopped")

    async def _setup_anomaly_integration(self):
        """Set up integration with anomaly detection service"""

        async def handle_anomaly_alert(alert: AnomalyAlert):
            """Handle anomaly alerts and broadcast to dashboard"""
            try:
                # Broadcast alert to all dashboard WebSocket connections
                await self.dashboard_service.websocket_manager.broadcast_system_alert(alert)

                # Update relevant widgets if they monitor the same parameters
                await self._update_widgets_for_parameter(alert.parameter_name, alert)

            except Exception as e:
                self.logger.error(f"Error handling anomaly alert: {str(e)}")

        # Register alert handler (would use proper event system in production)
        self.anomaly_alert_handler = handle_anomaly_alert

    async def _setup_performance_integration(self):
        """Set up integration with performance optimization service"""

        # Performance metrics could be displayed in system health widgets
        async def update_performance_widgets():
            """Update widgets showing system performance metrics"""
            try:
                if not self.performance_service:
                    return

                # Get performance metrics
                metrics = await self.performance_service.get_performance_metrics()

                # Find performance-related widgets and update them
                for layout in self.dashboard_service.dashboard_layouts.values():
                    for widget in layout.widgets:
                        if widget.data_source in ['query_time', 'cpu_usage', 'memory_usage', 'cache_hit_rate']:
                            # Create mock telemetry data from performance metrics
                            mock_data = self._create_mock_telemetry_from_metrics(metrics, widget.data_source)
                            widget_data = await self.dashboard_service.data_aggregator.get_widget_data(
                                widget, mock_data
                            )

                            # Broadcast update
                            await self.dashboard_service.websocket_manager.broadcast_widget_update(
                                widget.widget_id, widget_data
                            )

            except Exception as e:
                self.logger.error(f"Error updating performance widgets: {str(e)}")

        self.performance_widget_updater = update_performance_widgets

    async def _setup_telemetry_integration(self):
        """Set up integration with telemetry processing pipeline"""

        async def handle_telemetry_packet(packet: TelemetryPacket):
            """Handle incoming telemetry packets for dashboard updates"""
            try:
                # Add to cache
                spacecraft_cache = self.telemetry_cache.setdefault(packet.vehicle_id, [])
                spacecraft_cache.append(packet)

                # Limit cache size (keep last 1000 packets per spacecraft)
                if len(spacecraft_cache) > 1000:
                    spacecraft_cache = spacecraft_cache[-1000:]
                    self.telemetry_cache[packet.vehicle_id] = spacecraft_cache

                # Update relevant widgets
                await self._update_widgets_for_spacecraft(packet.vehicle_id, packet)

            except Exception as e:
                self.logger.error(f"Error handling telemetry packet: {str(e)}")

        self.telemetry_packet_handler = handle_telemetry_packet

    async def _telemetry_cache_manager(self):
        """Background task to manage telemetry data cache"""
        while self.running:
            try:
                # Clean old data from cache
                cutoff_time = datetime.now() - self.cache_max_age

                for spacecraft_id in list(self.telemetry_cache.keys()):
                    packets = self.telemetry_cache[spacecraft_id]
                    filtered_packets = [
                        packet for packet in packets
                        if packet.spacecraft_time >= cutoff_time
                    ]

                    if filtered_packets:
                        self.telemetry_cache[spacecraft_id] = filtered_packets
                    else:
                        del self.telemetry_cache[spacecraft_id]

                await asyncio.sleep(300)  # Clean every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in telemetry cache manager: {str(e)}")
                await asyncio.sleep(30)  # Brief pause on error

    async def _dashboard_data_updater(self):
        """Background task to update dashboard widgets with fresh data"""
        while self.running:
            try:
                # Update all dashboard layouts
                for layout in self.dashboard_service.dashboard_layouts.values():
                    await self._update_layout_widgets(layout)

                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard data updater: {str(e)}")
                await asyncio.sleep(30)

    async def _alert_processor(self):
        """Background task to process and manage alerts"""
        while self.running:
            try:
                # Process alerts from various sources
                if self.anomaly_service:
                    # Get recent anomaly alerts
                    recent_alerts = await self.anomaly_service.get_recent_alerts(
                        hours=1, severity_filter=['warning', 'critical']
                    )

                    # Broadcast new alerts
                    for alert in recent_alerts:
                        await self.dashboard_service.websocket_manager.broadcast_system_alert(alert)

                await asyncio.sleep(60)  # Check for new alerts every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert processor: {str(e)}")
                await asyncio.sleep(30)

    async def _update_layout_widgets(self, layout: DashboardLayout):
        """Update all widgets in a dashboard layout with fresh data"""
        for widget in layout.widgets:
            try:
                # Get relevant telemetry data for the widget
                telemetry_data = []

                if widget.spacecraft_id and widget.spacecraft_id in self.telemetry_cache:
                    telemetry_data = self.telemetry_cache[widget.spacecraft_id]
                else:
                    # Aggregate data from all spacecraft
                    for spacecraft_packets in self.telemetry_cache.values():
                        telemetry_data.extend(spacecraft_packets)

                # Get widget data
                widget_data = await self.dashboard_service.data_aggregator.get_widget_data(
                    widget, telemetry_data
                )

                # Broadcast update to subscribed connections
                await self.dashboard_service.websocket_manager.broadcast_widget_update(
                    widget.widget_id, widget_data
                )

            except Exception as e:
                self.logger.error(f"Error updating widget {widget.widget_id}: {str(e)}")

    async def _update_widgets_for_parameter(self, parameter_name: str, alert: AnomalyAlert):
        """Update widgets monitoring a specific parameter when an alert occurs"""
        for layout in self.dashboard_service.dashboard_layouts.values():
            for widget in layout.widgets:
                if widget.data_source == parameter_name:
                    try:
                        # Get relevant telemetry data
                        telemetry_data = []
                        if widget.spacecraft_id and widget.spacecraft_id in self.telemetry_cache:
                            telemetry_data = self.telemetry_cache[widget.spacecraft_id]

                        # Get updated widget data
                        widget_data = await self.dashboard_service.data_aggregator.get_widget_data(
                            widget, telemetry_data
                        )

                        # Add alert information to widget data
                        widget_data['alert'] = alert.to_dict()

                        # Broadcast update
                        await self.dashboard_service.websocket_manager.broadcast_widget_update(
                            widget.widget_id, widget_data
                        )

                    except Exception as e:
                        self.logger.error(f"Error updating widget for parameter {parameter_name}: {str(e)}")

    async def _update_widgets_for_spacecraft(self, spacecraft_id: str, packet: TelemetryPacket):
        """Update widgets monitoring a specific spacecraft when new data arrives"""
        for layout in self.dashboard_service.dashboard_layouts.values():
            for widget in layout.widgets:
                if (widget.spacecraft_id == spacecraft_id or widget.spacecraft_id is None) and \
                   widget.data_source in packet.payload:
                    try:
                        # Get relevant telemetry data
                        telemetry_data = self.telemetry_cache.get(spacecraft_id, [])

                        # Get updated widget data
                        widget_data = await self.dashboard_service.data_aggregator.get_widget_data(
                            widget, telemetry_data
                        )

                        # Broadcast update
                        await self.dashboard_service.websocket_manager.broadcast_widget_update(
                            widget.widget_id, widget_data
                        )

                    except Exception as e:
                        self.logger.error(f"Error updating widget for spacecraft {spacecraft_id}: {str(e)}")

    def _create_mock_telemetry_from_metrics(self,
                                          metrics: Dict[str, Any],
                                          parameter_name: str) -> List[TelemetryPacket]:
        """Create mock telemetry packets from performance metrics"""
        current_time = datetime.now()

        # Map performance metrics to telemetry format
        value = metrics.get(parameter_name, 0)

        mock_packet = TelemetryPacket(
            vehicle_id="SYSTEM",
            packet_id=str(uuid.uuid4()),
            spacecraft_time=current_time,
            ground_time=current_time,
            payload={parameter_name: value}
        )

        return [mock_packet]

    async def create_mission_dashboard(self,
                                     mission_id: str,
                                     mission_type: str = 'satellite',
                                     spacecraft_list: Optional[List[Spacecraft]] = None) -> DashboardLayout:
        """
        Create a comprehensive mission dashboard for a specific mission
        """
        try:
            # Get template configuration for mission type
            if mission_type in self.mission_templates:
                config_func = self.mission_templates[mission_type]
                layout_config = config_func(mission_id, spacecraft_list or [])
            else:
                # Default configuration
                layout_config = self._create_default_dashboard_config(mission_id, spacecraft_list or [])

            # Create layout
            layout = DashboardLayout(
                layout_id=f"mission_{mission_id}_{uuid.uuid4().hex[:8]}",
                **layout_config
            )

            # Save layout
            success = self.dashboard_service.save_layout(layout)
            if not success:
                raise Exception("Failed to save mission dashboard layout")

            self.logger.info(f"Created mission dashboard for {mission_id}: {layout.layout_id}")
            return layout

        except Exception as e:
            self.logger.error(f"Error creating mission dashboard for {mission_id}: {str(e)}")
            raise

    def _create_satellite_dashboard_config(self,
                                         mission_id: str,
                                         spacecraft_list: List[Spacecraft]) -> Dict[str, Any]:
        """Create dashboard configuration optimized for satellite missions"""

        widgets = []
        widget_y = 0

        for i, spacecraft in enumerate(spacecraft_list):
            # Orbital parameters row
            widgets.extend([
                DashboardWidget(
                    widget_id=f"altitude_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Altitude",
                    chart_type=ChartType.GAUGE,
                    data_source="altitude",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=0, position_y=widget_y,
                    width=3, height=3,
                    warning_threshold=200000,
                    critical_threshold=150000,
                    color_scheme="blue"
                ),
                DashboardWidget(
                    widget_id=f"velocity_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Velocity",
                    chart_type=ChartType.LINE,
                    data_source="velocity",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=3, position_y=widget_y,
                    width=6, height=3,
                    time_window_hours=4,
                    color_scheme="green"
                ),
                DashboardWidget(
                    widget_id=f"power_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Power",
                    chart_type=ChartType.STATUS,
                    data_source="power_level",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=9, position_y=widget_y,
                    width=3, height=3,
                    warning_threshold=20.0,
                    critical_threshold=10.0,
                    color_scheme="yellow"
                )
            ])
            widget_y += 3

        # Mission overview widgets
        widgets.extend([
            DashboardWidget(
                widget_id="mission_overview_temp",
                title="Fleet Temperature Overview",
                chart_type=ChartType.LINE,
                data_source="temperature",
                position_x=0, position_y=widget_y,
                width=12, height=4,
                time_window_hours=12,
                color_scheme="red"
            )
        ])

        return {
            'name': f"Mission {mission_id} - Satellite Fleet",
            'description': f"Comprehensive satellite mission dashboard for {mission_id}",
            'mission_id': mission_id,
            'theme': 'dark',
            'widgets': widgets
        }

    def _create_rover_dashboard_config(self,
                                     mission_id: str,
                                     spacecraft_list: List[Spacecraft]) -> Dict[str, Any]:
        """Create dashboard configuration optimized for rover missions"""

        widgets = []

        for i, spacecraft in enumerate(spacecraft_list):
            widgets.extend([
                DashboardWidget(
                    widget_id=f"position_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Position",
                    chart_type=ChartType.MAP,
                    data_source="position",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=0, position_y=i*4,
                    width=6, height=4,
                    color_scheme="blue"
                ),
                DashboardWidget(
                    widget_id=f"battery_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Battery",
                    chart_type=ChartType.GAUGE,
                    data_source="battery_level",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=6, position_y=i*4,
                    width=3, height=2,
                    warning_threshold=30,
                    critical_threshold=15,
                    color_scheme="orange"
                ),
                DashboardWidget(
                    widget_id=f"motors_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Motors",
                    chart_type=ChartType.STATUS,
                    data_source="motor_current",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=9, position_y=i*4,
                    width=3, height=2,
                    color_scheme="green"
                ),
                DashboardWidget(
                    widget_id=f"env_temp_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Environment",
                    chart_type=ChartType.LINE,
                    data_source="env_temperature",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=6, position_y=i*4+2,
                    width=6, height=2,
                    color_scheme="red"
                )
            ])

        return {
            'name': f"Mission {mission_id} - Rover Fleet",
            'description': f"Comprehensive rover mission dashboard for {mission_id}",
            'mission_id': mission_id,
            'theme': 'dark',
            'widgets': widgets
        }

    def _create_probe_dashboard_config(self,
                                     mission_id: str,
                                     spacecraft_list: List[Spacecraft]) -> Dict[str, Any]:
        """Create dashboard configuration optimized for deep space probe missions"""

        widgets = []

        for i, spacecraft in enumerate(spacecraft_list):
            widgets.extend([
                DashboardWidget(
                    widget_id=f"distance_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Distance from Earth",
                    chart_type=ChartType.LINE,
                    data_source="distance_earth",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=0, position_y=i*5,
                    width=8, height=3,
                    time_window_hours=168,  # 1 week
                    color_scheme="purple"
                ),
                DashboardWidget(
                    widget_id=f"rtg_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - RTG Power",
                    chart_type=ChartType.GAUGE,
                    data_source="rtg_power",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=8, position_y=i*5,
                    width=4, height=3,
                    warning_threshold=200,
                    critical_threshold=150,
                    color_scheme="orange"
                ),
                DashboardWidget(
                    widget_id=f"comm_delay_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Comm Delay",
                    chart_type=ChartType.LINE,
                    data_source="comm_delay",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=0, position_y=i*5+3,
                    width=6, height=2,
                    color_scheme="indigo"
                ),
                DashboardWidget(
                    widget_id=f"instruments_{spacecraft.vehicle_id}",
                    title=f"{spacecraft.name} - Instruments",
                    chart_type=ChartType.STATUS,
                    data_source="instrument_health",
                    spacecraft_id=spacecraft.vehicle_id,
                    position_x=6, position_y=i*5+3,
                    width=6, height=2,
                    color_scheme="green"
                )
            ])

        return {
            'name': f"Mission {mission_id} - Deep Space Probes",
            'description': f"Deep space probe mission dashboard for {mission_id}",
            'mission_id': mission_id,
            'theme': 'dark',
            'widgets': widgets
        }

    def _create_station_dashboard_config(self,
                                       mission_id: str,
                                       spacecraft_list: List[Spacecraft]) -> Dict[str, Any]:
        """Create dashboard configuration optimized for space station missions"""

        widgets = [
            # Life support systems
            DashboardWidget(
                widget_id="oxygen_level",
                title="Oxygen Level",
                chart_type=ChartType.GAUGE,
                data_source="oxygen_level",
                position_x=0, position_y=0,
                width=3, height=3,
                warning_threshold=18.0,
                critical_threshold=16.0,
                color_scheme="blue"
            ),
            DashboardWidget(
                widget_id="co2_level",
                title="CO2 Level",
                chart_type=ChartType.GAUGE,
                data_source="co2_level",
                position_x=3, position_y=0,
                width=3, height=3,
                warning_threshold=0.5,
                critical_threshold=1.0,
                color_scheme="red"
            ),
            DashboardWidget(
                widget_id="pressure",
                title="Cabin Pressure",
                chart_type=ChartType.LINE,
                data_source="cabin_pressure",
                position_x=6, position_y=0,
                width=6, height=3,
                time_window_hours=6,
                color_scheme="green"
            ),

            # Power and thermal
            DashboardWidget(
                widget_id="power_generation",
                title="Power Generation",
                chart_type=ChartType.LINE,
                data_source="power_generation",
                position_x=0, position_y=3,
                width=6, height=3,
                time_window_hours=12,
                color_scheme="yellow"
            ),
            DashboardWidget(
                widget_id="thermal_control",
                title="Thermal Control",
                chart_type=ChartType.STATUS,
                data_source="thermal_system",
                position_x=6, position_y=3,
                width=6, height=3,
                color_scheme="orange"
            ),

            # Communication and attitude
            DashboardWidget(
                widget_id="communication_status",
                title="Communication Status",
                chart_type=ChartType.STATUS,
                data_source="comm_status",
                position_x=0, position_y=6,
                width=6, height=2,
                color_scheme="indigo"
            ),
            DashboardWidget(
                widget_id="attitude_control",
                title="Attitude Control",
                chart_type=ChartType.SCATTER,
                data_source="attitude",
                position_x=6, position_y=6,
                width=6, height=2,
                color_scheme="purple"
            )
        ]

        return {
            'name': f"Mission {mission_id} - Space Station",
            'description': f"Space station mission dashboard for {mission_id}",
            'mission_id': mission_id,
            'theme': 'dark',
            'widgets': widgets
        }

    def _create_default_dashboard_config(self,
                                       mission_id: str,
                                       spacecraft_list: List[Spacecraft]) -> Dict[str, Any]:
        """Create default dashboard configuration"""
        return self.dashboard_service.create_default_layout(mission_id).to_dict()


def setup_dashboard_integration(app: FastAPI) -> DashboardIntegrationService:
    """
    Set up dashboard integration with a FastAPI application
    """

    # Create integration service
    integration_service = DashboardIntegrationService(app)

    # Include dashboard router
    app.include_router(dashboard_router)

    # Add startup and shutdown handlers
    @app.on_event("startup")
    async def startup_dashboard_integration():
        await integration_service.start()

    @app.on_event("shutdown")
    async def shutdown_dashboard_integration():
        await integration_service.stop()

    # Mount static files for dashboard frontend (if serving from same app)
    app.mount("/dashboard", StaticFiles(directory="src/services/dashboard-enhancement/static", html=True), name="dashboard")

    return integration_service


# Global integration service instance
dashboard_integration_service = DashboardIntegrationService()


# Export for use in main application
__all__ = [
    "DashboardIntegrationService",
    "dashboard_integration_service",
    "setup_dashboard_integration"
]
