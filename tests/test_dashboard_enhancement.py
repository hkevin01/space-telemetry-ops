"""
Tests for Mission Control Dashboard Enhancement

This module provides comprehensive tests for the enhanced mission control dashboard,
including unit tests for services, integration tests for WebSocket functionality,
and end-to-end tests for the complete dashboard system.

Test Coverage:
- Dashboard service functionality
- Widget configuration and management
- Real-time WebSocket updates
- Alert processing and display
- Performance optimization
- Integration with telemetry and anomaly detection
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

# Internal imports
from ..dashboard_service import (
    MissionControlDashboardService,
    DashboardLayout,
    DashboardWidget,
    ChartType,
    AggregationType,
    TelemetryDataAggregator,
    WebSocketManager
)
from ..api import router
from ..integration import DashboardIntegrationService
from ..config import dashboard_config, widget_config
from ...core.telemetry import TelemetryPacket
from ...core.models import Spacecraft
from ..anomaly_detection.anomaly_detection import AnomalyAlert, SeverityLevel


class TestDashboardWidget:
    """Test cases for DashboardWidget class"""

    def test_widget_creation(self):
        """Test creating a dashboard widget"""
        widget = DashboardWidget(
            widget_id="test_widget_1",
            title="Test Temperature",
            chart_type=ChartType.GAUGE,
            data_source="temperature",
            position_x=0,
            position_y=0,
            width=4,
            height=3
        )

        assert widget.widget_id == "test_widget_1"
        assert widget.title == "Test Temperature"
        assert widget.chart_type == ChartType.GAUGE
        assert widget.data_source == "temperature"
        assert widget.position_x == 0
        assert widget.position_y == 0
        assert widget.width == 4
        assert widget.height == 3

    def test_widget_to_dict(self):
        """Test widget serialization to dictionary"""
        widget = DashboardWidget(
            widget_id="test_widget_2",
            title="Test Pressure",
            chart_type=ChartType.LINE,
            data_source="pressure",
            warning_threshold=50.0,
            critical_threshold=80.0
        )

        widget_dict = widget.to_dict()

        assert widget_dict['widget_id'] == "test_widget_2"
        assert widget_dict['title'] == "Test Pressure"
        assert widget_dict['chart_type'] == "line"
        assert widget_dict['config']['warning_threshold'] == 50.0
        assert widget_dict['config']['critical_threshold'] == 80.0


class TestDashboardLayout:
    """Test cases for DashboardLayout class"""

    def test_layout_creation(self):
        """Test creating a dashboard layout"""
        layout = DashboardLayout(
            layout_id="test_layout_1",
            name="Test Mission Dashboard",
            description="Test dashboard for mission control",
            mission_id="MISSION_001"
        )

        assert layout.layout_id == "test_layout_1"
        assert layout.name == "Test Mission Dashboard"
        assert layout.mission_id == "MISSION_001"
        assert layout.widgets == []

    def test_layout_with_widgets(self):
        """Test layout with widgets"""
        widgets = [
            DashboardWidget(
                widget_id="widget_1",
                title="Temperature",
                chart_type=ChartType.GAUGE,
                data_source="temperature"
            ),
            DashboardWidget(
                widget_id="widget_2",
                title="Pressure",
                chart_type=ChartType.LINE,
                data_source="pressure"
            )
        ]

        layout = DashboardLayout(
            layout_id="test_layout_2",
            name="Test Layout with Widgets",
            description="Test layout",
            widgets=widgets
        )

        assert len(layout.widgets) == 2
        assert layout.widgets[0].widget_id == "widget_1"
        assert layout.widgets[1].widget_id == "widget_2"

    def test_layout_to_dict(self):
        """Test layout serialization to dictionary"""
        widget = DashboardWidget(
            widget_id="widget_1",
            title="Test Widget",
            chart_type=ChartType.STATUS,
            data_source="status"
        )

        layout = DashboardLayout(
            layout_id="test_layout_3",
            name="Test Serialization",
            description="Test layout serialization",
            widgets=[widget]
        )

        layout_dict = layout.to_dict()

        assert layout_dict['layout_id'] == "test_layout_3"
        assert layout_dict['name'] == "Test Serialization"
        assert len(layout_dict['widgets']) == 1
        assert layout_dict['widgets'][0]['widget_id'] == "widget_1"


class TestTelemetryDataAggregator:
    """Test cases for TelemetryDataAggregator class"""

    @pytest.fixture
    def aggregator(self):
        """Create a telemetry data aggregator for testing"""
        return TelemetryDataAggregator()

    @pytest.fixture
    def sample_telemetry_data(self):
        """Create sample telemetry data for testing"""
        base_time = datetime.now()
        data = []

        for i in range(10):
            packet = TelemetryPacket(
                vehicle_id="SAT_001",
                packet_id=f"packet_{i}",
                spacecraft_time=base_time - timedelta(minutes=i*5),
                ground_time=base_time - timedelta(minutes=i*5),
                payload={
                    "temperature": 20.0 + i,
                    "pressure": 1000.0 - i*10,
                    "altitude": 400000 + i*1000
                }
            )
            data.append(packet)

        return data

    @pytest.mark.asyncio
    async def test_get_widget_data_gauge(self, aggregator, sample_telemetry_data):
        """Test getting widget data for gauge chart"""
        widget = DashboardWidget(
            widget_id="temp_gauge",
            title="Temperature Gauge",
            chart_type=ChartType.GAUGE,
            data_source="temperature",
            time_window_hours=2
        )

        result = await aggregator.get_widget_data(widget, sample_telemetry_data)

        assert result['widget_id'] == "temp_gauge"
        assert result['chart_type'] == "gauge"
        assert 'data' in result
        assert 'statistics' in result
        assert result['statistics']['current_value'] == 20.0  # First (most recent) value

    @pytest.mark.asyncio
    async def test_get_widget_data_line(self, aggregator, sample_telemetry_data):
        """Test getting widget data for line chart"""
        widget = DashboardWidget(
            widget_id="temp_line",
            title="Temperature Trend",
            chart_type=ChartType.LINE,
            data_source="temperature",
            time_window_hours=2
        )

        result = await aggregator.get_widget_data(widget, sample_telemetry_data)

        assert result['widget_id'] == "temp_line"
        assert result['chart_type'] == "line"
        assert 'datasets' in result['data']
        assert len(result['data']['datasets']) > 0
        assert result['data']['datasets'][0]['label'] == "Temperature Trend"

    @pytest.mark.asyncio
    async def test_get_widget_data_status(self, aggregator, sample_telemetry_data):
        """Test getting widget data for status indicator"""
        widget = DashboardWidget(
            widget_id="temp_status",
            title="Temperature Status",
            chart_type=ChartType.STATUS,
            data_source="temperature",
            warning_threshold=25.0,
            critical_threshold=30.0
        )

        result = await aggregator.get_widget_data(widget, sample_telemetry_data)

        assert result['widget_id'] == "temp_status"
        assert result['chart_type'] == "status"
        assert 'status' in result['data']
        assert result['data']['status'] == 'normal'  # 20.0 < 25.0 (warning)

    @pytest.mark.asyncio
    async def test_get_widget_data_no_data(self, aggregator):
        """Test getting widget data when no telemetry data is available"""
        widget = DashboardWidget(
            widget_id="empty_widget",
            title="Empty Widget",
            chart_type=ChartType.LINE,
            data_source="nonexistent"
        )

        result = await aggregator.get_widget_data(widget, [])

        assert result['widget_id'] == "empty_widget"
        assert result['statistics']['data_points'] == 0
        assert 'message' in result

    @pytest.mark.asyncio
    async def test_spacecraft_filtering(self, aggregator):
        """Test filtering telemetry data by spacecraft ID"""
        # Create data for multiple spacecraft
        base_time = datetime.now()
        data = []

        for spacecraft in ["SAT_001", "SAT_002"]:
            for i in range(5):
                packet = TelemetryPacket(
                    vehicle_id=spacecraft,
                    packet_id=f"{spacecraft}_packet_{i}",
                    spacecraft_time=base_time - timedelta(minutes=i*5),
                    ground_time=base_time - timedelta(minutes=i*5),
                    payload={"temperature": 20.0 + (0 if spacecraft == "SAT_001" else 10)}
                )
                data.append(packet)

        # Test widget with spacecraft filter
        widget = DashboardWidget(
            widget_id="sat_001_temp",
            title="SAT_001 Temperature",
            chart_type=ChartType.GAUGE,
            data_source="temperature",
            spacecraft_id="SAT_001"
        )

        result = await aggregator.get_widget_data(widget, data)

        # Should only get data from SAT_001 (temperature = 20.0)
        assert result['statistics']['current_value'] == 20.0


class TestWebSocketManager:
    """Test cases for WebSocketManager class"""

    @pytest.fixture
    def ws_manager(self):
        """Create a WebSocket manager for testing"""
        return WebSocketManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket for testing"""
        websocket = Mock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    async def test_websocket_connect(self, ws_manager, mock_websocket):
        """Test WebSocket connection handling"""
        connection_id = "test_conn_1"

        await ws_manager.connect(mock_websocket, connection_id)

        assert connection_id in ws_manager.active_connections
        assert ws_manager.active_connections[connection_id] == mock_websocket
        assert connection_id in ws_manager.connection_subscriptions
        mock_websocket.accept.assert_called_once()

    def test_websocket_disconnect(self, ws_manager, mock_websocket):
        """Test WebSocket disconnection handling"""
        connection_id = "test_conn_2"

        # Manually add connection
        ws_manager.active_connections[connection_id] = mock_websocket
        ws_manager.connection_subscriptions[connection_id] = set()

        ws_manager.disconnect(connection_id)

        assert connection_id not in ws_manager.active_connections
        assert connection_id not in ws_manager.connection_subscriptions

    @pytest.mark.asyncio
    async def test_widget_subscription(self, ws_manager, mock_websocket):
        """Test widget subscription management"""
        connection_id = "test_conn_3"
        widget_id = "test_widget"

        # Setup connection
        await ws_manager.connect(mock_websocket, connection_id)

        # Subscribe to widget
        await ws_manager.subscribe_to_widget(connection_id, widget_id)

        assert widget_id in ws_manager.connection_subscriptions[connection_id]

        # Unsubscribe from widget
        await ws_manager.unsubscribe_from_widget(connection_id, widget_id)

        assert widget_id not in ws_manager.connection_subscriptions[connection_id]

    @pytest.mark.asyncio
    async def test_broadcast_widget_update(self, ws_manager, mock_websocket):
        """Test broadcasting widget updates to subscribed connections"""
        connection_id = "test_conn_4"
        widget_id = "test_widget"
        test_data = {"value": 42, "timestamp": datetime.now().isoformat()}

        # Setup connection and subscription
        await ws_manager.connect(mock_websocket, connection_id)
        await ws_manager.subscribe_to_widget(connection_id, widget_id)

        # Broadcast update
        await ws_manager.broadcast_widget_update(widget_id, test_data)

        # Verify message was sent
        mock_websocket.send_text.assert_called_once()
        sent_message = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_message['type'] == 'widget_update'
        assert sent_message['widget_id'] == widget_id
        assert sent_message['data'] == test_data

    @pytest.mark.asyncio
    async def test_broadcast_system_alert(self, ws_manager, mock_websocket):
        """Test broadcasting system alerts to all connections"""
        connection_id = "test_conn_5"

        # Setup connection
        await ws_manager.connect(mock_websocket, connection_id)

        # Create test alert
        alert = AnomalyAlert(
            id="test_alert",
            parameter_name="temperature",
            spacecraft_id="SAT_001",
            severity=SeverityLevel.WARNING,
            message="Test alert message",
            timestamp=datetime.now(),
            value=75.0,
            threshold=70.0
        )

        # Broadcast alert
        await ws_manager.broadcast_system_alert(alert)

        # Verify message was sent
        mock_websocket.send_text.assert_called_once()
        sent_message = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_message['type'] == 'system_alert'
        assert 'alert' in sent_message


class TestMissionControlDashboardService:
    """Test cases for MissionControlDashboardService class"""

    @pytest.fixture
    def dashboard_service(self):
        """Create a dashboard service for testing"""
        return MissionControlDashboardService()

    def test_create_default_layout(self, dashboard_service):
        """Test creating a default dashboard layout"""
        mission_id = "TEST_MISSION_001"

        layout = dashboard_service.create_default_layout(mission_id)

        assert layout.layout_id == f"default_{mission_id}"
        assert layout.mission_id == mission_id
        assert len(layout.widgets) > 0

        # Check that default widgets are created
        widget_ids = [w.widget_id for w in layout.widgets]
        assert "temp_gauge" in widget_ids
        assert "pressure_line" in widget_ids
        assert "power_status" in widget_ids
        assert "altitude_scatter" in widget_ids

    def test_save_and_get_layout(self, dashboard_service):
        """Test saving and retrieving dashboard layouts"""
        mission_id = "TEST_MISSION_002"
        layout = dashboard_service.create_default_layout(mission_id)

        # Save layout
        success = dashboard_service.save_layout(layout)
        assert success is True

        # Retrieve layout
        retrieved_layout = dashboard_service.get_layout(layout.layout_id)
        assert retrieved_layout is not None
        assert retrieved_layout.layout_id == layout.layout_id
        assert retrieved_layout.name == layout.name

    def test_delete_layout(self, dashboard_service):
        """Test deleting dashboard layouts"""
        mission_id = "TEST_MISSION_003"
        layout = dashboard_service.create_default_layout(mission_id)

        # Save and then delete layout
        dashboard_service.save_layout(layout)
        success = dashboard_service.delete_layout(layout.layout_id)
        assert success is True

        # Verify layout is deleted
        retrieved_layout = dashboard_service.get_layout(layout.layout_id)
        assert retrieved_layout is None

    def test_list_layouts(self, dashboard_service):
        """Test listing dashboard layouts"""
        # Create multiple layouts
        layouts = []
        for i in range(3):
            mission_id = f"TEST_MISSION_00{i+4}"
            layout = dashboard_service.create_default_layout(mission_id)
            dashboard_service.save_layout(layout)
            layouts.append(layout)

        # List all layouts
        all_layouts = dashboard_service.list_layouts()
        assert len(all_layouts) >= 3

        # List layouts for specific mission
        specific_layouts = dashboard_service.list_layouts(mission_id="TEST_MISSION_004")
        assert len(specific_layouts) == 1
        assert specific_layouts[0].mission_id == "TEST_MISSION_004"


class TestDashboardIntegrationService:
    """Test cases for DashboardIntegrationService class"""

    @pytest.fixture
    def integration_service(self):
        """Create a dashboard integration service for testing"""
        return DashboardIntegrationService()

    @pytest.mark.asyncio
    async def test_initialization(self, integration_service):
        """Test service initialization"""
        mock_anomaly_service = Mock()
        mock_performance_service = Mock()
        mock_telemetry_processor = Mock()

        await integration_service.initialize(
            anomaly_service=mock_anomaly_service,
            performance_service=mock_performance_service,
            telemetry_processor=mock_telemetry_processor
        )

        assert integration_service.anomaly_service == mock_anomaly_service
        assert integration_service.performance_service == mock_performance_service
        assert integration_service.telemetry_processor == mock_telemetry_processor

    @pytest.mark.asyncio
    async def test_start_stop_service(self, integration_service):
        """Test starting and stopping the integration service"""
        # Start service
        await integration_service.start()
        assert integration_service.running is True
        assert len(integration_service.background_tasks) > 0

        # Stop service
        await integration_service.stop()
        assert integration_service.running is False
        assert len(integration_service.background_tasks) == 0

    @pytest.mark.asyncio
    async def test_create_mission_dashboard(self, integration_service):
        """Test creating mission-specific dashboards"""
        mission_id = "INTEGRATION_TEST_001"
        mission_type = "satellite"

        # Create test spacecraft
        spacecraft_list = [
            Spacecraft(vehicle_id="SAT_001", name="Test Satellite 1"),
            Spacecraft(vehicle_id="SAT_002", name="Test Satellite 2")
        ]

        # Create mission dashboard
        layout = await integration_service.create_mission_dashboard(
            mission_id=mission_id,
            mission_type=mission_type,
            spacecraft_list=spacecraft_list
        )

        assert layout.mission_id == mission_id
        assert "satellite" in layout.name.lower()
        assert len(layout.widgets) > 0

        # Verify spacecraft-specific widgets were created
        spacecraft_widget_ids = [w.spacecraft_id for w in layout.widgets if w.spacecraft_id]
        assert "SAT_001" in spacecraft_widget_ids
        assert "SAT_002" in spacecraft_widget_ids


class TestDashboardAPI:
    """Test cases for Dashboard API endpoints"""

    @pytest.fixture
    def client(self):
        """Create a test client for the dashboard API"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_create_dashboard_layout(self, client):
        """Test creating dashboard layout via API"""
        layout_data = {
            "name": "API Test Dashboard",
            "description": "Test dashboard created via API",
            "mission_id": "API_TEST_001",
            "theme": "dark"
        }

        response = client.post("/api/dashboard/layouts", json=layout_data)

        assert response.status_code == 200
        result = response.json()
        assert result['name'] == layout_data['name']
        assert result['mission_id'] == layout_data['mission_id']
        assert 'layout_id' in result

    def test_list_dashboard_layouts(self, client):
        """Test listing dashboard layouts via API"""
        response = client.get("/api/dashboard/layouts")

        assert response.status_code == 200
        layouts = response.json()
        assert isinstance(layouts, list)

    def test_get_dashboard_layout(self, client):
        """Test retrieving specific dashboard layout via API"""
        # First create a layout
        layout_data = {
            "name": "Get Test Dashboard",
            "description": "Test dashboard for GET endpoint",
            "mission_id": "GET_TEST_001"
        }

        create_response = client.post("/api/dashboard/layouts", json=layout_data)
        created_layout = create_response.json()
        layout_id = created_layout['layout_id']

        # Now retrieve it
        response = client.get(f"/api/dashboard/layouts/{layout_id}")

        assert response.status_code == 200
        layout = response.json()
        assert layout['layout_id'] == layout_id
        assert layout['name'] == layout_data['name']

    def test_add_widget_to_layout(self, client):
        """Test adding widget to layout via API"""
        # First create a layout
        layout_data = {
            "name": "Widget Test Dashboard",
            "description": "Test dashboard for widget operations"
        }

        create_response = client.post("/api/dashboard/layouts", json=layout_data)
        layout_id = create_response.json()['layout_id']

        # Add widget
        widget_data = {
            "title": "Test Temperature Widget",
            "chart_type": "gauge",
            "data_source": "temperature",
            "position_x": 0,
            "position_y": 0,
            "width": 4,
            "height": 3,
            "warning_threshold": 50.0,
            "critical_threshold": 80.0
        }

        response = client.post(f"/api/dashboard/layouts/{layout_id}/widgets", json=widget_data)

        assert response.status_code == 200
        result = response.json()
        assert 'widget' in result
        assert result['widget']['title'] == widget_data['title']

    def test_create_template_layout(self, client):
        """Test creating template layout via API"""
        mission_type = "satellite"

        response = client.post(f"/api/dashboard/templates/{mission_type}")

        assert response.status_code == 200
        template = response.json()
        assert "satellite" in template['name'].lower()
        assert len(template['widgets']) > 0


@pytest.mark.integration
class TestDashboardIntegration:
    """Integration tests for the complete dashboard system"""

    @pytest.fixture
    def app(self):
        """Create a FastAPI app with dashboard integration"""
        from fastapi import FastAPI
        from ..integration import setup_dashboard_integration

        app = FastAPI()
        integration_service = setup_dashboard_integration(app)
        return app, integration_service

    @pytest.mark.asyncio
    async def test_end_to_end_dashboard_creation(self, app):
        """Test complete dashboard creation and operation"""
        app_instance, integration_service = app

        # Initialize services
        await integration_service.initialize()
        await integration_service.start()

        try:
            # Create mission dashboard
            mission_id = "E2E_TEST_001"
            spacecraft_list = [
                Spacecraft(vehicle_id="SAT_E2E_001", name="E2E Test Satellite")
            ]

            layout = await integration_service.create_mission_dashboard(
                mission_id=mission_id,
                mission_type="satellite",
                spacecraft_list=spacecraft_list
            )

            # Verify layout was created
            assert layout.layout_id is not None
            assert layout.mission_id == mission_id
            assert len(layout.widgets) > 0

            # Test telemetry data integration
            test_packet = TelemetryPacket(
                vehicle_id="SAT_E2E_001",
                packet_id="e2e_test_packet",
                spacecraft_time=datetime.now(),
                ground_time=datetime.now(),
                payload={"temperature": 25.5, "altitude": 450000}
            )

            # Simulate telemetry data processing
            integration_service.telemetry_cache["SAT_E2E_001"] = [test_packet]

            # Update dashboard with new data
            await integration_service._update_layout_widgets(layout)

            # Verify data was processed
            assert "SAT_E2E_001" in integration_service.telemetry_cache
            assert len(integration_service.telemetry_cache["SAT_E2E_001"]) == 1

        finally:
            await integration_service.stop()

    @pytest.mark.asyncio
    async def test_websocket_real_time_updates(self, app):
        """Test real-time WebSocket updates"""
        app_instance, integration_service = app

        await integration_service.initialize()
        await integration_service.start()

        try:
            # Create test layout
            layout = integration_service.dashboard_service.create_default_layout("WS_TEST")
            integration_service.dashboard_service.save_layout(layout)

            # Create mock WebSocket connection
            mock_websocket = Mock()
            mock_websocket.accept = AsyncMock()
            mock_websocket.send_text = AsyncMock()

            ws_manager = integration_service.dashboard_service.websocket_manager

            # Connect WebSocket
            connection_id = "test_ws_conn"
            await ws_manager.connect(mock_websocket, connection_id)

            # Subscribe to widget updates
            widget_id = layout.widgets[0].widget_id
            await ws_manager.subscribe_to_widget(connection_id, widget_id)

            # Simulate widget update
            test_data = {"value": 42.5, "timestamp": datetime.now().isoformat()}
            await ws_manager.broadcast_widget_update(widget_id, test_data)

            # Verify WebSocket message was sent
            mock_websocket.send_text.assert_called()

        finally:
            await integration_service.stop()


@pytest.mark.performance
class TestDashboardPerformance:
    """Performance tests for dashboard components"""

    @pytest.mark.asyncio
    async def test_widget_update_performance(self):
        """Test performance of widget updates with large datasets"""
        aggregator = TelemetryDataAggregator()

        # Create large dataset (1000 data points)
        large_dataset = []
        base_time = datetime.now()

        for i in range(1000):
            packet = TelemetryPacket(
                vehicle_id="PERF_SAT_001",
                packet_id=f"perf_packet_{i}",
                spacecraft_time=base_time - timedelta(seconds=i*10),
                ground_time=base_time - timedelta(seconds=i*10),
                payload={"temperature": 20.0 + (i % 50)}
            )
            large_dataset.append(packet)

        # Create widget
        widget = DashboardWidget(
            widget_id="perf_test_widget",
            title="Performance Test",
            chart_type=ChartType.LINE,
            data_source="temperature",
            time_window_hours=24
        )

        # Measure processing time
        import time
        start_time = time.time()

        result = await aggregator.get_widget_data(widget, large_dataset)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance (should process 1000 points in under 1 second)
        assert processing_time < 1.0
        assert result['statistics']['data_points'] > 0
        assert len(result['data']['datasets'][0]['data']) > 0

    @pytest.mark.asyncio
    async def test_websocket_broadcast_performance(self):
        """Test performance of WebSocket broadcasting to many connections"""
        ws_manager = WebSocketManager()

        # Create multiple mock connections
        connections = []
        for i in range(100):  # 100 concurrent connections
            mock_ws = Mock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_text = AsyncMock()

            connection_id = f"perf_conn_{i}"
            await ws_manager.connect(mock_ws, connection_id)
            await ws_manager.subscribe_to_widget(connection_id, "test_widget")
            connections.append((connection_id, mock_ws))

        # Measure broadcast time
        import time
        start_time = time.time()

        test_data = {"value": 123.45}
        await ws_manager.broadcast_widget_update("test_widget", test_data)

        end_time = time.time()
        broadcast_time = end_time - start_time

        # Verify performance (should broadcast to 100 connections in under 0.5 seconds)
        assert broadcast_time < 0.5

        # Verify all connections received the message
        for connection_id, mock_ws in connections:
            mock_ws.send_text.assert_called()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
